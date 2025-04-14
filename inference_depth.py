import einops
import numpy as np
import random
import torch
from PIL import Image
import os
from pytorch_lightning import seed_everything
from PTDiffusion.tools import create_model, load_state_dict
from PTDiffusion.phase_guided_sampler import Phase_Guided_Sampler
import torchvision.transforms
from torchvision.utils import save_image
import torchvision.transforms.functional as TF

from diffusers import ControlNetModel, StableDiffusionControlNetPipeline
from diffusers.utils import load_image


os.environ["CUDA_VISIBLE_DEVICES"] = '0' # must change this line to make it work, originally set as 1
# resolution of the generated image
H = W = 512
# guidance scale of the classifier-free guidance
unconditional_guidance_scale = 7.5
# set inversion steps
encode_steps = 1000
# set sampling steps
decode_steps = 100

# depth condition
render_size = H
dtype_half = torch.float16
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

controlnet_name = "lllyasviel/control_v11f1p_sd15_depth"
controlnet = ControlNetModel.from_pretrained(controlnet_name, torch_dtype=dtype_half, low_cpu_mem_usage=True).to(device)
model_path = "runwayml/stable-diffusion-v1-5"
diffusion_model = StableDiffusionControlNetPipeline.from_pretrained(model_path, controlnet=controlnet, torch_dtype=dtype_half, low_cpu_mem_usage=True).to(device)
diffusion_model.enable_model_cpu_offload()

controlnet = diffusion_model.controlnet.to(dtype_half)
controlnet.requires_grad_(False)
controlnet.enable_gradient_checkpointing()

# load the model
model = create_model('models/model_ldm_v15.yaml').cuda()
model.load_state_dict(load_state_dict('models/v1-5-pruned-emaonly.ckpt', location='cuda'), strict=False)
sampler = Phase_Guided_Sampler(model)


def set_random_seed(seed):
    if seed == -1:
        seed = random.randint(0, 65535)
    seed_everything(seed)


def load_ref_img(img_path, contrast=2., add_noise=False, noise_value=0.05):
    img = Image.open(img_path).convert('RGB').resize((H, W))
    img = torchvision.transforms.ColorJitter(contrast=(contrast, contrast))(img)
    img = np.array(img)
    if len(img.shape) == 2:
        img = np.stack([img, img, img], axis=-1)
    img = (img.astype(np.float32) / 127.5) - 1.0           # -1 ~ 1
    img_tensor = torch.from_numpy(img).permute(2, 0, 1)[None, ...].cuda()   # 1, 3, 512, 512
    if add_noise:
        noise = (torch.rand_like(img_tensor) - 0.5) / 0.5      # -1 ~ 1
        img_tensor = (1 - noise_value) * img_tensor + noise_value * noise
    return img_tensor


def inversion(img_tensor):
    if os.path.exists('latent.py'):
        os.remove('latent.py')
    encoder_posterior = model.encode_first_stage(img_tensor)
    z = model.get_first_stage_encoding(encoder_posterior).detach()
    sampler.make_schedule(ddim_num_steps=encode_steps)
    un_cond = {"c_crossattn": [model.get_learned_conditioning([''])]}
    latent, out = sampler.encode(x0=z, cond=un_cond, t_enc=encode_steps)
    torch.save(latent, 'latent.pt')


def load_inverted_noise():
    return torch.load('latent.pt').cuda()


def sample_illusion_image_depth(latent, text_prompt, text_embeddings, diffusion_model, controlnet, controlnet_cond_input, decode_steps=100, direct_transfer_steps=60, decayed_transfer_steps=0,
                          async_ahead_steps=0, exponent=0.5):
    un_cond = {"c_crossattn": [model.get_learned_conditioning([''])]}
    cond = {"c_crossattn": [model.get_learned_conditioning([text_prompt])]}
    sampler.make_schedule(ddim_num_steps=decode_steps)
    x_rec = sampler.decode_with_phase_substitution_depth(text_embeddings=text_embeddings, diffusion_model=diffusion_model, controlnet=controlnet, controlnet_cond_input=controlnet_cond_input, ref_latent=latent, cond=cond, t_dec=decode_steps,
                                                   unconditional_guidance_scale=unconditional_guidance_scale,
                                                   unconditional_conditioning=un_cond, direct_transfer_steps=direct_transfer_steps,
                                                   blending_ratio=0,
                                                   decayed_transfer_steps=decayed_transfer_steps, async_ahead_steps=async_ahead_steps,
                                                   exponent=exponent)
    x_sample = torch.clip(model.decode_first_stage(x_rec), min=-1, max=1).squeeze()
    x_sample = (einops.rearrange(x_sample, 'c h w -> h w c') * 127.5 + 127.5).cpu().numpy().astype(np.uint8)
    return x_sample


def encode_prompt(batch_size, prompt, tokenizer, text_encoder, device):
    text_input = tokenizer([prompt], padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
    with torch.no_grad():
        text_embeddings = text_encoder(text_input.input_ids.to(device))[0]

    max_length = text_input.input_ids.shape[-1]
    uncond_input = tokenizer(
        [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
    )
    with torch.no_grad():
        uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]

    return torch.cat([uncond_embeddings, text_embeddings])


# load a reference image and run inversion
image_name = 'face1.jpg'
# image_name = 'face2.jpg'
# image_name = 'tum_white.png' # good results
# image_name = 'binary_image_TUM.jpg'
# image_name = 'black_dog.jpg'
# image_name = 'yellow_dog.jpg'
# image_name = 'depth_scene.png'

image_path = 'test_img/' + image_name

contrast = 2 # default value for face1 and face2
# contrast = 1
# contrast = 3

inversion(load_ref_img(image_path, contrast=contrast, add_noise=False))

# prompt = 'ancient ruins'
# prompt = 'modern building'
# prompt = 'sky'
prompt = 'a photo of a japanese style living room'

output_dir = './outputs/depth'
os.makedirs(output_dir, exist_ok=True)
save_image_name = image_name.replace('.', '_')

depth_image = load_image('./test_img/depth_scene.png')
depth_image = depth_image.resize((render_size, render_size), resample=0)
depth_tensor = TF.to_tensor(depth_image).unsqueeze(0).to(device).to(dtype_half)

save_image((depth_tensor / 2 + 0.5).clamp(0, 1), output_dir + f'/depth_tensor.png')

# components
tokenizer = diffusion_model.tokenizer
text_encoder = diffusion_model.text_encoder
text_encoder = text_encoder.to(device).requires_grad_(False)

batch_size = 1 
text_embeddings = encode_prompt(batch_size, prompt, tokenizer, text_encoder, device)
unet_cross_attention_kwargs = {'scale': 0}
controlnet_cond_input = torch.cat([depth_tensor] * 2)

# generate illusion picture
set_random_seed(6000)
sample = sample_illusion_image_depth(latent=load_inverted_noise(), text_prompt=prompt, text_embeddings=text_embeddings, diffusion_model=diffusion_model, controlnet=controlnet, controlnet_cond_input=controlnet_cond_input, direct_transfer_steps=40, decayed_transfer_steps=20)
sample = Image.fromarray(sample)
sample.save(output_dir + f'/sample_{prompt}_test_{save_image_name}_contrast_{contrast}.jpg')



