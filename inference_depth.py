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
        print('Image is grayscale, stack the channels!')
        img = np.stack([img, img, img], axis=-1)
    img = (img.astype(np.float32) / 127.5) - 1.0           # -1 ~ 1
    img_tensor = torch.from_numpy(img).permute(2, 0, 1)[None, ...].cuda()   # 1, 3, 512, 512
    if add_noise:
        noise = (torch.rand_like(img_tensor) - 0.5) / 0.5      # -1 ~ 1
        img_tensor = (1 - noise_value) * img_tensor + noise_value * noise
    return img_tensor


def load_ref_img_grayscale(img_path, add_noise=False, noise_value=0.05):
    img = Image.open(img_path).resize((H, W))
    img = np.array(img)
    if len(img.shape) == 2:
        print('Image is grayscale, stack the channels!')
        img = np.stack([img, img, img], axis=-1)
    img = (img.astype(np.float32) / 127.5) - 1.0           # -1 ~ 1
    img_tensor = torch.from_numpy(img).permute(2, 0, 1)[None, ...].cuda()   # 1, 3, 512, 512
    if add_noise:
        noise = (torch.rand_like(img_tensor) - 0.5) / 0.5      # -1 ~ 1
        img_tensor = (1 - noise_value) * img_tensor + noise_value * noise
    return img_tensor


def inversion(img_tensor, uncond, diffusion_model):
    if os.path.exists('latent.py'):
        os.remove('latent.py')
    encoder_posterior = model.encode_first_stage(img_tensor) # encode the image
    z = model.get_first_stage_encoding(encoder_posterior).detach() # .sample * scale_factor
    print('z shape:', z.shape) #TODO same with the diffusers version, then the problem comes from sampler.encode step.
    print('z min:', z.min())
    print('z max:', z.max())
    print('z mean:', z.mean())
    sampler.make_schedule(ddim_num_steps=encode_steps) # set ddim steps
    un_cond = {"c_crossattn": [model.get_learned_conditioning([''])]} 
    # latent, out = sampler.encode(x0=z, cond=un_cond, t_enc=encode_steps)
    # latent = sampler.encode_simple(x0=z, cond=un_cond, t_enc=encode_steps)
    latent = sampler.encode_diffusers(diffusion_model.to("cuda", torch.float16), img_tensor.to("cuda", torch.float16), uncond, encode_steps)
    print(latent.shape) # [1, 4, 64, 64]
    torch.save(latent, 'latent.pt')
    return latent


def load_inverted_noise():
    return torch.load('latent.pt').cuda().to(torch.float32)


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


# prompt = 'ancient ruins'
# prompt = 'modern building'
# prompt = 'sky'
prompt = 'a photo of a japanese style living room'
# prompt = 'a photo of a scientific style living room'

def encode_prompt_with_a_prompt_and_null(batch_size, prompt, tokenizer, text_encoder, device, particle_num_vsd):
    text_input = tokenizer(
        [prompt], padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt"
    )
    with torch.no_grad():
        text_embeddings = text_encoder(text_input.input_ids.to(device))[0]

    max_length = text_input.input_ids.shape[-1]
    uncond_input = tokenizer(
        [""] * batch_size, padding="max_length", max_length=max_length, truncation=True, return_tensors="pt"
    )
    with torch.no_grad():
        uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]

    return torch.cat([uncond_embeddings[:particle_num_vsd], text_embeddings[:particle_num_vsd]])

batch_size = 1
text_embeddings_2 = encode_prompt_with_a_prompt_and_null(batch_size, prompt, diffusion_model.tokenizer, diffusion_model.text_encoder, device, 1)
uncond, cond = text_embeddings_2.chunk(2)

# load a reference image and run inversion
image_name = 'face1.jpg'
# image_name = 'face2.jpg'
# image_name = 'tum_white.png' # good results
# image_name = 'binary_image_TUM.jpg'
# image_name = 'black_dog.jpg'
# image_name = 'yellow_dog.jpg'
# image_name = 'depth_scene.png'
# image_name = 'binary_image_TUM.png'

save_image_name = image_name.replace('.', '_')

image_path = 'test_img/' + image_name

output_dir = './outputs/depth'
os.makedirs(output_dir, exist_ok=True)

contrast = 2 # default value for face1 and face2
# contrast = 1
# contrast = 3

latents_ref_inversion = inversion(load_ref_img(image_path, contrast=contrast, add_noise=False), uncond, diffusion_model)
# print('latents_ref_inversion shape:', latents_ref_inversion.shape) # (1, 4, 64, 64)
# print('latents_ref_inversion min:', latents_ref_inversion.min())
# print('latents_ref_inversion max:', latents_ref_inversion.max())
# print('latents_ref_inversion mean:', latents_ref_inversion.mean())

# inversion(load_ref_img(image_path, contrast=contrast, add_noise=True))
# inversion(load_ref_img_grayscale(image_path, add_noise=True)) # need to add noise to prevent poor results, since the text are too sharp contrast / structural information

# save latents_ref_inversion using diffusers
ref_sample = 1 / diffusion_model.vae.config.scaling_factor * latents_ref_inversion.clone().to(torch.float16).detach()
# print('ref_sample shape:', ref_sample.shape) # (1, 4, 64, 64)
# print('ref_sample min:', ref_sample.min())
# print('ref_sample max:', ref_sample.max())
# print('ref_sample mean:', ref_sample.mean())
ref_image_ = diffusion_model.vae.decode(ref_sample).sample.to(dtype_half)
# print('ref_image_ shape:', ref_image_.shape) # (1, 4, 64, 64)
# print('ref_image_ min:', ref_image_.min())
# print('ref_image_ max:', ref_image_.max())
# print('ref_image_ mean:', ref_image_.mean())
save_image((ref_image_ / 2 + 0.5).clamp(0, 1), f'{output_dir}/test_{save_image_name}.png')

# save ref image using the original model (the same with the first one)
ref_sample = torch.clip(model.decode_first_stage(latents_ref_inversion.to(torch.float32)), min=-1, max=1).squeeze()
ref_sample = (einops.rearrange(ref_sample, 'c h w -> h w c') * 127.5 + 127.5).cpu().numpy().astype(np.uint8)
ref_sample = Image.fromarray(ref_sample)
ref_sample.save(output_dir + f'/test_{save_image_name}_ref_sample.jpg')

direct_transfer_steps = 40
decayed_transfer_steps = 20 # default: 20

depth_image = load_image('./test_img/depth_scene.png')
# depth_image = load_image('./test_img/depth_tensor.png')
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
exponent = 1.0 # default: 0.5
set_random_seed(6000)
sample = sample_illusion_image_depth(latent=load_inverted_noise(), text_prompt=prompt, text_embeddings=text_embeddings, diffusion_model=diffusion_model, controlnet=controlnet, controlnet_cond_input=controlnet_cond_input, direct_transfer_steps=direct_transfer_steps, decayed_transfer_steps=decayed_transfer_steps, exponent=exponent)
sample = Image.fromarray(sample)
sample.save(output_dir + f'/sample_{prompt}_test_{save_image_name}_contrast_{contrast}.jpg')



