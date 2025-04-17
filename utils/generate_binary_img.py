from PIL import Image, ImageDraw, ImageFont

# Image dimensions
width, height = 512, 512

# 1-bit (binary) image: mode '1', background=1 (white)
img = Image.new('1', (width, height), color=1)

draw = ImageDraw.Draw(img)

# Text to draw
text = "TUM"

# Load a default font (you can also use a TTF file via ImageFont.truetype)
font = ImageFont.load_default(200)

# Compute text bounding box for centering
bbox = draw.textbbox((0, 0), text, font=font)
text_w = bbox[2] - bbox[0]
text_h = bbox[3] - bbox[1]

x = (width  - text_w) // 2
y = (height - text_h) // 2

# Draw text in black (0)
draw.text((x, y), text, fill=0, font=font)

# Save as PNG (JPEG doesn't support 1‑bit mode)
img.save(f"./test_img/binary_image_{text}.png")
