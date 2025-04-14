from PIL import Image, ImageDraw, ImageFont

# Define image dimensions
width, height = 512, 512

# Create a new RGB image with a white background
image = Image.new('RGB', (width, height), color='white')

# Prepare drawing context
draw = ImageDraw.Draw(image)

# Define the text to display
text = "TUM"

# Load a basic font (default font)
font = ImageFont.load_default(200)

# Calculate the width and height of the text to center it
bbox = draw.textbbox((0, 0), text, font=font)
text_width = bbox[2] - bbox[0]
text_height = bbox[3] - bbox[1]

x = (width - text_width) // 2
y = (height - text_height) // 2

# Draw the text in black
draw.text((x, y), text, fill='black', font=font)

# Save the image as a JPEG file
image.save(f"./test_img/binary_image_{text}.jpg", "JPEG")

