import pytesseract
from PIL import Image

image=Image.open('./schema.png')
text=pytesseract.image_to_string(image)
print(text)