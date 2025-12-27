import pytesseract
from PIL import Image

image=Image.open('./schema.png')
text=pytesseract.image_to_string(image)
#this is  a test
print(text)