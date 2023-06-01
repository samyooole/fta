import pytesseract as pyt
import cv2


import os
from PIL import Image
from pdf2image import convert_from_path
import pytesseract

filePath = 'working_pickles/AHKIA-Agreement_on_Investment.pdf'
doc = convert_from_path(filePath)
path, fileName = os.path.split(filePath)
fileBaseName, fileExtension = os.path.splitext(fileName)

for page_number, page_data in enumerate(doc):
    txt = pytesseract.image_to_string(Image.fromarray(page_data)).encode("utf-8")
    print("Page # {} - {}".format(str(page_number),txt))