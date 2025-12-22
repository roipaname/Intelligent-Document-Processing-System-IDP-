import pytesseract
from PIL import Image
import numpy as np
import cv2
from pdf2image import convert_from_path
from typing import List, Tuple,Dict,Optional
import logging

logging.basicConfig(level=logging.INFO)
logger= logging.getLogger(__name__)


class OCRResult:
    def __init__(self,text:str,confidence:float, bounding_boxes:List[Dict],page:int=1):
        self.text=text
        self.confidence=confidence
        self.bounding_boxes=bounding_boxes
        self.page=page
    def to_dict(self)->Dict:
        return {
            "text":self.text,
            "confidence":self.confidence,
            "bounding_boxes":self.bounding_boxes,
            "page":self.page,
            "word_count":len(self.text.split())
        }




