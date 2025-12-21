import pytesseract
from pdf2image import convert_from_path
from PIL import Image, ImageEnhance
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OCRResult:
    """Container for OCR results"""
    def __init__(self, text: str, confidence: float, bounding_boxes: List[Dict], page: int = 1):
        self.text = text
        self.confidence = confidence
        self.bounding_boxes = bounding_boxes
        self.page = page
    
    def to_dict(self):
        return {
            "page": self.page,
            "text": self.text,
            "confidence": self.confidence,
            "word_count": len(self.text.split()),
            "bounding_boxes": self.bounding_boxes
        }


class TesseractOCR:
    """
    Advanced OCR engine using Tesseract with image preprocessing
    
    Features:
    - Multi-page PDF support
    - Image enhancement (denoising, deskewing, contrast)
    - Confidence scoring per word
    - Bounding box extraction for layout analysis
    """
    
    def __init__(
        self,
        tesseract_cmd: Optional[str] = None,
        language: str = "eng",
        dpi: int = 300,
        psm: int = 6  # Page segmentation mode
    ):
        """
        Initialize OCR engine
        
        Args:
            tesseract_cmd: Path to tesseract binary (auto-detected if None)
            language: OCR language (eng, fra, deu, etc.)
            dpi: DPI for PDF to image conversion
            psm: Page segmentation mode
                0 = Orientation and script detection (OSD) only
                1 = Automatic page segmentation with OSD
                3 = Fully automatic page segmentation, but no OSD (Default)
                4 = Assume a single column of text of variable sizes
                5 = Assume a single uniform block of vertically aligned text
                6 = Assume a single uniform block of text
                7 = Treat the image as a single text line
                11 = Sparse text. Find as much text as possible in no particular order
        """
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        
        self.language = language
        self.dpi = dpi
        self.psm = psm
        self.config = f'--oem 3 --psm {psm}'  # OEM 3 = LSTM only
        
        logger.info(f"Initialized Tesseract OCR: lang={language}, dpi={dpi}, psm={psm}")
    
    def extract_from_file(self, file_path: str) -> List[OCRResult]:
        """
        Main entry point - detects file type and extracts text
        
        Args:
            file_path: Path to PDF or image file
            
        Returns:
            List of OCRResult objects (one per page)
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        logger.info(f"Processing file: {file_path.name}")
        
        # Route to appropriate handler
        if file_path.suffix.lower() == '.pdf':
            return self._extract_from_pdf(file_path)
        elif file_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']:
            return self._extract_from_image(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")
    
    def _extract_from_pdf(self, pdf_path: Path) -> List[OCRResult]:
        """Extract text from multi-page PDF"""
        logger.info(f"Converting PDF to images (DPI={self.dpi})...")
        
        try:
            # Convert PDF pages to images
            images = convert_from_path(
                str(pdf_path),
                dpi=self.dpi,
                fmt='png',
                thread_count=4  # Parallel conversion
            )
            logger.info(f"Converted {len(images)} pages")
        except Exception as e:
            logger.error(f"PDF conversion failed: {e}")
            raise
        
        results = []
        for page_num, image in enumerate(images, start=1):
            logger.info(f"Processing page {page_num}/{len(images)}...")
            result = self._process_image(image, page_num)
            results.append(result)
        
        return results
    
    def _extract_from_image(self, image_path: Path) -> List[OCRResult]:
        """Extract text from single image"""
        image = Image.open(image_path)
        result = self._process_image(image, page=1)
        return [result]
    
    def _process_image(self, image: Image.Image, page: int) -> OCRResult:
        """
        Process a single image with preprocessing pipeline
        
        Pipeline:
        1. Convert to grayscale
        2. Enhance contrast
        3. Denoise
        4. Deskew (straighten)
        5. Binarize (threshold)
        6. Run OCR
        """
        # Preprocessing pipeline
        processed = self._preprocess_image(image)
        
        # Extract text
        text = pytesseract.image_to_string(
            processed,
            lang=self.language,
            config=self.config
        )
        
        # Extract detailed data with bounding boxes
        data = pytesseract.image_to_data(
            processed,
            lang=self.language,
            config=self.config,
            output_type=pytesseract.Output.DICT
        )
        
        # Parse bounding boxes and calculate confidence
        bounding_boxes, avg_confidence = self._parse_ocr_data(data)
        
        logger.info(f"Page {page}: Extracted {len(text)} chars, confidence={avg_confidence:.2f}")
        
        return OCRResult(
            text=text.strip(),
            confidence=avg_confidence,
            bounding_boxes=bounding_boxes,
            page=page
        )
    
    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """
        Advanced image preprocessing pipeline
        
        Steps:
        1. Convert to grayscale
        2. Increase contrast
        3. Denoise
        4. Deskew
        5. Binarize (Otsu's thresholding)
        """
        # Convert PIL Image to OpenCV format
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # 1. Grayscale conversion
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        
        # 2. Enhance contrast using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        contrast = clahe.apply(gray)
        
        # 3. Denoise
        denoised = cv2.fastNlMeansDenoising(contrast, h=10)
        
        # 4. Deskew (straighten tilted images)
        deskewed = self._deskew(denoised)
        
        # 5. Binarize with Otsu's method
        _, binary = cv2.threshold(
            deskewed,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        
        # Convert back to PIL Image
        return Image.fromarray(binary)
    
    def _deskew(self, image: np.ndarray) -> np.ndarray:
        """
        Detect and correct image skew (tilt)
        
        Uses Hough transform to detect dominant lines
        """
        # Detect edges
        edges = cv2.Canny(image, 50, 150, apertureSize=3)
        
        # Detect lines using Hough transform
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
        
        if lines is None:
            return image  # No skew detected
        
        # Calculate average angle
        angles = []
        for rho, theta in lines[:, 0]:
            angle = (theta * 180 / np.pi) - 90
            angles.append(angle)
        
        if not angles:
            return image
        
        # Get median angle (more robust than mean)
        median_angle = np.median(angles)
        
        # Only correct if skew is significant (> 0.5 degrees)
        if abs(median_angle) > 0.5:
            logger.debug(f"Deskewing image by {median_angle:.2f} degrees")
            (h, w) = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
            rotated = cv2.warpAffine(
                image,
                M,
                (w, h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE
            )
            return rotated
        
        return image
    
    def _parse_ocr_data(self, data: Dict) -> Tuple[List[Dict], float]:
        """
        Parse Tesseract output data to extract bounding boxes and confidence
        
        Args:
            data: Dictionary from pytesseract.image_to_data
            
        Returns:
            Tuple of (bounding_boxes, average_confidence)
        """
        bounding_boxes = []
        confidences = []
        
        n_boxes = len(data['text'])
        
        for i in range(n_boxes):
            # Filter out low confidence detections
            conf = int(data['conf'][i])
            if conf < 0:  # -1 means no text detected
                continue
            
            text = data['text'][i].strip()
            if not text:  # Skip empty text
                continue
            
            bbox = {
                "text": text,
                "x": data['left'][i],
                "y": data['top'][i],
                "width": data['width'][i],
                "height": data['height'][i],
                "confidence": conf,
                "block_num": data['block_num'][i],
                "line_num": data['line_num'][i],
                "word_num": data['word_num'][i]
            }
            
            bounding_boxes.append(bbox)
            confidences.append(conf)
        
        # Calculate average confidence
        avg_confidence = np.mean(confidences) if confidences else 0.0
        
        return bounding_boxes, avg_confidence
    
    def extract_text_only(self, file_path: str) -> str:
        """Quick method to extract just text (no bounding boxes)"""
        results = self.extract_from_file(file_path)
        return "\n\n".join([r.text for r in results])
    
    def get_word_boxes(self, file_path: str, page: int = 1) -> List[Dict]:
        """Get bounding boxes for all words on a specific page"""
        results = self.extract_from_file(file_path)
        if page > len(results):
            raise ValueError(f"Page {page} not found (document has {len(results)} pages)")
        
        return results[page - 1].bounding_boxes


# ============ TESTING ============

if __name__ == "__main__":
    # Test the OCR engine
    ocr = TesseractOCR(dpi=300, psm=6)
    
    # Test with a sample file
    test_file = "./data/uploads/Hons_Quote_AI.pdf"
    
    if Path(test_file).exists():
        print("Testing OCR engine...")
        results = ocr.extract_from_file(test_file)
        
        for result in results:
            print(f"\n{'='*50}")
            print(f"PAGE {result.page}")
            print(f"{'='*50}")
            print(f"Confidence: {result.confidence:.2f}%")
            print(f"Words detected: {len(result.bounding_boxes)}")
            print(f"\nText preview (first 500 chars):")
            print(result.text[:500])
            print(f"\nFirst 5 words with bounding boxes:")
            for box in result.bounding_boxes[:5]:
                print(f"  '{box['text']}' at ({box['x']}, {box['y']}) - confidence: {box['confidence']}")
    else:
        print(f"Test file not found: {test_file}")
        print("Place a sample PDF/image file to test OCR")