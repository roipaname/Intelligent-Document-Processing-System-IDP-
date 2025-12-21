import cv2
import numpy as np
from PIL import Image
from dataclasses import dataclass
from typing import List, Tuple, Dict
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class RegionType(str, Enum):
    """Types of document regions"""
    HEADER = "header"
    FOOTER = "footer"
    TABLE = "table"
    TEXT_BLOCK = "text_block"
    IMAGE = "image"
    SIGNATURE = "signature"
    LOGO = "logo"


@dataclass
class Region:
    """Represents a region in the document"""
    type: RegionType
    bbox: Tuple[int, int, int, int]  # (x, y, width, height)
    confidence: float
    page: int = 1
    text: str = ""
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self):
        return {
            "type": self.type.value,
            "bbox": {
                "x": self.bbox[0],
                "y": self.bbox[1],
                "width": self.bbox[2],
                "height": self.bbox[3]
            },
            "confidence": self.confidence,
            "page": self.page,
            "text": self.text,
            "metadata": self.metadata
        }


class LayoutAnalyzer:
    """
    Analyzes document layout to identify different regions
    
    Uses computer vision techniques:
    - Connected component analysis
    - Morphological operations
    - Hough line detection for tables
    - Position-based heuristics
    """
    
    def __init__(self, min_confidence: float = 0.6):
        self.min_confidence = min_confidence
        logger.info("Initialized Layout Analyzer")
    
    def analyze(self, image: Image.Image, page: int = 1) -> List[Region]:
        """
        Analyze document layout and return detected regions
        
        Args:
            image: PIL Image object
            page: Page number
            
        Returns:
            List of Region objects sorted by position (top to bottom)
        """
        # Convert to OpenCV format
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        
        height, width = gray.shape
        
        logger.info(f"Analyzing layout for page {page} ({width}x{height})")
        
        regions = []
        
        # Detect different region types
        regions.extend(self._detect_tables(gray, page, width, height))
        regions.extend(self._detect_text_blocks(gray, page, width, height))
        regions.extend(self._detect_header_footer(gray, page, width, height))
        
        # Sort regions by Y position (top to bottom)
        regions.sort(key=lambda r: r.bbox[1])
        
        logger.info(f"Detected {len(regions)} regions")
        
        return regions
    
    def _detect_tables(self, image: np.ndarray, page: int, width: int, height: int) -> List[Region]:
        """
        Detect table regions using line detection
        
        Tables are characterized by:
        - Horizontal and vertical lines forming a grid
        - Regular spacing
        - Rectangular bounding boxes
        """
        regions = []
        
        # Detect horizontal lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        detect_horizontal = cv2.morphologyEx(
            image,
            cv2.MORPH_OPEN,
            horizontal_kernel,
            iterations=2
        )
        
        # Detect vertical lines
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        detect_vertical = cv2.morphologyEx(
            image,
            cv2.MORPH_OPEN,
            vertical_kernel,
            iterations=2
        )
        
        # Combine horizontal and vertical lines
        table_mask = cv2.addWeighted(detect_horizontal, 0.5, detect_vertical, 0.5, 0)
        
        # Threshold to get binary image
        _, table_binary = cv2.threshold(table_mask, 50, 255, cv2.THRESH_BINARY)
        
        # Find contours (potential tables)
        contours, _ = cv2.findContours(
            table_binary,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter by size - tables should be reasonably large
            if w < width * 0.3 or h < 50:
                continue
            
            # Calculate confidence based on line density
            roi = table_mask[y:y+h, x:x+w]
            line_density = np.sum(roi > 0) / (w * h)
            confidence = min(line_density * 10, 1.0)  # Scale to 0-1
            
            if confidence >= self.min_confidence:
                region = Region(
                    type=RegionType.TABLE,
                    bbox=(x, y, w, h),
                    confidence=confidence,
                    page=page,
                    metadata={"line_density": line_density}
                )
                regions.append(region)
                logger.debug(f"Detected table at ({x},{y}) size=({w}x{h}) conf={confidence:.2f}")
        
        return regions
    
    def _detect_text_blocks(self, image: np.ndarray, page: int, width: int, height: int) -> List[Region]:
        """
        Detect text block regions using connected component analysis
        
        Text blocks are characterized by:
        - Groups of connected text
        - Rectangular shapes
        - Middle section of document
        """
        regions = []
        
        # Invert image (text becomes white)
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Dilate to connect nearby text
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
        dilated = cv2.dilate(binary, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter text blocks
            # - Should be wide enough (at least 30% of page width)
            # - Not too small
            # - Aspect ratio should be reasonable
            if w < width * 0.3 or h < 20:
                continue
            
            aspect_ratio = w / h
            if aspect_ratio < 1.5:  # Too square, might be image or logo
                continue
            
            # Calculate confidence based on text density
            roi = binary[y:y+h, x:x+w]
            text_density = np.sum(roi > 0) / (w * h)
            confidence = min(text_density * 2, 1.0)
            
            if confidence >= self.min_confidence:
                region = Region(
                    type=RegionType.TEXT_BLOCK,
                    bbox=(x, y, w, h),
                    confidence=confidence,
                    page=page,
                    metadata={"aspect_ratio": aspect_ratio, "text_density": text_density}
                )
                regions.append(region)
        
        return regions
    
    def _detect_header_footer(self, image: np.ndarray, page: int, width: int, height: int) -> List[Region]:
        """
        Detect header and footer regions based on position
        
        Heuristics:
        - Header: Top 15% of page
        - Footer: Bottom 10% of page
        """
        regions = []
        
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Header region (top 15%)
        header_height = int(height * 0.15)
        header_roi = binary[0:header_height, :]
        
        if np.sum(header_roi > 0) > 100:  # Check if there's content
            # Find bounding box of header content
            contours, _ = cv2.findContours(header_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                x, y, w, h = cv2.boundingRect(np.vstack(contours))
                region = Region(
                    type=RegionType.HEADER,
                    bbox=(x, y, w, h),
                    confidence=0.85,
                    page=page
                )
                regions.append(region)
                logger.debug(f"Detected header at top of page")
        
        # Footer region (bottom 10%)
        footer_start = int(height * 0.90)
        footer_roi = binary[footer_start:, :]
        
        if np.sum(footer_roi > 0) > 100:
            contours, _ = cv2.findContours(footer_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                x, y, w, h = cv2.boundingRect(np.vstack(contours))
                region = Region(
                    type=RegionType.FOOTER,
                    bbox=(x, footer_start + y, w, h),
                    confidence=0.85,
                    page=page
                )
                regions.append(region)
                logger.debug(f"Detected footer at bottom of page")
        
        return regions
    
    def extract_region_text(self, image: Image.Image, region: Region, ocr_engine) -> str:
        """
        Extract text from a specific region using OCR
        
        Args:
            image: PIL Image
            region: Region object with bounding box
            ocr_engine: TesseractOCR instance
            
        Returns:
            Extracted text
        """
        x, y, w, h = region.bbox
        
        # Crop region from image
        cropped = image.crop((x, y, x + w, y + h))
        
        # Run OCR on cropped region
        import pytesseract
        text = pytesseract.image_to_string(cropped, config=ocr_engine.config)
        
        return text.strip()


# ============ TESTING ============

if __name__ == "__main__":
    from pdf2image import convert_from_path
    
    # Test layout analysis
    test_pdf = "./data/uploads/Hons_Quote_AI.pdf"
    
    if Path(test_pdf).exists():
        print("Testing Layout Analyzer...")
        
        # Convert first page to image
        images = convert_from_path(test_pdf, dpi=300, first_page=1, last_page=1)
        
        analyzer = LayoutAnalyzer()
        regions = analyzer.analyze(images[0], page=1)
        
        print(f"\nDetected {len(regions)} regions:")
        for i, region in enumerate(regions, 1):
            print(f"\n{i}. {region.type.value.upper()}")
            print(f"   Position: ({region.bbox[0]}, {region.bbox[1]})")
            print(f"   Size: {region.bbox[2]}x{region.bbox[3]}")
            print(f"   Confidence: {region.confidence:.2f}")
    else:
        print(f"Test file not found: {test_pdf}")