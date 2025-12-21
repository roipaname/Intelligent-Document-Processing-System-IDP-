import cv2
import numpy as np
from PIL import Image
from typing import List, Dict, Tuple
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class TableExtractor:
    """
    Extract structured data from tables in documents
    
    Approach:
    1. Detect table boundaries
    2. Detect rows and columns
    3. Extract cell contents
    4. Build structured data (DataFrame)
    """
    
    def __init__(self):
        self.min_cell_width = 20
        self.min_cell_height = 10
        logger.info("Initialized Table Extractor")
    
    def extract_tables(self, image: Image.Image, ocr_engine) -> List[pd.DataFrame]:
        """
        Extract all tables from an image
        
        Args:
            image: PIL Image
            ocr_engine: TesseractOCR instance for text extraction
            
        Returns:
            List of pandas DataFrames (one per table)
        """
        # Convert to OpenCV
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        
        # Detect table regions
        table_regions = self._detect_table_regions(gray)
        
        logger.info(f"Found {len(table_regions)} potential tables")
        
        tables = []
        for i, region in enumerate(table_regions):
            logger.info(f"Processing table {i+1}/{len(table_regions)}")
            
            # Extract table structure
            cells = self._extract_cells(gray, region)
            
            if not cells:
                logger.warning(f"No cells found in table {i+1}")
                continue
            
            # Extract text from each cell
            table_data = self._extract_cell_text(image, cells, ocr_engine)
            
            # Convert to DataFrame
            df = self._cells_to_dataframe(table_data)
            
            if df is not None and not df.empty:
                tables.append(df)
                logger.info(f"Extracted table {i+1}: {df.shape[0]} rows x {df.shape[1]} cols")
        
        return tables
    
    def _detect_table_regions(self, gray: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect regions that likely contain tables"""
        # Detect horizontal and vertical lines
        horizontal = self._detect_lines(gray, horizontal=True)
        vertical = self._detect_lines(gray, horizontal=False)
        
        # Combine to find intersections (tables)
        table_mask = cv2.addWeighted(horizontal, 0.5, vertical, 0.5, 0)
        _, table_binary = cv2.threshold(table_mask, 50, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(table_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            # Filter small regions
            if w > 100 and h > 50:
                regions.append((x, y, w, h))
        
        return regions
    
    def _detect_lines(self, gray: np.ndarray, horizontal: bool = True) -> np.ndarray:
        """Detect horizontal or vertical lines"""
        if horizontal:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        else:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        
        lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel, iterations=2)
        return lines
    
    def _extract_cells(self, gray: np.ndarray, region: Tuple[int, int, int, int]) -> List[Dict]:
        """Extract individual cells from table region"""
        x, y, w, h = region
        table_roi = gray[y:y+h, x:x+w]
        
        # Detect grid lines
        horizontal_lines = self._find_line_positions(table_roi, horizontal=True)
        vertical_lines = self._find_line_positions(table_roi, horizontal=False)
        
        if len(horizontal_lines) < 2 or len(vertical_lines) < 2:
            logger.warning("Not enough grid lines detected")
            return []
        
        # Build cells from line intersections
        cells = []
        for i in range(len(horizontal_lines) - 1):
            for j in range(len(vertical_lines) - 1):
                cell = {
                    "row": i,
                    "col": j,
                    "x": x + vertical_lines[j],
                    "y": y + horizontal_lines[i],
                    "width": vertical_lines[j+1] - vertical_lines[j],
                    "height": horizontal_lines[i+1] - horizontal_lines[i]
                }
                
                # Filter tiny cells
                if cell["width"] >= self.min_cell_width and cell["height"] >= self.min_cell_height:
                    cells.append(cell)
        
        return cells
    
    def _find_line_positions(self, image: np.ndarray, horizontal: bool = True) -> List[int]:
        """Find positions of grid lines"""
        if horizontal:
            # Sum along columns to find horizontal lines
            projection = np.sum(image < 128, axis=1)
        else:
            # Sum along rows to find vertical lines
            projection = np.sum(image < 128, axis=0)
        
        # Find peaks (line positions)
        threshold = np.max(projection) * 0.3
        lines = []
        
        in_line = False
        line_start = 0
        
        for i, val in enumerate(projection):
            if val > threshold and not in_line:
                in_line = True
                line_start = i
            elif val <= threshold and in_line:
                in_line = False
                lines.append((line_start + i) // 2)  # Use middle of line
        
        return lines
    
    def _extract_cell_text(
        self,
        image: Image.Image,
        cells: List[Dict],
        ocr_engine
    ) -> List[Dict]:
        """Extract text for each detected cell using OCR"""
        table_data = []

        for cell in cells:
            x, y = cell["x"], cell["y"]
            w, h = cell["width"], cell["height"]

            # Crop cell region from original image
            cell_img = image.crop((x, y, x + w, y + h))

            try:
                text = ocr_engine.extract_text(cell_img)
            except Exception as e:
                logger.error(f"OCR failed for cell ({cell['row']}, {cell['col']}): {e}")
                text = ""

            table_data.append({
                "row": cell["row"],
                "col": cell["col"],
                "text": text.strip()
            })

        return table_data

    def _cells_to_dataframe(self, table_data: List[Dict]) -> pd.DataFrame | None:
        """Convert extracted cell text into a pandas DataFrame"""
        if not table_data:
            return None

        # Determine table size
        max_row = max(cell["row"] for cell in table_data)
        max_col = max(cell["col"] for cell in table_data)

        # Initialize empty table
        data = [["" for _ in range(max_col + 1)] for _ in range(max_row + 1)]

        # Fill table
        for cell in table_data:
            data[cell["row"]][cell["col"]] = cell["text"]

        # Create DataFrame
        df = pd.DataFrame(data)

        # Optional: treat first row as header if it looks like one
        if df.shape[0] > 1:
            header_score = sum(
                1 for val in df.iloc[0] if isinstance(val, str) and val.strip()
            )
            if header_score >= df.shape[1] // 2:
                df.columns = df.iloc[0]
                df = df[1:].reset_index(drop=True)

        return df
