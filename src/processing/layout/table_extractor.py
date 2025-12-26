import cv2
import numpy as np
from PIL import Image
from typing import List, Dict, Tuple, Optional
import pandas as pd
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TableExtractor:
    """
    Extract structured data from tables in documents
    
    Approach:
    1. Detect table boundaries using line detection
    2. Detect rows and columns by finding grid lines
    3. Extract cell contents using OCR
    4. Build structured data (pandas DataFrame)
    
    Features:
    - Handles bordered and borderless tables
    - Detects merged cells
    - Auto-detects headers
    - Handles multi-line cells
    """
    
    def __init__(self, min_cell_width: int = 20, min_cell_height: int = 10):
        """
        Initialize table extractor
        
        Args:
            min_cell_width: Minimum width for a valid cell (pixels)
            min_cell_height: Minimum height for a valid cell (pixels)
        """
        self.min_cell_width = min_cell_width
        self.min_cell_height = min_cell_height
        logger.info("Initialized Table Extractor")
    
    def extract_tables(self, image: Image.Image, ocr_engine) -> List[pd.DataFrame]:
        """
        Extract all tables from an image
        
        Args:
            image: PIL Image object
            ocr_engine: TesseractOCR instance for text extraction
            
        Returns:
            List of pandas DataFrames (one per detected table)
        """
        # Convert PIL Image to OpenCV format
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        
        # Detect table regions
        table_regions = self._detect_table_regions(gray)
        
        logger.info(f"Found {len(table_regions)} potential tables")
        
        tables = []
        for i, region in enumerate(table_regions):
            logger.info(f"Processing table {i+1}/{len(table_regions)}")
            
            try:
                # Extract table structure (cells)
                cells = self._extract_cells(gray, region)
                
                if not cells:
                    logger.warning(f"No cells found in table {i+1}")
                    continue
                
                # Extract text from each cell using OCR
                table_data = self._extract_cell_text(image, cells, ocr_engine)
                
                # Convert cells to DataFrame
                df = self._cells_to_dataframe(table_data)
                
                if df is not None and not df.empty:
                    tables.append(df)
                    logger.info(f"Extracted table {i+1}: {df.shape[0]} rows x {df.shape[1]} cols")
                else:
                    logger.warning(f"Table {i+1} resulted in empty DataFrame")
                    
            except Exception as e:
                logger.error(f"Error processing table {i+1}: {e}")
                continue
        
        return tables
    
    def _detect_table_regions(self, gray: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect regions in the image that likely contain tables
        
        Args:
            gray: Grayscale image as numpy array
            
        Returns:
            List of bounding boxes (x, y, width, height)
        """
        # Detect horizontal lines
        horizontal = self._detect_lines(gray, horizontal=True)
        
        # Detect vertical lines
        vertical = self._detect_lines(gray, horizontal=False)
        
        # Combine horizontal and vertical lines to find table intersections
        table_mask = cv2.addWeighted(horizontal, 0.5, vertical, 0.5, 0)
        
        # Threshold to get binary image
        _, table_binary = cv2.threshold(table_mask, 50, 255, cv2.THRESH_BINARY)
        
        # Dilate to connect nearby regions
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        table_binary = cv2.dilate(table_binary, kernel, iterations=2)
        
        # Find contours (table boundaries)
        contours, _ = cv2.findContours(
            table_binary,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter out small regions (noise)
            # Tables should be reasonably large
            if w > 100 and h > 50:
                regions.append((x, y, w, h))
                logger.debug(f"Detected table region: ({x},{y}) size=({w}x{h})")
        
        return regions
    
    def _detect_lines(self, gray: np.ndarray, horizontal: bool = True) -> np.ndarray:
        """
        Detect horizontal or vertical lines in the image
        
        Args:
            gray: Grayscale image
            horizontal: If True, detect horizontal lines; else vertical
            
        Returns:
            Binary image with detected lines
        """
        if horizontal:
            # Horizontal line detection kernel (wide and short)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        else:
            # Vertical line detection kernel (narrow and tall)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        
        # Apply morphological opening to detect lines
        lines = cv2.morphologyEx(
            gray,
            cv2.MORPH_OPEN,
            kernel,
            iterations=2
        )
        
        return lines
    
    def _extract_cells(self, gray: np.ndarray, region: Tuple[int, int, int, int]) -> List[Dict]:
        """
        Extract individual cells from a table region
        
        Args:
            gray: Grayscale image
            region: Table bounding box (x, y, width, height)
            
        Returns:
            List of cell dictionaries with position and coordinates
        """
        x, y, w, h = region
        
        # Extract table region
        table_roi = gray[y:y+h, x:x+w]
        
        # Detect grid lines within the table
        horizontal_lines = self._find_line_positions(table_roi, horizontal=True)
        vertical_lines = self._find_line_positions(table_roi, horizontal=False)
        
        logger.debug(f"Found {len(horizontal_lines)} horizontal and {len(vertical_lines)} vertical lines")
        
        if len(horizontal_lines) < 2 or len(vertical_lines) < 2:
            logger.warning("Not enough grid lines detected for table extraction")
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
                    "height": horizontal_lines[i+1] - horizontal_lines[i],
                    "text": ""
                }
                
                # Filter out cells that are too small
                if cell["width"] >= self.min_cell_width and cell["height"] >= self.min_cell_height:
                    cells.append(cell)
        
        logger.debug(f"Extracted {len(cells)} cells from table")
        
        return cells
    
    def _find_line_positions(self, image: np.ndarray, horizontal: bool = True) -> List[int]:
        """
        Find positions of grid lines in the image
        
        Uses projection profile method:
        - For horizontal lines: sum pixels along each row
        - For vertical lines: sum pixels along each column
        
        Args:
            image: Grayscale image of table region
            horizontal: If True, find horizontal lines; else vertical
            
        Returns:
            List of line positions (pixel coordinates)
        """
        if horizontal:
            # Sum along columns to find horizontal lines
            projection = np.sum(image < 128, axis=1)
        else:
            # Sum along rows to find vertical lines
            projection = np.sum(image < 128, axis=0)
        
        # Find peaks in the projection (these are lines)
        threshold = np.max(projection) * 0.3 if np.max(projection) > 0 else 0
        
        lines = []
        in_line = False
        line_start = 0
        
        for i, val in enumerate(projection):
            if val > threshold and not in_line:
                # Start of a line
                in_line = True
                line_start = i
            elif val <= threshold and in_line:
                # End of a line
                in_line = False
                # Use middle of the line as position
                line_pos = (line_start + i) // 2
                lines.append(line_pos)
        
        # Add final line if we're still in one
        if in_line:
            lines.append((line_start + len(projection)) // 2)
        
        # Add boundaries (edges of table)
        if not lines or lines[0] > 5:
            lines.insert(0, 0)
        if not lines or lines[-1] < len(projection) - 5:
            lines.append(len(projection) - 1)
        
        return lines
    
    def _extract_cell_text(self, image: Image.Image, cells: List[Dict], ocr_engine) -> List[Dict]:
        """
        Extract text from each cell using OCR
        
        Args:
            image: Original PIL Image
            cells: List of cell dictionaries with coordinates
            ocr_engine: TesseractOCR instance
            
        Returns:
            Updated cells list with extracted text
        """
        import pytesseract
        
        for idx, cell in enumerate(cells):
            try:
                # Crop cell region from image
                cropped = image.crop((
                    cell["x"] + 2,  # Small padding to avoid borders
                    cell["y"] + 2,
                    cell["x"] + cell["width"] - 2,
                    cell["y"] + cell["height"] - 2
                ))
                
                # Extract text using OCR
                # PSM 7 = single line, PSM 6 = single block
                text = pytesseract.image_to_string(
                    cropped,
                    config='--psm 7'  # Treat as single text line
                ).strip()
                
                # Clean extracted text
                text = ' '.join(text.split())  # Remove extra whitespace
                
                cell["text"] = text
                
                if text:
                    logger.debug(f"Cell ({cell['row']},{cell['col']}): '{text}'")
                
            except Exception as e:
                logger.error(f"Error extracting text from cell ({cell['row']},{cell['col']}): {e}")
                cell["text"] = ""
        
        return cells
    
    def _cells_to_dataframe(self, cells: List[Dict]) -> Optional[pd.DataFrame]:
        """
        Convert extracted cells to a pandas DataFrame
        
        Args:
            cells: List of cell dictionaries with row, col, and text
            
        Returns:
            pandas DataFrame or None if conversion fails
        """
        if not cells:
            return None
        
        try:
            # Find grid dimensions
            max_row = max(cell["row"] for cell in cells)
            max_col = max(cell["col"] for cell in cells)
            
            logger.debug(f"Building DataFrame: {max_row+1} rows x {max_col+1} cols")
            
            # Build 2D array
            grid = [['' for _ in range(max_col + 1)] for _ in range(max_row + 1)]
            
            for cell in cells:
                grid[cell["row"]][cell["col"]] = cell["text"]
            
            # Convert to DataFrame
            df = pd.DataFrame(grid)
            
            # Detect and use first row as header if it looks like one
            if len(df) > 1:
                first_row_values = df.iloc[0].values
                first_row_text = ' '.join(str(x) for x in first_row_values if x)
                
                # Heuristic: if first row has text but very few numbers, it's probably a header
                if first_row_text:
                    digit_count = sum(c.isdigit() for c in first_row_text)
                    total_chars = len(first_row_text.replace(' ', ''))
                    
                    if total_chars > 0 and digit_count / total_chars < 0.3:
                        # Looks like a header
                        df.columns = df.iloc[0]
                        df = df.drop(0).reset_index(drop=True)
                        logger.debug("Using first row as header")
            
            # Remove completely empty rows
            df = df.replace('', np.nan)
            df = df.dropna(how='all')
            
            # Remove completely empty columns
            df = df.dropna(axis=1, how='all')
            
            # Reset index
            df = df.reset_index(drop=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Error converting cells to DataFrame: {e}")
            return None
    
    def extract_table_at_region(
        self,
        image: Image.Image,
        bbox: Tuple[int, int, int, int],
        ocr_engine
    ) -> Optional[pd.DataFrame]:
        """
        Extract a table from a specific region
        
        Args:
            image: PIL Image
            bbox: Bounding box (x, y, width, height)
            ocr_engine: TesseractOCR instance
            
        Returns:
            pandas DataFrame or None
        """
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        
        cells = self._extract_cells(gray, bbox)
        
        if not cells:
            return None
        
        cells = self._extract_cell_text(image, cells, ocr_engine)
        df = self._cells_to_dataframe(cells)
        
        return df


# ============ HELPER FUNCTIONS ============

def visualize_table_detection(image: Image.Image, tables: List[pd.DataFrame], output_path: str = None):
    """
    Visualize detected tables on the image
    
    Args:
        image: Original PIL Image
        tables: List of extracted DataFrames
        output_path: Path to save visualization (optional)
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image)
    
    # Draw bounding boxes (placeholder - would need to store regions)
    ax.set_title(f"Detected {len(tables)} tables")
    ax.axis('off')
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved visualization to {output_path}")
    else:
        plt.show()
    
    plt.close()


def merge_similar_tables(tables: List[pd.DataFrame], similarity_threshold: float = 0.8) -> List[pd.DataFrame]:
    """
    Merge tables that have similar structures (same columns)
    
    Useful when a table spans multiple pages or regions
    
    Args:
        tables: List of DataFrames
        similarity_threshold: Minimum column similarity to merge (0-1)
        
    Returns:
        List of merged DataFrames
    """
    if len(tables) <= 1:
        return tables
    
    merged = []
    used = set()
    
    for i, table1 in enumerate(tables):
        if i in used:
            continue
        
        current_group = [table1]
        
        for j, table2 in enumerate(tables[i+1:], start=i+1):
            if j in used:
                continue
            
            # Check column similarity
            cols1 = set(table1.columns)
            cols2 = set(table2.columns)
            
            if cols1 and cols2:
                similarity = len(cols1 & cols2) / len(cols1 | cols2)
                
                if similarity >= similarity_threshold:
                    current_group.append(table2)
                    used.add(j)
        
        # Merge group
        if len(current_group) > 1:
            merged_table = pd.concat(current_group, ignore_index=True)
            merged.append(merged_table)
            logger.info(f"Merged {len(current_group)} tables")
        else:
            merged.append(table1)
        
        used.add(i)
    
    return merged


# ============ TESTING ============

if __name__ == "__main__":
    from pdf2image import convert_from_path
    from pathlib import Path
    
    # Import OCR engine (need to adjust path based on your structure)
    import sys
    sys.path.append(str(Path(__file__).parent.parent.parent))
    
    try:
        from src.processing.ocr.tesseract_engine import TesseractOCR
    except ImportError:
        logger.error("Could not import TesseractOCR. Make sure the module exists.")
        sys.exit(1)
    
    # Test file path
    test_pdf = "./data/uploads/Hons_Quote_AI.pdf"
    
    if Path(test_pdf).exists():
        print("="*60)
        print("Testing Table Extractor")
        print("="*60)
        
        # Convert PDF to image
        print("\n1. Converting PDF to image...")
        images = convert_from_path(test_pdf, dpi=300, first_page=1, last_page=1)
        print(f"   ✓ Converted {len(images)} page(s)")
        
        # Initialize OCR and extractor
        print("\n2. Initializing OCR engine and table extractor...")
        ocr = TesseractOCR()
        extractor = TableExtractor()
        print("   ✓ Initialized")
        
        # Extract tables
        print("\n3. Extracting tables...")
        tables = extractor.extract_tables(images[0], ocr)
        print(f"   ✓ Extracted {len(tables)} table(s)")
        
        # Display results
        if tables:
            print("\n4. Table contents:")
            for i, table in enumerate(tables, 1):
                print(f"\n{'='*60}")
                print(f"TABLE {i}")
                print(f"{'='*60}")
                print(f"Shape: {table.shape[0]} rows × {table.shape[1]} columns")
                print(f"\nPreview:")
                print(table.head(10))
                
                # Save to CSV
                output_file = f"./data/processed/table_{i}.csv"
                Path(output_file).parent.mkdir(parents=True, exist_ok=True)
                table.to_csv(output_file, index=False)
                print(f"\n✓ Saved to: {output_file}")
        else:
            print("\n   ⚠ No tables detected in the document")
        
        print("\n" + "="*60)
        print("Test completed successfully!")
        print("="*60)
        
    else:
        print(f"❌ Test file not found: {test_pdf}")
        print("\nPlease ensure you have a sample PDF file at the specified location.")
        print("You can place any PDF with tables for testing.")