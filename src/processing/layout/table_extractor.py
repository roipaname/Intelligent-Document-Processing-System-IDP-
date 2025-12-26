import cv2
import numpy as np
from PIL import Image
from typing import List, Dict, Tuple, Optional
import pandas as pd
import logging
from pathlib import Path
import camelot
from pdfminer.high_level import extract_text
from pdf2image import convert_from_path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def pdf_has_text(pdf_path: str, min_chars: int = 50) -> bool:
    """
    Check if PDF has extractable text (native PDF, not scanned)
    
    Args:
        pdf_path: Path to PDF file
        min_chars: Minimum number of characters to consider as "has text"
        
    Returns:
        True if PDF has extractable text, False otherwise
    """
    try:
        text = extract_text(pdf_path)
        has_text = text and len(text.strip()) > min_chars
        logger.info(f"PDF text detection: {'Text-based PDF' if has_text else 'Scanned PDF (image)'}")
        return has_text
    except Exception as e:
        logger.warning(f"Error checking PDF text: {e}")
        return False


class TableExtractor:
    """
    Intelligent table extractor that adapts to PDF type
    
    Strategy:
    - Text-based PDFs: Use Camelot (fast, accurate, no OCR needed)
    - Scanned PDFs: Use computer vision + OCR (slower but works on images)
    
    Features:
    - Automatic PDF type detection
    - Handles bordered and borderless tables
    - Detects merged cells
    - Auto-detects headers
    - Handles multi-line cells
    """
    
    def __init__(
        self,
        min_cell_width: int = 20,
        min_cell_height: int = 10,
        dpi: int = 300,
        poppler_path: Optional[str] = None
    ):
        """
        Initialize table extractor
        
        Args:
            min_cell_width: Minimum width for a valid cell (pixels) - OCR mode
            min_cell_height: Minimum height for a valid cell (pixels) - OCR mode
            dpi: DPI for PDF to image conversion - OCR mode
            poppler_path: Path to poppler binaries (required on some systems)
        """
        self.min_cell_width = min_cell_width
        self.min_cell_height = min_cell_height
        self.dpi = dpi
        self.poppler_path = poppler_path
        logger.info("Initialized Intelligent Table Extractor")
    
    def extract_tables_from_pdf(
        self,
        pdf_path: str,
        pages: str = 'all',
        flavor: str = 'lattice'
    ) -> List[pd.DataFrame]:
        """
        Main entry point: Intelligently extract tables from PDF
        
        Decision Tree:
        1. Check if PDF has extractable text
        2. If YES → Use Camelot (fast, no OCR)
        3. If NO → Convert to image → Use computer vision + OCR
        
        Args:
            pdf_path: Path to PDF file
            pages: Pages to process ('all' or '1,2,3' or '1-3')
            flavor: Camelot flavor ('lattice' for bordered, 'stream' for borderless)
            
        Returns:
            List of pandas DataFrames (one per table)
        """
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        logger.info(f"Processing PDF: {pdf_path.name}")
        
        # DECISION POINT: Does PDF have extractable text?
        if pdf_has_text(str(pdf_path)):
            # PATH A: Text-based PDF → Use Camelot (NO OCR)
            logger.info("📄 Using Camelot extraction (text-based PDF)")
            return self._extract_with_camelot(str(pdf_path), pages, flavor)
        else:
            # PATH B: Scanned PDF → Use OCR
            logger.info("🖼️  Using OCR extraction (scanned/image PDF)")
            return self._extract_with_ocr(str(pdf_path), pages)
    
    def _extract_with_camelot(
        self,
        pdf_path: str,
        pages: str = 'all',
        flavor: str = 'lattice'
    ) -> List[pd.DataFrame]:
        """
        Extract tables using Camelot (for text-based PDFs)
        
        Advantages:
        - Very fast (no OCR needed)
        - High accuracy
        - Preserves text formatting
        - Handles complex tables
        
        Args:
            pdf_path: Path to PDF
            pages: Pages to extract ('all', '1', '1,2,3', '1-3')
            flavor: 'lattice' (bordered tables) or 'stream' (borderless)
            
        Returns:
            List of DataFrames
        """
        try:
            # Extract tables with Camelot
            tables = camelot.read_pdf(
                pdf_path,
                pages=pages,
                flavor=flavor,
                suppress_stdout=True
            )
            
            logger.info(f"Camelot found {len(tables)} table(s)")
            
            dataframes = []
            for i, table in enumerate(tables):
                df = table.df
                
                # Clean the DataFrame
                df = self._clean_camelot_dataframe(df)
                
                if df is not None and not df.empty:
                    dataframes.append(df)
                    logger.info(
                        f"Table {i+1}: {df.shape[0]} rows × {df.shape[1]} cols "
                        f"(accuracy: {table.accuracy:.1f}%)"
                    )
            
            # If lattice failed and found no tables, try stream flavor
            if not dataframes and flavor == 'lattice':
                logger.info("No tables found with lattice, trying stream flavor...")
                return self._extract_with_camelot(pdf_path, pages, flavor='stream')
            
            return dataframes
            
        except Exception as e:
            logger.error(f"Camelot extraction failed: {e}")
            logger.info("Falling back to OCR extraction...")
            return self._extract_with_ocr(pdf_path, pages)
    
    def _extract_with_ocr(self, pdf_path: str, pages: str = 'all') -> List[pd.DataFrame]:
        """
        Extract tables using OCR (for scanned/image PDFs)
        
        Pipeline:
        1. Convert PDF pages to images
        2. Detect table regions using computer vision
        3. Extract cells using line detection
        4. OCR each cell
        5. Build DataFrames
        
        Args:
            pdf_path: Path to PDF
            pages: Pages to extract
            
        Returns:
            List of DataFrames
        """
        # Import OCR engine
        try:
            from src.processing.ocr.tesseract_engine import TesseractOCR
        except ImportError:
            logger.error("TesseractOCR not available. Install required dependencies.")
            return []
        
        # Convert PDF to images
        logger.info("Converting PDF to images for OCR...")
        
        try:
            # Parse pages parameter
            if pages == 'all':
                images = convert_from_path(
                    pdf_path,
                    dpi=self.dpi,
                    poppler_path=self.poppler_path
                )
            else:
                # Parse page numbers (e.g., '1,2,3' or '1-3')
                page_nums = self._parse_page_numbers(pages)
                images = convert_from_path(
                    pdf_path,
                    dpi=self.dpi,
                    first_page=min(page_nums),
                    last_page=max(page_nums),
                    poppler_path=self.poppler_path
                )
        except Exception as e:
            logger.error(f"PDF to image conversion failed: {e}")
            return []
        
        logger.info(f"Converted {len(images)} page(s) to images")
        
        # Initialize OCR engine
        ocr_engine = TesseractOCR()
        
        # Extract tables from each page
        all_tables = []
        for page_num, image in enumerate(images, start=1):
            logger.info(f"Processing page {page_num}/{len(images)}...")
            tables = self.extract_tables(image, ocr_engine)
            all_tables.extend(tables)
        
        return all_tables
    
    def extract_tables(self, image: Image.Image, ocr_engine) -> List[pd.DataFrame]:
        """
        Extract tables from a single image using computer vision + OCR
        
        Args:
            image: PIL Image object
            ocr_engine: TesseractOCR instance
            
        Returns:
            List of DataFrames
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
    
    def _clean_camelot_dataframe(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Clean DataFrame extracted by Camelot
        
        - Remove empty rows/columns
        - Detect and set headers
        - Strip whitespace
        """
        if df.empty:
            return None
        
        # Strip whitespace from all cells
        df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
        
        # Use first row as header if it looks like one
        if len(df) > 1:
            first_row = df.iloc[0]
            # Check if first row has mostly text (not numbers)
            text_cells = sum(
                isinstance(val, str) and val and not val.replace('.', '').replace(',', '').isdigit()
                for val in first_row
            )
            
            if text_cells >= len(first_row) * 0.5:  # At least 50% text
                df.columns = df.iloc[0]
                df = df.drop(0).reset_index(drop=True)
        
        # Remove completely empty rows
        df = df.replace('', np.nan)
        df = df.dropna(how='all')
        
        # Remove completely empty columns
        df = df.dropna(axis=1, how='all')
        
        # Reset index
        df = df.reset_index(drop=True)
        
        return df if not df.empty else None
    
    def _parse_page_numbers(self, pages: str) -> List[int]:
        """Parse page number string to list of integers"""
        page_nums = []
        
        for part in pages.split(','):
            if '-' in part:
                start, end = map(int, part.split('-'))
                page_nums.extend(range(start, end + 1))
            else:
                page_nums.append(int(part))
        
        return sorted(set(page_nums))
    
    # ========== COMPUTER VISION METHODS (for OCR path) ==========
    
    def _detect_table_regions(self, gray: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect table regions using line detection"""
        horizontal = self._detect_lines(gray, horizontal=True)
        vertical = self._detect_lines(gray, horizontal=False)
        
        table_mask = cv2.addWeighted(horizontal, 0.5, vertical, 0.5, 0)
        _, table_binary = cv2.threshold(table_mask, 50, 255, cv2.THRESH_BINARY)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        table_binary = cv2.dilate(table_binary, kernel, iterations=2)
        
        contours, _ = cv2.findContours(
            table_binary,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 100 and h > 50:
                regions.append((x, y, w, h))
                logger.debug(f"Detected table region: ({x},{y}) size=({w}x{h})")
        
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
        
        horizontal_lines = self._find_line_positions(table_roi, horizontal=True)
        vertical_lines = self._find_line_positions(table_roi, horizontal=False)
        
        logger.debug(f"Found {len(horizontal_lines)} horizontal and {len(vertical_lines)} vertical lines")
        
        if len(horizontal_lines) < 2 or len(vertical_lines) < 2:
            logger.warning("Not enough grid lines detected for table extraction")
            return []
        
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
                
                if cell["width"] >= self.min_cell_width and cell["height"] >= self.min_cell_height:
                    cells.append(cell)
        
        logger.debug(f"Extracted {len(cells)} cells from table")
        return cells
    
    def _find_line_positions(self, image: np.ndarray, horizontal: bool = True) -> List[int]:
        """Find positions of grid lines"""
        if horizontal:
            projection = np.sum(image < 128, axis=1)
        else:
            projection = np.sum(image < 128, axis=0)
        
        threshold = np.max(projection) * 0.3 if np.max(projection) > 0 else 0
        
        lines = []
        in_line = False
        line_start = 0
        
        for i, val in enumerate(projection):
            if val > threshold and not in_line:
                in_line = True
                line_start = i
            elif val <= threshold and in_line:
                in_line = False
                line_pos = (line_start + i) // 2
                lines.append(line_pos)
        
        if in_line:
            lines.append((line_start + len(projection)) // 2)
        
        if not lines or lines[0] > 5:
            lines.insert(0, 0)
        if not lines or lines[-1] < len(projection) - 5:
            lines.append(len(projection) - 1)
        
        return lines
    
    def _extract_cell_text(self, image: Image.Image, cells: List[Dict], ocr_engine) -> List[Dict]:
        """Extract text from each cell using OCR"""
        import pytesseract
        
        for idx, cell in enumerate(cells):
            try:
                cropped = image.crop((
                    cell["x"] + 2,
                    cell["y"] + 2,
                    cell["x"] + cell["width"] - 2,
                    cell["y"] + cell["height"] - 2
                ))
                
                text = pytesseract.image_to_string(
                    cropped,
                    config='--psm 7'
                ).strip()
                
                text = ' '.join(text.split())
                cell["text"] = text
                
                if text:
                    logger.debug(f"Cell ({cell['row']},{cell['col']}): '{text}'")
                
            except Exception as e:
                logger.error(f"Error extracting text from cell ({cell['row']},{cell['col']}): {e}")
                cell["text"] = ""
        
        return cells
    
    def _cells_to_dataframe(self, cells: List[Dict]) -> Optional[pd.DataFrame]:
        """Convert extracted cells to DataFrame"""
        if not cells:
            return None
        
        try:
            max_row = max(cell["row"] for cell in cells)
            max_col = max(cell["col"] for cell in cells)
            
            logger.debug(f"Building DataFrame: {max_row+1} rows x {max_col+1} cols")
            
            grid = [['' for _ in range(max_col + 1)] for _ in range(max_row + 1)]
            
            for cell in cells:
                grid[cell["row"]][cell["col"]] = cell["text"]
            
            df = pd.DataFrame(grid)
            
            # Detect header
            if len(df) > 1:
                first_row_values = df.iloc[0].values
                first_row_text = ' '.join(str(x) for x in first_row_values if x)
                
                if first_row_text:
                    digit_count = sum(c.isdigit() for c in first_row_text)
                    total_chars = len(first_row_text.replace(' ', ''))
                    
                    if total_chars > 0 and digit_count / total_chars < 0.3:
                        df.columns = df.iloc[0]
                        df = df.drop(0).reset_index(drop=True)
                        logger.debug("Using first row as header")
            
            df = df.replace('', np.nan)
            df = df.dropna(how='all')
            df = df.dropna(axis=1, how='all')
            df = df.reset_index(drop=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Error converting cells to DataFrame: {e}")
            return None


# ============ TESTING ============

if __name__ == "__main__":
    test_pdf = "./data/uploads/Hons_Quote_AI.pdf"
    
    if Path(test_pdf).exists():
        print("="*70)
        print("INTELLIGENT TABLE EXTRACTION TEST")
        print("="*70)
        
        # Initialize extractor
        extractor = TableExtractor(poppler_path="/opt/local/bin")
        
        # Extract tables (automatically chooses Camelot or OCR)
        print("\n🔍 Analyzing PDF and extracting tables...")
        tables = extractor.extract_tables_from_pdf(test_pdf, pages='all')
        
        print(f"\n✅ Extracted {len(tables)} table(s)\n")
        
        # Display results
        if tables:
            for i, table in enumerate(tables, 1):
                print(f"\n{'='*70}")
                print(f"TABLE {i}")
                print(f"{'='*70}")
                print(f"Shape: {table.shape[0]} rows × {table.shape[1]} columns\n")
                print(table.head(10))
                
                # Save to CSV
                output_file = f"./data/processed/table_{i}.csv"
                Path(output_file).parent.mkdir(parents=True, exist_ok=True)
                table.to_csv(output_file, index=False)
                print(f"\n💾 Saved to: {output_file}")
        else:
            print("⚠️  No tables detected in the document")
        
        print("\n" + "="*70)
        print("✅ Test completed!")
        print("="*70)
        
    else:
        print(f"❌ Test file not found: {test_pdf}")