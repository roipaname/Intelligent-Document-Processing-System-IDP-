#src/processing/pipeline.py
import logging
from pathlib import Path
from typing import Dict, List, Optional
import time
from datetime import datetime

from src.processing.ocr.tesseract_engine import TesseractOCR, OCRResult
from src.processing.layout.layout_analyzer import LayoutAnalyzer, Region
from src.processing.layout.table_extractor import TableExtractor
from src.config.settings import settings

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """
    Main document processing pipeline
    
    Pipeline stages:
    1. Document type detection
    2. Text extraction (OCR or native PDF)
    3. Layout analysis
    4. Table extraction
    5. Entity extraction (TODO)
    6. Validation (TODO)
    """
    
    def __init__(self):
        """Initialize all processing components"""
        self.ocr_engine = TesseractOCR(
            tesseract_cmd=settings.tesseract_cmd,
            language=settings.ocr_language,
            dpi=settings.ocr_dpi,
            psm=settings.ocr_psm
        )
        
        self.layout_analyzer = LayoutAnalyzer(
            min_confidence=settings.confidence_threshold
        )
        
        self.table_extractor = TableExtractor(
            dpi=settings.ocr_dpi,
            poppler_path=settings.poppler_path
        )
        
        logger.info("Initialized DocumentProcessor")
    
    def process(self, file_path: str) -> Dict:
        """
        Process a document through the complete pipeline
        
        Args:
            file_path: Path to document file
            
        Returns:
            Dictionary with extracted data and metadata
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        logger.info(f"Starting processing pipeline for: {file_path.name}")
        
        result = {
            "filename": file_path.name,
            "file_type": file_path.suffix,
            "stages": [],
            "extracted_data": {},
            "tables": [],
            "layout_regions": [],
            "overall_confidence": 0.0,
            "processing_time_ms": 0,
            "error": None
        }
        
        start_time = time.time()
        
        try:
            # Stage 1: Text Extraction
            logger.info("Stage 1: Text Extraction")
            stage_result = self._stage_text_extraction(file_path)
            result["stages"].append(stage_result)
            result["extracted_data"]["raw_text"] = stage_result.get("text", "")
            result["extracted_data"]["ocr_results"] = stage_result.get("ocr_results", [])
            
            # Stage 2: Table Extraction (if PDF and enabled)
            if settings.enable_table_extraction and file_path.suffix == '.pdf':
                logger.info("Stage 2: Table Extraction")
                stage_result = self._stage_table_extraction(file_path)
                result["stages"].append(stage_result)
                result["tables"] = stage_result.get("tables", [])
            
            # Stage 3: Layout Analysis (for images)
            if file_path.suffix in ['.png', '.jpg', '.jpeg', '.tiff']:
                logger.info("Stage 3: Layout Analysis")
                stage_result = self._stage_layout_analysis(file_path)
                result["stages"].append(stage_result)
                result["layout_regions"] = stage_result.get("regions", [])
            
            # Calculate overall confidence
            confidences = []
            for stage in result["stages"]:
                if "confidence" in stage and stage["confidence"] is not None:
                    confidences.append(stage["confidence"])
            
            if confidences:
                result["overall_confidence"] = sum(confidences) / len(confidences)
            
            # Calculate processing time
            result["processing_time_ms"] = int((time.time() - start_time) * 1000)
            
            logger.info(f"✅ Processing completed in {result['processing_time_ms']}ms")
            
        except Exception as e:
            logger.error(f"❌ Processing failed: {e}")
            result["error"] = str(e)
            result["stages"].append({
                "stage": "error",
                "status": "failed",
                "error": str(e)
            })
            result["processing_time_ms"] = int((time.time() - start_time) * 1000)
        
        return result
    
    def _stage_text_extraction(self, file_path: Path) -> Dict:
        """Stage 1: Extract text from document"""
        stage_start = time.time()
        
        try:
            if file_path.suffix == '.pdf':
                # Check if PDF has extractable text
                from pdfminer.high_level import extract_text
                
                try:
                    text = extract_text(str(file_path))
                    if text and len(text.strip()) > settings.min_text_chars:
                        # Text-based PDF
                        logger.info("✓ Text-based PDF detected (using PDFMiner)")
                        return {
                            "stage": "text_extraction",
                            "status": "completed",
                            "method": "pdfminer",
                            "text": text,
                            "char_count": len(text),
                            "duration_ms": int((time.time() - stage_start) * 1000),
                            "confidence": 1.0
                        }
                except Exception as e:
                    logger.warning(f"PDFMiner failed: {e}, falling back to OCR")
            
            # Use OCR for scanned PDFs or images
            logger.info("✓ Using OCR for text extraction")
            ocr_results = self.ocr_engine.extract_from_file(str(file_path))
            
            # Combine text from all pages
            full_text = "\n\n".join([r.text for r in ocr_results])
            
            # Calculate average confidence
            avg_confidence = sum(r.confidence for r in ocr_results) / len(ocr_results) if ocr_results else 0
            
            return {
                "stage": "text_extraction",
                "status": "completed",
                "method": "ocr",
                "text": full_text,
                "char_count": len(full_text),
                "pages": len(ocr_results),
                "ocr_results": [r.to_dict() for r in ocr_results],
                "duration_ms": int((time.time() - stage_start) * 1000),
                "confidence": avg_confidence / 100  # Convert to 0-1 scale
            }
            
        except Exception as e:
            logger.error(f"Text extraction failed: {e}")
            return {
                "stage": "text_extraction",
                "status": "failed",
                "error": str(e),
                "duration_ms": int((time.time() - stage_start) * 1000),
                "confidence": 0.0
            }
    
    def _stage_table_extraction(self, file_path: Path) -> Dict:
        """Stage 2: Extract tables from document"""
        stage_start = time.time()
        
        try:
            tables = self.table_extractor.extract_tables_from_pdf(
                str(file_path),
                pages='all'
            )
            
            # Convert DataFrames to dictionaries
            table_data = []
            for i, df in enumerate(tables):
                table_data.append({
                    "table_index": i + 1,
                    "rows": int(df.shape[0]),
                    "columns": int(df.shape[1]),
                    "headers": df.columns.tolist() if hasattr(df.columns, 'tolist') else list(df.columns),
                    "data": df.values.tolist()
                })
            
            return {
                "stage": "table_extraction",
                "status": "completed",
                "tables": table_data,
                "table_count": len(tables),
                "duration_ms": int((time.time() - stage_start) * 1000)
            }
            
        except Exception as e:
            logger.error(f"Table extraction failed: {e}")
            return {
                "stage": "table_extraction",
                "status": "failed",
                "error": str(e),
                "tables": [],
                "table_count": 0,
                "duration_ms": int((time.time() - stage_start) * 1000)
            }
    
    def _stage_layout_analysis(self, file_path: Path) -> Dict:
        """Stage 3: Analyze document layout"""
        stage_start = time.time()
        
        try:
            from PIL import Image
            
            image = Image.open(file_path)
            regions = self.layout_analyzer.analyze(image, page=1)
            
            return {
                "stage": "layout_analysis",
                "status": "completed",
                "regions": [r.to_dict() for r in regions],
                "region_count": len(regions),
                "duration_ms": int((time.time() - stage_start) * 1000)
            }
            
        except Exception as e:
            logger.error(f"Layout analysis failed: {e}")
            return {
                "stage": "layout_analysis",
                "status": "failed",
                "error": str(e),
                "regions": [],
                "region_count": 0,
                "duration_ms": int((time.time() - stage_start) * 1000)
            }


# ============ TESTING ============

if __name__ == "__main__":
    from src.config.settings import init_directories
    
    init_directories()
    
    test_file = "./data/uploads/Hons_Quote_AI.pdf"
    
    if Path(test_file).exists():
        print("\n" + "="*70)
        print("DOCUMENT PROCESSING PIPELINE TEST")
        print("="*70)
        
        processor = DocumentProcessor()
        result = processor.process(test_file)
        
        print(f"\n📄 File: {result['filename']}")
        print(f"⏱️  Processing Time: {result['processing_time_ms']}ms")
        print(f"📊 Overall Confidence: {result['overall_confidence']:.2%}")
        
        print("\n🔄 Processing Stages:")
        for stage in result['stages']:
            status_icon = "✅" if stage['status'] == 'completed' else "❌"
            print(f"  {status_icon} {stage['stage']}: {stage['status']} ({stage.get('duration_ms', 0)}ms)")
            if 'method' in stage:
                print(f"     Method: {stage['method']}")
        
        print(f"\n📋 Tables Extracted: {len(result.get('tables', []))}")
        print(f"🗺️  Layout Regions: {len(result.get('layout_regions', []))}")
        
        if result.get('error'):
            print(f"\n⚠️  Error: {result['error']}")
        
        print("\n" + "="*70)
        
    else:
        print(f"❌ Test file not found: {test_file}")