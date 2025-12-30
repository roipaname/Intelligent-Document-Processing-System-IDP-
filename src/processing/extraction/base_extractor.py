#src/processing/extraction/base_extractor.py
"""
Base extractor class for all document types
"""
import re
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class BaseExtractor:
    """
    Base class for document data extraction
    
    Provides common utility methods for:
    - Pattern matching
    - Date parsing
    - Amount extraction
    - Confidence scoring
    """
    
    def __init__(self):
        """Initialize common extraction patterns"""
        # Common date patterns
        self.date_patterns = [
            r'\d{4}-\d{2}-\d{2}',          # 2024-01-15
            r'\d{2}/\d{2}/\d{4}',          # 01/15/2024
            r'\d{2}-\d{2}-\d{4}',          # 01-15-2024
            r'\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}',  # 15 January 2024
        ]
        
        # Common amount patterns
        self.amount_patterns = [
            r'\$\s*([0-9,]+\.?\d{0,2})',   # $1,234.56
            r'([0-9,]+\.\d{2})',           # 1,234.56
            r'R\s*([0-9,]+\.?\d{0,2})',    # R1,234.56 (ZAR)
            r'€\s*([0-9,]+\.?\d{0,2})',    # €1,234.56
            r'£\s*([0-9,]+\.?\d{0,2})',    # £1,234.56
        ]
        
        # Email pattern
        self.email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        
        # Phone patterns
        self.phone_patterns = [
            r'\+?\d{1,3}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}',  # International
            r'\(\d{3}\)\s*\d{3}[-.\s]?\d{4}',  # (123) 456-7890
            r'\d{3}[-.\s]?\d{3}[-.\s]?\d{4}',  # 123-456-7890
        ]
        
        logger.info(f"Initialized {self.__class__.__name__}")
    
    def extract(self, text: str, tables: List[Dict] = None, **kwargs) -> Dict:
        """
        Main extraction method - to be implemented by subclasses
        
        Args:
            text: Raw extracted text
            tables: Extracted tables
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with extracted data
        """
        raise NotImplementedError("Subclasses must implement extract()")
    
    def find_pattern(
        self, 
        text: str, 
        patterns: List[str], 
        flags: int = re.IGNORECASE
    ) -> Optional[str]:
        """
        Find first match from list of patterns
        
        Args:
            text: Text to search
            patterns: List of regex patterns
            flags: Regex flags
            
        Returns:
            Matched string or None
        """
        for pattern in patterns:
            match = re.search(pattern, text, flags)
            if match:
                return match.group(1) if match.groups() else match.group(0)
        return None
    
    def find_all_patterns(
        self, 
        text: str, 
        patterns: List[str], 
        flags: int = re.IGNORECASE
    ) -> List[str]:
        """Find all matches from list of patterns"""
        results = []
        for pattern in patterns:
            matches = re.findall(pattern, text, flags)
            results.extend(matches)
        return results
    
    def extract_date(
        self, 
        text: str, 
        context_keywords: List[str] = None
    ) -> Optional[datetime]:
        """
        Extract date with optional context
        
        Args:
            text: Text to search
            context_keywords: Keywords to search near (e.g., ['invoice date'])
            
        Returns:
            Parsed datetime or None
        """
        search_text = text
        
        # If context keywords provided, narrow search
        if context_keywords:
            lines = text.split('\n')
            for i, line in enumerate(lines):
                if any(kw.lower() in line.lower() for kw in context_keywords):
                    # Search in this line and next 2 lines
                    search_text = ' '.join(lines[i:min(i+3, len(lines))])
                    break
        
        # Find date pattern
        date_str = self.find_pattern(search_text, self.date_patterns)
        
        if date_str:
            return self.parse_date(date_str)
        
        return None
    
    def parse_date(self, date_str: str) -> Optional[datetime]:
        """
        Parse date string to datetime object
        
        Args:
            date_str: Date string
            
        Returns:
            Datetime object or None
        """
        formats = [
            '%Y-%m-%d',
            '%d/%m/%Y',
            '%m/%d/%Y',
            '%d-%m-%Y',
            '%m-%d-%Y',
            '%d %B %Y',
            '%d %b %Y',
            '%B %d, %Y',
            '%b %d, %Y',
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(date_str.strip(), fmt)
            except ValueError:
                continue
        
        logger.warning(f"Could not parse date: {date_str}")
        return None
    
    def extract_amount(
        self, 
        text: str, 
        context_keywords: List[str] = None
    ) -> Optional[float]:
        """
        Extract monetary amount with optional context
        
        Args:
            text: Text to search
            context_keywords: Keywords near amount (e.g., ['total', 'amount due'])
            
        Returns:
            Float amount or None
        """
        search_text = text
        
        # If context keywords provided, narrow search
        if context_keywords:
            lines = text.split('\n')
            for line in lines:
                if any(kw.lower() in line.lower() for kw in context_keywords):
                    search_text = line
                    break
        
        # Find amount
        amount_str = self.find_pattern(search_text, self.amount_patterns)
        
        if amount_str:
            try:
                # Remove commas and parse
                amount = float(amount_str.replace(',', ''))
                return amount
            except ValueError:
                logger.warning(f"Could not parse amount: {amount_str}")
        
        return None
    
    def extract_email(self, text: str) -> Optional[str]:
        """Extract email address"""
        match = re.search(self.email_pattern, text, re.IGNORECASE)
        return match.group(0) if match else None
    
    def extract_phone(self, text: str) -> Optional[str]:
        """Extract phone number"""
        phone = self.find_pattern(text, self.phone_patterns)
        if phone:
            # Clean phone number
            phone = re.sub(r'[^\d+]', '', phone)
            return phone
        return None
    
    def extract_lines_near_keyword(
        self, 
        text: str, 
        keyword: str, 
        num_lines: int = 3
    ) -> List[str]:
        """
        Extract lines near a keyword
        
        Args:
            text: Full text
            keyword: Keyword to search for
            num_lines: Number of lines to extract after keyword
            
        Returns:
            List of lines
        """
        lines = text.split('\n')
        
        for i, line in enumerate(lines):
            if keyword.lower() in line.lower():
                return [l.strip() for l in lines[i:min(i+num_lines, len(lines))] if l.strip()]
        
        return []
    
    def calculate_confidence(self, extracted_data: Dict, required_fields: List[str]) -> float:
        """
        Calculate extraction confidence based on required fields
        
        Args:
            extracted_data: Extracted data dictionary
            required_fields: List of required field names
            
        Returns:
            Confidence score (0.0 to 1.0)
        """
        if not required_fields:
            return 1.0
        
        # Count extracted required fields
        extracted_count = sum(
            1 for field in required_fields 
            if extracted_data.get(field) is not None
        )
        
        base_confidence = extracted_count / len(required_fields)
        
        return base_confidence
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters
        text = re.sub(r'[^\w\s.,;:!?()-]', '', text)
        return text.strip()