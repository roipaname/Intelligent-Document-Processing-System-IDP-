#src/processing/extraction/base_extractor.py
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import re
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class BaseExtractor(ABC):
    """
    Base class for document field extractors
    
    Provides common utility methods for text extraction
    """
    
    def __init__(self):
        """Initialize common patterns"""
        # Date patterns
        self.date_patterns = [
            (r'\d{4}-\d{2}-\d{2}', '%Y-%m-%d'),           # 2024-01-15
            (r'\d{2}/\d{2}/\d{4}', '%d/%m/%Y'),           # 15/01/2024
            (r'\d{2}/\d{2}/\d{4}', '%m/%d/%Y'),           # 01/15/2024
            (r'\d{2}-\d{2}-\d{4}', '%d-%m-%Y'),           # 15-01-2024
            (r'\d{1,2}\s+[A-Za-z]+\s+\d{4}', '%d %B %Y'), # 15 January 2024
            (r'[A-Za-z]+\s+\d{1,2},?\s+\d{4}', '%B %d, %Y'), # January 15, 2024
        ]
        
        # Amount patterns
        self.amount_patterns = [
            r'(?:total|amount|sum|balance)\s*:?\s*\$?\s*([0-9,]+\.?\d{0,2})',
            r'\$\s*([0-9,]+\.?\d{2})',
            r'(?:R|ZAR|USD|EUR|GBP)\s*([0-9,]+\.?\d{2})',
            r'([0-9,]+\.\d{2})',
        ]
        
        # Email pattern
        self.email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        
        # Phone patterns
        self.phone_patterns = [
            r'\+?\d{1,3}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}',
            r'\d{3}[-.\s]?\d{3}[-.\s]?\d{4}',
        ]
    
    @abstractmethod
    def extract(self, text: str, tables: List[Dict] = None) -> Dict:
        """
        Extract structured data from text
        
        Args:
            text: Raw text from document
            tables: Extracted tables (optional)
            
        Returns:
            Dictionary with extracted fields
        """
        pass
    
    def extract_date(self, text: str, keywords: List[str]) -> Optional[datetime]:
        """
        Extract date near specific keywords
        
        Args:
            text: Text to search
            keywords: Keywords to look for (e.g., ['invoice date', 'date'])
            
        Returns:
            Parsed datetime object or None
        """
        lines = text.split('\n')
        
        for i, line in enumerate(lines):
            line_lower = line.lower()
            
            # Check if line contains any keyword
            if any(keyword.lower() in line_lower for keyword in keywords):
                # Search this line and next few lines
                search_text = ' '.join(lines[i:min(i+3, len(lines))])
                
                # Try each date pattern
                for pattern, date_format in self.date_patterns:
                    match = re.search(pattern, search_text)
                    if match:
                        date_str = match.group(0)
                        parsed_date = self._parse_date(date_str, date_format)
                        if parsed_date:
                            logger.debug(f"Found date '{date_str}' near keywords {keywords}")
                            return parsed_date
        
        return None
    
    def _parse_date(self, date_str: str, date_format: str) -> Optional[datetime]:
        """Parse date string with given format"""
        try:
            return datetime.strptime(date_str, date_format)
        except ValueError:
            # Try alternate format if initial fails
            if date_format == '%d/%m/%Y':
                try:
                    return datetime.strptime(date_str, '%m/%d/%Y')
                except ValueError:
                    pass
            return None
    
    def extract_amount(self, text: str, keywords: List[str]) -> Optional[float]:
        """
        Extract monetary amount near keywords
        
        Args:
            text: Text to search
            keywords: Keywords to look for
            
        Returns:
            Extracted amount or None
        """
        lines = text.split('\n')
        
        for line in lines:
            line_lower = line.lower()
            
            # Check if line contains keyword
            if any(keyword.lower() in line_lower for keyword in keywords):
                # Try to extract amount from this line
                for pattern in self.amount_patterns:
                    match = re.search(pattern, line, re.IGNORECASE)
                    if match:
                        amount_str = match.group(1).replace(',', '')
                        try:
                            amount = float(amount_str)
                            logger.debug(f"Found amount {amount} near keywords {keywords}")
                            return amount
                        except ValueError:
                            continue
        
        return None
    
    def extract_email(self, text: str) -> Optional[str]:
        """Extract email address"""
        match = re.search(self.email_pattern, text)
        if match:
            email = match.group(0)
            logger.debug(f"Found email: {email}")
            return email
        return None
    
    def extract_phone(self, text: str) -> Optional[str]:
        """Extract phone number"""
        for pattern in self.phone_patterns:
            match = re.search(pattern, text)
            if match:
                phone = match.group(0)
                logger.debug(f"Found phone: {phone}")
                return phone
        return None
    
    def extract_by_label(self, text: str, label: str, pattern: str = r':\s*(.+)') -> Optional[str]:
        """
        Extract value by label (e.g., 'Invoice Number: INV-001')
        
        Args:
            text: Text to search
            label: Label to look for
            pattern: Regex pattern for value extraction
            
        Returns:
            Extracted value or None
        """
        lines = text.split('\n')
        
        for line in lines:
            if label.lower() in line.lower():
                match = re.search(f'{re.escape(label)}{pattern}', line, re.IGNORECASE)
                if match:
                    value = match.group(1).strip()
                    logger.debug(f"Found {label}: {value}")
                    return value
        
        return None
    
    def extract_lines_after_keyword(self, text: str, keyword: str, num_lines: int = 3) -> List[str]:
        """
        Extract N lines after a keyword
        
        Args:
            text: Text to search
            keyword: Keyword to find
            num_lines: Number of lines to extract after keyword
            
        Returns:
            List of lines
        """
        lines = text.split('\n')
        
        for i, line in enumerate(lines):
            if keyword.lower() in line.lower():
                # Extract next N lines
                extracted = []
                for j in range(i+1, min(i+1+num_lines, len(lines))):
                    line_text = lines[j].strip()
                    if line_text:
                        extracted.append(line_text)
                return extracted
        
        return []
    
    def calculate_confidence(self, extracted_data: Dict, required_fields: List[str]) -> float:
        """
        Calculate extraction confidence based on required fields
        
        Args:
            extracted_data: Dictionary of extracted fields
            required_fields: List of field names that are required
            
        Returns:
            Confidence score (0-1)
        """
        if not required_fields:
            return 1.0
        
        extracted_count = sum(
            1 for field in required_fields 
            if extracted_data.get(field) not in [None, '', []]
        )
        
        confidence = extracted_count / len(required_fields)
        
        logger.debug(f"Confidence: {confidence:.2%} ({extracted_count}/{len(required_fields)} required fields)")
        
        return confidence