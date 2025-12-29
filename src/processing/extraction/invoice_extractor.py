#src/processing/extraction/invoice_extractor.py
import re
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class InvoiceExtractor:
    """
    Extract structured invoice data from text
    
    Uses regex patterns and heuristics to identify:
    - Invoice number
    - Dates (invoice date, due date)
    - Vendor information
    - Amounts (subtotal, tax, total)
    - Line items
    """
    
    def __init__(self):
        """Initialize extraction patterns"""
        # Invoice number patterns
        self.invoice_patterns = [
            r'invoice\s*#?\s*:?\s*([A-Z0-9\-]+)',
            r'inv\s*#?\s*:?\s*([A-Z0-9\-]+)',
            r'bill\s*#?\s*:?\s*([A-Z0-9\-]+)',
            r'invoice\s*number\s*:?\s*([A-Z0-9\-]+)',
        ]
        
        # Date patterns
        self.date_patterns = [
            r'\d{4}-\d{2}-\d{2}',  # 2024-01-15
            r'\d{2}/\d{2}/\d{4}',  # 01/15/2024
            r'\d{2}-\d{2}-\d{4}',  # 01-15-2024
        ]
        
        # Amount patterns
        self.amount_patterns = [
            r'(?:total|amount|sum)\s*:?\s*\$?\s*([0-9,]+\.?\d{0,2})',
            r'\$\s*([0-9,]+\.?\d{2})',
            r'([0-9,]+\.\d{2})',
        ]
        
        logger.info("Initialized InvoiceExtractor")
    
    def extract(self, text: str, tables: List[Dict] = None) -> Dict:
        """
        Extract invoice data from text and tables
        
        Args:
            text: Raw text from document
            tables: Extracted tables (optional)
            
        Returns:
            Dictionary with extracted invoice fields
        """
        logger.info("Extracting invoice data...")
        
        invoice_data = {
            "invoice_number": self._extract_invoice_number(text),
            "invoice_date": self._extract_date(text, "invoice"),
            "due_date": self._extract_date(text, "due"),
            "vendor_name": self._extract_vendor_name(text),
            "vendor_address": self._extract_vendor_address(text),
            "total_amount": self._extract_amount(text, "total"),
            "tax_amount": self._extract_amount(text, "tax"),
            "subtotal": self._extract_amount(text, "subtotal"),
            "currency": self._extract_currency(text),
            "line_items": []
        }
        
        # Extract line items from tables if available
        if tables:
            invoice_data["line_items"] = self._extract_line_items_from_tables(tables)
        
        # Calculate confidence
        confidence = self._calculate_confidence(invoice_data)
        invoice_data["confidence"] = confidence
        
        logger.info(f"Extraction complete. Confidence: {confidence:.2%}")
        
        return invoice_data
    
    def _extract_invoice_number(self, text: str) -> Optional[str]:
        """Extract invoice number"""
        text_lower = text.lower()
        
        for pattern in self.invoice_patterns:
            match = re.search(pattern, text_lower, re.IGNORECASE)
            if match:
                invoice_num = match.group(1).upper()
                logger.debug(f"Found invoice number: {invoice_num}")
                return invoice_num
        
        return None
    
    def _extract_date(self, text: str, date_type: str = "invoice") -> Optional[datetime]:
        """Extract date (invoice date or due date)"""
        # Look for context keywords
        if date_type == "invoice":
            keywords = ["invoice date", "date", "issued"]
        else:
            keywords = ["due date", "payment due", "due"]
        
        lines = text.split('\n')
        
        for i, line in enumerate(lines):
            line_lower = line.lower()
            
            # Check if line contains relevant keyword
            if any(keyword in line_lower for keyword in keywords):
                # Look for date pattern in this line and next few lines
                search_text = ' '.join(lines[i:i+3])
                
                for pattern in self.date_patterns:
                    match = re.search(pattern, search_text)
                    if match:
                        date_str = match.group(0)
                        parsed_date = self._parse_date(date_str)
                        if parsed_date:
                            logger.debug(f"Found {date_type} date: {parsed_date}")
                            return parsed_date
        
        return None
    
    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse date string to datetime object"""
        formats = [
            '%Y-%m-%d',
            '%d/%m/%Y',
            '%m/%d/%Y',
            '%d-%m-%Y',
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        
        return None
    
    def _extract_vendor_name(self, text: str) -> Optional[str]:
        """Extract vendor/company name (usually at top of document)"""
        lines = text.split('\n')
        
        # Typically the vendor name is in the first few lines
        for line in lines[:5]:
            line = line.strip()
            # Skip empty lines and lines with just numbers/dates
            if line and len(line) > 3 and not re.match(r'^\d+[\-/]\d+', line):
                # Check if it looks like a company name
                if len(line.split()) <= 5:  # Company names are usually short
                    logger.debug(f"Found potential vendor name: {line}")
                    return line
        
        return None
    
    def _extract_vendor_address(self, text: str) -> Optional[str]:
        """Extract vendor address"""
        # Look for address pattern (multiple lines with street, city, postal)
        lines = text.split('\n')
        
        address_lines = []
        for i, line in enumerate(lines[:10]):  # Check first 10 lines
            line = line.strip()
            # Look for address indicators
            if any(word in line.lower() for word in ['street', 'road', 'avenue', 'drive', 'suite']):
                # Collect this and next 2-3 lines
                address_lines = [l.strip() for l in lines[i:i+3] if l.strip()]
                address = ', '.join(address_lines)
                logger.debug(f"Found address: {address}")
                return address
        
        return None
    
    def _extract_amount(self, text: str, amount_type: str = "total") -> Optional[float]:
        """Extract monetary amount"""
        text_lower = text.lower()
        
        # Define keywords for each amount type
        keywords = {
            "total": ["total", "amount due", "balance"],
            "tax": ["tax", "vat", "gst"],
            "subtotal": ["subtotal", "sub-total", "sub total"]
        }
        
        lines = text.split('\n')
        
        for line in lines:
            line_lower = line.lower()
            
            # Check if line contains relevant keyword
            if any(keyword in line_lower for keyword in keywords.get(amount_type, [])):
                # Extract amount from this line
                for pattern in self.amount_patterns:
                    match = re.search(pattern, line, re.IGNORECASE)
                    if match:
                        amount_str = match.group(1).replace(',', '')
                        try:
                            amount = float(amount_str)
                            logger.debug(f"Found {amount_type}: {amount}")
                            return amount
                        except ValueError:
                            continue
        
        return None
    
    def _extract_currency(self, text: str) -> str:
        """Extract currency code"""
        # Look for common currency codes
        currencies = {
            'usd': 'USD', 'us$': 'USD', '$': 'USD',
            'eur': 'EUR', '€': 'EUR',
            'gbp': 'GBP', '£': 'GBP',
            'zar': 'ZAR', 'r': 'ZAR',
        }
        
        text_lower = text.lower()
        
        for key, value in currencies.items():
            if key in text_lower:
                logger.debug(f"Found currency: {value}")
                return value
        
        return "USD"  # Default
    
    def _extract_line_items_from_tables(self, tables: List[Dict]) -> List[Dict]:
        """Extract line items from tables"""
        line_items = []
        
        for table in tables:
            # Look for tables with typical invoice columns
            headers = table.get('headers', [])
            data = table.get('data', [])
            
            # Check if this looks like a line items table
            has_description = any('description' in str(h).lower() or 'item' in str(h).lower() for h in headers)
            has_amount = any('amount' in str(h).lower() or 'price' in str(h).lower() or 'total' in str(h).lower() for h in headers)
            
            if has_description and has_amount and len(data) > 0:
                logger.debug(f"Found line items table with {len(data)} rows")
                
                for row in data:
                    try:
                        # Extract line item details (this is simplified)
                        item = {
                            "description": str(row[0]) if len(row) > 0 else "",
                            "quantity": float(row[1]) if len(row) > 1 else 1.0,
                            "unit_price": float(str(row[2]).replace(',', '')) if len(row) > 2 else 0.0,
                            "total": float(str(row[-1]).replace(',', '')) if len(row) > 0 else 0.0
                        }
                        
                        if item["description"]:  # Only add if we have a description
                            line_items.append(item)
                    except (ValueError, IndexError) as e:
                        logger.warning(f"Could not parse line item: {e}")
                        continue
        
        logger.info(f"Extracted {len(line_items)} line items")
        return line_items
    
    def _calculate_confidence(self, invoice_data: Dict) -> float:
        """Calculate overall extraction confidence"""
        # Count how many required fields were extracted
        required_fields = ['invoice_number', 'total_amount', 'vendor_name']
        extracted_count = sum(1 for field in required_fields if invoice_data.get(field))
        
        base_confidence = extracted_count / len(required_fields)
        
        # Bonus for optional fields
        optional_fields = ['invoice_date', 'tax_amount', 'line_items']
        optional_count = sum(1 for field in optional_fields if invoice_data.get(field))
        bonus = optional_count * 0.05
        
        confidence = min(base_confidence + bonus, 1.0)
        
        return confidence


# ============ TESTING ============

if __name__ == "__main__":
    sample_text = """
    ACME Corporation
    123 Business St, New York, NY 10001
    
    INVOICE
    
    Invoice Number: INV-2024-001
    Invoice Date: 2024-01-15
    Due Date: 2024-02-15
    
    Bill To:
    John Doe
    456 Client Ave
    
    Description              Qty    Price      Total
    Web Development          10     $100.00    $1,000.00
    Hosting Services         1      $50.00     $50.00
    
    Subtotal:                                  $1,050.00
    Tax (15%):                                 $157.50
    Total Amount Due:                          $1,207.50
    """
    
    extractor = InvoiceExtractor()
    result = extractor.extract(sample_text)
    
    print("\n" + "="*60)
    print("INVOICE EXTRACTION TEST")
    print("="*60)
    
    for key, value in result.items():
        if value and key != 'line_items':
            print(f"{key:20}: {value}")
    
    if result.get('line_items'):
        print(f"\nLine Items: {len(result['line_items'])}")
        for i, item in enumerate(result['line_items'], 1):
            print(f"  {i}. {item['description']}: ${item['total']}")