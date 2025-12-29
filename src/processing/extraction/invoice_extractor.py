#src/processing/extraction/invoice_extractor.py
from typing import Dict, List, Optional
import re
import logging
from .base_extractor import BaseExtractor

logger = logging.getLogger(__name__)


class InvoiceExtractor(BaseExtractor):
    """
    Extract structured data from invoices
    
    Extracts:
    - Invoice number
    - Invoice date, due date
    - Vendor information (name, address, contact)
    - Customer/bill-to information
    - Line items
    - Amounts (subtotal, tax, total)
    - Payment terms
    """
    
    def __init__(self):
        """Initialize invoice-specific patterns"""
        super().__init__()
        
        # Invoice number patterns
        self.invoice_number_patterns = [
            r'invoice\s*#?\s*:?\s*([A-Z0-9\-/]+)',
            r'inv\.?\s*#?\s*:?\s*([A-Z0-9\-/]+)',
            r'bill\s*#?\s*:?\s*([A-Z0-9\-/]+)',
            r'invoice\s*number\s*:?\s*([A-Z0-9\-/]+)',
            r'ref\s*:?\s*([A-Z0-9\-/]+)',
        ]
        
        # Tax patterns
        self.tax_keywords = ['tax', 'vat', 'gst', 'sales tax']
        
        logger.info("Initialized InvoiceExtractor")
    
    def extract(self, text: str, tables: List[Dict] = None) -> Dict:
        """
        Extract all invoice fields
        
        Args:
            text: Raw extracted text
            tables: Extracted tables
            
        Returns:
            Dictionary with invoice data
        """
        logger.info("Starting invoice extraction...")
        
        invoice_data = {
            # Identifiers
            "invoice_number": self._extract_invoice_number(text),
            "po_number": self._extract_po_number(text),
            
            # Dates
            "invoice_date": self.extract_date(text, ['invoice date', 'date', 'issued']),
            "due_date": self.extract_date(text, ['due date', 'payment due', 'due']),
            
            # Vendor information
            "vendor_name": self._extract_vendor_name(text),
            "vendor_address": self._extract_vendor_address(text),
            "vendor_email": self.extract_email(text),
            "vendor_phone": self.extract_phone(text),
            "vendor_tax_id": self._extract_tax_id(text),
            
            # Customer information
            "customer_name": self._extract_customer_name(text),
            "customer_address": self._extract_customer_address(text),
            
            # Amounts
            "subtotal": self.extract_amount(text, ['subtotal', 'sub-total', 'sub total']),
            "tax_amount": self.extract_amount(text, self.tax_keywords),
            "total_amount": self.extract_amount(text, ['total', 'amount due', 'balance', 'grand total']),
            "currency": self._extract_currency(text),
            
            # Line items
            "line_items": [],
            
            # Payment info
            "payment_terms": self._extract_payment_terms(text),
        }
        
        # Extract line items from tables
        if tables:
            invoice_data["line_items"] = self._extract_line_items(tables)
        
        # Calculate confidence
        required_fields = ['invoice_number', 'total_amount', 'vendor_name']
        confidence = self.calculate_confidence(invoice_data, required_fields)
        invoice_data["confidence"] = confidence
        
        logger.info(f"✅ Invoice extraction complete. Confidence: {confidence:.2%}")
        
        return invoice_data
    
    def _extract_invoice_number(self, text: str) -> Optional[str]:
        """Extract invoice number"""
        text_lower = text.lower()
        
        for pattern in self.invoice_number_patterns:
            match = re.search(pattern, text_lower, re.IGNORECASE)
            if match:
                invoice_num = match.group(1).upper().strip()
                logger.debug(f"Found invoice number: {invoice_num}")
                return invoice_num
        
        return None
    
    def _extract_po_number(self, text: str) -> Optional[str]:
        """Extract purchase order number"""
        patterns = [
            r'P\.?O\.?\s*#?\s*:?\s*([A-Z0-9\-]+)',
            r'purchase\s*order\s*:?\s*([A-Z0-9\-]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                po_num = match.group(1).strip()
                logger.debug(f"Found PO number: {po_num}")
                return po_num
        
        return None
    
    def _extract_vendor_name(self, text: str) -> Optional[str]:
        """Extract vendor/company name (usually at top)"""
        lines = text.split('\n')
        
        # Vendor name typically in first 5 lines
        for line in lines[:5]:
            line = line.strip()
            
            # Skip empty, very short, or date-like lines
            if not line or len(line) < 3:
                continue
            
            # Skip lines that look like dates or addresses
            if re.match(r'^\d+[\-/]\d+', line):
                continue
            
            if any(word in line.lower() for word in ['date', 'invoice', 'bill']):
                continue
            
            # Company names are usually 1-5 words
            word_count = len(line.split())
            if 1 <= word_count <= 5:
                logger.debug(f"Found vendor name: {line}")
                return line
        
        return None
    
    def _extract_vendor_address(self, text: str) -> Optional[str]:
        """Extract vendor address"""
        lines = text.split('\n')
        
        # Look for address keywords in first 15 lines
        for i, line in enumerate(lines[:15]):
            line_lower = line.lower()
            
            # Address indicators
            if any(word in line_lower for word in ['street', 'road', 'avenue', 'drive', 'suite', 'floor']):
                # Collect this and next 2 lines
                address_lines = []
                for j in range(i, min(i+3, len(lines))):
                    addr_line = lines[j].strip()
                    if addr_line and len(addr_line) > 3:
                        address_lines.append(addr_line)
                
                if address_lines:
                    address = ', '.join(address_lines)
                    logger.debug(f"Found vendor address: {address}")
                    return address
        
        return None
    
    def _extract_tax_id(self, text: str) -> Optional[str]:
        """Extract tax ID / VAT number"""
        patterns = [
            r'(?:tax\s*id|vat|tax\s*number)\s*:?\s*([A-Z0-9\-]+)',
            r'(?:ein|fein)\s*:?\s*(\d{2}-?\d{7})',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                tax_id = match.group(1).strip()
                logger.debug(f"Found tax ID: {tax_id}")
                return tax_id
        
        return None
    
    def _extract_customer_name(self, text: str) -> Optional[str]:
        """Extract customer/bill-to name"""
        # Look for "Bill To" section
        lines = self.extract_lines_after_keyword(text, 'bill to', num_lines=1)
        
        if lines:
            customer_name = lines[0]
            logger.debug(f"Found customer name: {customer_name}")
            return customer_name
        
        return None
    
    def _extract_customer_address(self, text: str) -> Optional[str]:
        """Extract customer address"""
        # Look for lines after "Bill To"
        lines = self.extract_lines_after_keyword(text, 'bill to', num_lines=4)
        
        if len(lines) > 1:
            # Skip first line (customer name), join rest as address
            address = ', '.join(lines[1:])
            logger.debug(f"Found customer address: {address}")
            return address
        
        return None
    
    def _extract_currency(self, text: str) -> str:
        """Extract currency code"""
        currency_map = {
            'usd': 'USD', 'us$': 'USD', 'dollar': 'USD',
            'eur': 'EUR', '€': 'EUR', 'euro': 'EUR',
            'gbp': 'GBP', '£': 'GBP', 'pound': 'GBP',
            'zar': 'ZAR', 'rand': 'ZAR',
            'cad': 'CAD', 'aud': 'AUD', 'jpy': 'JPY',
        }
        
        text_lower = text.lower()
        
        for key, value in currency_map.items():
            if key in text_lower:
                logger.debug(f"Found currency: {value}")
                return value
        
        # Check for currency symbols at start of amounts
        if '$' in text:
            return 'USD'
        elif '€' in text:
            return 'EUR'
        elif '£' in text:
            return 'GBP'
        elif 'R' in text and 'ZAR' not in text.upper():
            return 'ZAR'
        
        return 'USD'  # Default
    
    def _extract_payment_terms(self, text: str) -> Optional[str]:
        """Extract payment terms"""
        patterns = [
            r'(?:payment\s*terms?|terms)\s*:?\s*(.{5,50})',
            r'(?:net\s*\d+|due\s*on\s*receipt)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                terms = match.group(0).strip()
                logger.debug(f"Found payment terms: {terms}")
                return terms
        
        return None
    
    def _extract_line_items(self, tables: List[Dict]) -> List[Dict]:
        """Extract line items from tables"""
        line_items = []
        
        for table in tables:
            headers = [str(h).lower() for h in table.get('headers', [])]
            data = table.get('data', [])
            
            # Check if this looks like a line items table
            has_description = any('description' in h or 'item' in h or 'product' in h for h in headers)
            has_amount = any('amount' in h or 'price' in h or 'total' in h for h in headers)
            
            if has_description and has_amount and len(data) > 0:
                logger.debug(f"Processing line items table with {len(data)} rows")
                
                # Try to identify column indices
                desc_idx = self._find_column_index(headers, ['description', 'item', 'product'])
                qty_idx = self._find_column_index(headers, ['qty', 'quantity', 'amount'])
                price_idx = self._find_column_index(headers, ['price', 'rate', 'unit'])
                total_idx = self._find_column_index(headers, ['total', 'amount'])
                
                for row in data:
                    try:
                        # Skip header rows or empty rows
                        if not row or len(row) == 0:
                            continue
                        
                        # Skip if first cell looks like a header
                        if any(keyword in str(row[0]).lower() for keyword in ['description', 'item', 'total']):
                            continue
                        
                        item = {
                            "description": self._clean_cell_value(row[desc_idx]) if desc_idx < len(row) else "",
                            "quantity": self._parse_number(row[qty_idx]) if qty_idx < len(row) else 1.0,
                            "unit_price": self._parse_number(row[price_idx]) if price_idx < len(row) else 0.0,
                            "total": self._parse_number(row[total_idx]) if total_idx < len(row) else 0.0
                        }
                        
                        # Only add if we have a description
                        if item["description"] and len(item["description"]) > 2:
                            line_items.append(item)
                            logger.debug(f"Added line item: {item['description']}")
                    
                    except (ValueError, IndexError) as e:
                        logger.warning(f"Could not parse line item row: {e}")
                        continue
        
        logger.info(f"Extracted {len(line_items)} line items")
        return line_items
    
    def _find_column_index(self, headers: List[str], keywords: List[str]) -> int:
        """Find column index by keywords"""
        for i, header in enumerate(headers):
            if any(keyword in header for keyword in keywords):
                return i
        return 0  # Default to first column
    
    def _clean_cell_value(self, value) -> str:
        """Clean cell value"""
        if value is None:
            return ""
        return str(value).strip()
    
    def _parse_number(self, value) -> float:
        """Parse numeric value from cell"""
        if value is None:
            return 0.0
        
        # Convert to string and clean
        value_str = str(value).replace(',', '').replace('$', '').replace('R', '').strip()
        
        try:
            return float(value_str)
        except ValueError:
            return 0.0


# ============ TESTING ============

if __name__ == "__main__":
    sample_text = """
    ACME CORPORATION
    123 Business Street, Suite 100
    New York, NY 10001
    Phone: +1 (555) 123-4567
    Email: billing@acme.com
    Tax ID: 12-3456789
    
    INVOICE
    
    Invoice Number: INV-2024-001
    Invoice Date: January 15, 2024
    Due Date: February 15, 2024
    PO Number: PO-2024-456
    
    Bill To:
    John Doe Enterprises
    456 Client Avenue
    Los Angeles, CA 90001
    
    Description                    Qty    Unit Price    Total
    Web Development Services       40     $150.00       $6,000.00
    Monthly Hosting                 1     $50.00        $50.00
    SSL Certificate                 1     $100.00       $100.00
    
    Subtotal:                                           $6,150.00
    Tax (10%):                                          $615.00
    Total Amount Due:                                   $6,765.00
    
    Payment Terms: Net 30
    """
    
    sample_tables = [{
        'headers': ['Description', 'Qty', 'Unit Price', 'Total'],
        'data': [
            ['Web Development Services', '40', '$150.00', '$6,000.00'],
            ['Monthly Hosting', '1', '$50.00', '$50.00'],
            ['SSL Certificate', '1', '$100.00', '$100.00'],
        ]
    }]
    
    extractor = InvoiceExtractor()
    result = extractor.extract(sample_text, sample_tables)
    
    print("\n" + "="*70)
    print("INVOICE EXTRACTION TEST RESULTS")
    print("="*70)
    
    print(f"\n📋 Invoice Details:")
    print(f"  Invoice Number: {result['invoice_number']}")
    print(f"  Date: {result['invoice_date']}")
    print(f"  Due Date: {result['due_date']}")
    print(f"  PO Number: {result['po_number']}")
    
    print(f"\n🏢 Vendor:")
    print(f"  Name: {result['vendor_name']}")
    print(f"  Email: {result['vendor_email']}")
    print(f"  Phone: {result['vendor_phone']}")
    print(f"  Tax ID: {result['vendor_tax_id']}")
    
    print(f"\n💰 Amounts:")
    print(f"  Subtotal: {result['currency']} {result['subtotal']}")
    print(f"  Tax: {result['currency']} {result['tax_amount']}")
    print(f"  Total: {result['currency']} {result['total_amount']}")
    
    print(f"\n📦 Line Items: {len(result['line_items'])}")
    for i, item in enumerate(result['line_items'], 1):
        print(f"  {i}. {item['description']}: {item['quantity']} × ${item['unit_price']} = ${item['total']}")
    
    print(f"\n✅ Confidence: {result['confidence']:.2%}")