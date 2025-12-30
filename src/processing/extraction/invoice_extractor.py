#src/processing/extraction/invoice_extractor.py
"""
Invoice-specific field extraction
"""
import re
from typing import Dict, List, Optional
from datetime import datetime
import logging

from .base_extractor import BaseExtractor

logger = logging.getLogger(__name__)


class InvoiceExtractor(BaseExtractor):
    """
    Extract structured invoice data from text
    
    Extracts:
    - Invoice number
    - Dates (invoice date, due date)
    - Vendor information (name, address, contact)
    - Customer information
    - Amounts (subtotal, tax, total)
    - Line items from tables
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
            r'ref\.?\s*#?\s*:?\s*([A-Z0-9\-/]+)',
        ]
        
        # PO number patterns
        self.po_patterns = [
            r'po\s*#?\s*:?\s*([A-Z0-9\-/]+)',
            r'purchase\s*order\s*#?\s*:?\s*([A-Z0-9\-/]+)',
            r'p\.?o\.?\s*number\s*:?\s*([A-Z0-9\-/]+)',
        ]
        
        # Tax ID patterns
        self.tax_id_patterns = [
            r'tax\s*id\s*:?\s*([A-Z0-9\-]+)',
            r'vat\s*#?\s*:?\s*([A-Z0-9\-]+)',
            r'ein\s*:?\s*([0-9\-]+)',
        ]
        
        logger.info("Initialized InvoiceExtractor")
    
    def extract(self, text: str, tables: List[Dict] = None, **kwargs) -> Dict:
        """
        Extract invoice data from text and tables
        
        Args:
            text: Raw text from OCR
            tables: Extracted tables
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with extracted invoice fields
        """
        logger.info("Extracting invoice data...")
        
        invoice_data = {
            # Document identifiers
            "invoice_number": self._extract_invoice_number(text),
            "po_number": self._extract_po_number(text),
            
            # Dates
            "invoice_date": self._extract_invoice_date(text),
            "due_date": self._extract_due_date(text),
            
            # Vendor information
            "vendor_name": self._extract_vendor_name(text),
            "vendor_address": self._extract_vendor_address(text),
            "vendor_email": self.extract_email(text),
            "vendor_phone": self.extract_phone(text),
            "vendor_tax_id": self._extract_tax_id(text),
            
            # Amounts
            "subtotal": self.extract_amount(text, ["subtotal", "sub-total", "sub total"]),
            "tax_amount": self.extract_amount(text, ["tax", "vat", "gst"]),
            "total_amount": self.extract_amount(text, ["total", "amount due", "balance", "grand total"]),
            
            # Currency
            "currency": self._extract_currency(text),
            
            # Line items
            "line_items": [],
            
            # Payment terms
            "payment_terms": self._extract_payment_terms(text),
        }
        
        # Extract line items from tables
        if tables:
            invoice_data["line_items"] = self._extract_line_items(tables)
        
        # Calculate missing values
        invoice_data = self._calculate_missing_amounts(invoice_data)
        
        # Calculate confidence
        required_fields = ['invoice_number', 'total_amount', 'vendor_name']
        confidence = self.calculate_confidence(invoice_data, required_fields)
        invoice_data["overall_confidence"] = confidence
        
        logger.info(f"Invoice extraction complete. Confidence: {confidence:.2%}")
        
        return invoice_data
    
    def _extract_invoice_number(self, text: str) -> Optional[str]:
        """Extract invoice number"""
        invoice_num = self.find_pattern(text, self.invoice_number_patterns)
        
        if invoice_num:
            invoice_num = invoice_num.strip().upper()
            logger.debug(f"Found invoice number: {invoice_num}")
            return invoice_num
        
        logger.warning("Invoice number not found")
        return None
    
    def _extract_po_number(self, text: str) -> Optional[str]:
        """Extract purchase order number"""
        po_num = self.find_pattern(text, self.po_patterns)
        
        if po_num:
            po_num = po_num.strip().upper()
            logger.debug(f"Found PO number: {po_num}")
            return po_num
        
        return None
    
    def _extract_invoice_date(self, text: str) -> Optional[datetime]:
        """Extract invoice date"""
        date = self.extract_date(text, ["invoice date", "date", "issued", "bill date"])
        
        if date:
            logger.debug(f"Found invoice date: {date}")
        else:
            logger.warning("Invoice date not found")
        
        return date
    
    def _extract_due_date(self, text: str) -> Optional[datetime]:
        """Extract due date"""
        date = self.extract_date(text, ["due date", "payment due", "due by"])
        
        if date:
            logger.debug(f"Found due date: {date}")
        
        return date
    
    def _extract_vendor_name(self, text: str) -> Optional[str]:
        """
        Extract vendor/company name
        Usually appears at the top of the invoice
        """
        lines = text.split('\n')
        
        # Skip empty lines
        non_empty_lines = [l.strip() for l in lines if l.strip()]
        
        if not non_empty_lines:
            return None
        
        # First non-empty line is often the vendor name
        for line in non_empty_lines[:5]:
            # Skip lines that are clearly not company names
            if any(keyword in line.lower() for keyword in ['invoice', 'bill', 'date', 'page']):
                continue
            
            # Skip lines with only numbers or special characters
            if re.match(r'^[\d\W]+$', line):
                continue
            
            # Check if it looks like a company name
            if 3 < len(line) < 100 and len(line.split()) <= 6:
                logger.debug(f"Found vendor name: {line}")
                return line
        
        return None
    
    def _extract_vendor_address(self, text: str) -> Optional[str]:
        """Extract vendor address"""
        lines = text.split('\n')
        
        address_lines = []
        found_address_start = False
        
        for i, line in enumerate(lines[:15]):  # Check first 15 lines
            line = line.strip()
            
            # Look for address indicators
            if any(word in line.lower() for word in ['street', 'road', 'avenue', 'drive', 'suite', 'floor', 'building']):
                found_address_start = True
                address_lines.append(line)
                
                # Get next 2-3 lines
                for j in range(i+1, min(i+4, len(lines))):
                    next_line = lines[j].strip()
                    if next_line and not any(kw in next_line.lower() for kw in ['invoice', 'bill', 'date', 'customer']):
                        address_lines.append(next_line)
                    else:
                        break
                
                break
        
        if address_lines:
            address = ', '.join(address_lines)
            logger.debug(f"Found address: {address}")
            return address
        
        return None
    
    def _extract_tax_id(self, text: str) -> Optional[str]:
        """Extract tax ID / VAT number"""
        tax_id = self.find_pattern(text, self.tax_id_patterns)
        
        if tax_id:
            logger.debug(f"Found tax ID: {tax_id}")
            return tax_id
        
        return None
    
    def _extract_currency(self, text: str) -> str:
        """
        Extract currency code from text
        Defaults to USD if not found
        """
        currency_map = {
            'usd': 'USD', 'us$': 'USD', '$': 'USD', 'dollar': 'USD',
            'eur': 'EUR', '€': 'EUR', 'euro': 'EUR',
            'gbp': 'GBP', '£': 'GBP', 'pound': 'GBP',
            'zar': 'ZAR', 'rand': 'ZAR',
            'inr': 'INR', '₹': 'INR', 'rupee': 'INR',
            'jpy': 'JPY', '¥': 'JPY', 'yen': 'JPY',
        }
        
        text_lower = text.lower()
        
        # Check for explicit currency codes
        for key, value in currency_map.items():
            if re.search(r'\b' + key + r'\b', text_lower):
                logger.debug(f"Found currency: {value}")
                return value
        
        # Default to USD
        logger.debug("Currency not found, defaulting to USD")
        return "USD"
    
    def _extract_payment_terms(self, text: str) -> Optional[str]:
        """Extract payment terms (e.g., Net 30, Due on receipt)"""
        patterns = [
            r'(net\s+\d+)',
            r'(due\s+on\s+receipt)',
            r'(payment\s+terms?\s*:?\s*[^\n]+)',
        ]
        
        terms = self.find_pattern(text, patterns)
        
        if terms:
            logger.debug(f"Found payment terms: {terms}")
            return terms.strip()
        
        return None
    
    def _extract_line_items(self, tables: List[Dict]) -> List[Dict]:
        """
        Extract line items from tables
        
        Args:
            tables: List of extracted tables
            
        Returns:
            List of line item dictionaries
        """
        line_items = []
        
        for table in tables:
            headers = [str(h).lower() for h in table.get('headers', [])]
            data = table.get('data', [])
            
            # Check if this looks like a line items table
            has_description = any(kw in ' '.join(headers) for kw in ['description', 'item', 'product', 'service'])
            has_amount = any(kw in ' '.join(headers) for kw in ['amount', 'price', 'total', 'cost'])
            
            if has_description and has_amount and len(data) > 0:
                logger.debug(f"Found line items table with {len(data)} rows")
                
                # Find column indices
                desc_idx = next((i for i, h in enumerate(headers) if any(kw in h for kw in ['description', 'item', 'product'])), 0)
                qty_idx = next((i for i, h in enumerate(headers) if 'qty' in h or 'quantity' in h), None)
                price_idx = next((i for i, h in enumerate(headers) if 'price' in h or 'rate' in h), None)
                total_idx = next((i for i, h in enumerate(headers) if 'total' in h or 'amount' in h), -1)
                
                for row in data:
                    try:
                        # Extract description
                        description = str(row[desc_idx]).strip() if desc_idx < len(row) else ""
                        
                        # Skip empty or header rows
                        if not description or description.lower() in headers:
                            continue
                        
                        # Extract quantity
                        quantity = 1.0
                        if qty_idx is not None and qty_idx < len(row):
                            qty_str = str(row[qty_idx]).replace(',', '').strip()
                            try:
                                quantity = float(qty_str)
                            except ValueError:
                                pass
                        
                        # Extract unit price
                        unit_price = 0.0
                        if price_idx is not None and price_idx < len(row):
                            price_str = str(row[price_idx]).replace(',', '').replace('$', '').strip()
                            try:
                                unit_price = float(price_str)
                            except ValueError:
                                pass
                        
                        # Extract total
                        total = 0.0
                        if total_idx < len(row):
                            total_str = str(row[total_idx]).replace(',', '').replace('$', '').strip()
                            try:
                                total = float(total_str)
                            except ValueError:
                                # If total not found, calculate it
                                total = quantity * unit_price
                        
                        item = {
                            "description": description,
                            "quantity": quantity,
                            "unit_price": unit_price,
                            "total": total
                        }
                        
                        line_items.append(item)
                        
                    except (ValueError, IndexError) as e:
                        logger.warning(f"Could not parse line item: {e}")
                        continue
        
        logger.info(f"Extracted {len(line_items)} line items")
        return line_items
    
    def _calculate_missing_amounts(self, invoice_data: Dict) -> Dict:
        """
        Calculate missing amounts based on available data
        
        Args:
            invoice_data: Invoice data dictionary
            
        Returns:
            Updated invoice data with calculated amounts
        """
        subtotal = invoice_data.get('subtotal')
        tax = invoice_data.get('tax_amount')
        total = invoice_data.get('total_amount')
        
        # If we have line items, calculate subtotal
        if not subtotal and invoice_data.get('line_items'):
            subtotal = sum(item['total'] for item in invoice_data['line_items'])
            invoice_data['subtotal'] = round(subtotal, 2)
            logger.debug(f"Calculated subtotal from line items: {subtotal}")
        
        # Calculate missing values
        if subtotal and tax and not total:
            invoice_data['total_amount'] = round(subtotal + tax, 2)
            logger.debug(f"Calculated total: {invoice_data['total_amount']}")
        
        elif total and tax and not subtotal:
            invoice_data['subtotal'] = round(total - tax, 2)
            logger.debug(f"Calculated subtotal: {invoice_data['subtotal']}")
        
        elif total and subtotal and not tax:
            invoice_data['tax_amount'] = round(total - subtotal, 2)
            logger.debug(f"Calculated tax: {invoice_data['tax_amount']}")
        
        return invoice_data


# ============ TESTING ============

if __name__ == "__main__":
    sample_text = """
    ACME Corporation
    123 Business Street, Suite 100
    New York, NY 10001
    Tax ID: 12-3456789
    Phone: (555) 123-4567
    Email: billing@acme.com
    
    INVOICE
    
    Invoice Number: INV-2024-001
    Invoice Date: January 15, 2024
    Due Date: February 15, 2024
    PO Number: PO-2024-100
    
    Bill To:
    John Doe
    456 Client Avenue
    Los Angeles, CA 90001
    
    Description              Qty    Unit Price    Total
    Web Development          10     $100.00       $1,000.00
    Hosting Services         1      $50.00        $50.00
    Domain Registration      1      $15.00        $15.00
    
    Subtotal:                                     $1,065.00
    Tax (10%):                                    $106.50
    Total Amount Due:                             $1,171.50
    
    Payment Terms: Net 30
    """
    
    sample_tables = [{
        'headers': ['Description', 'Qty', 'Unit Price', 'Total'],
        'data': [
            ['Web Development', '10', '100.00', '1000.00'],
            ['Hosting Services', '1', '50.00', '50.00'],
            ['Domain Registration', '1', '15.00', '15.00']
        ]
    }]
    
    extractor = InvoiceExtractor()
    result = extractor.extract(sample_text, sample_tables)
    
    print("\n" + "="*70)
    print("INVOICE EXTRACTION TEST")
    print("="*70)
    
    for key, value in result.items():
        if value and key != 'line_items':
            print(f"{key:25}: {value}")
    
    if result.get('line_items'):
        print(f"\n{'Line Items:':<25} {len(result['line_items'])} items")
        for i, item in enumerate(result['line_items'], 1):
            print(f"  {i}. {item['description']:<30} ${item['total']:>10.2f}")
    
    print("\n" + "="*70)