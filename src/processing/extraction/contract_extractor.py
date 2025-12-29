#src/processing/extraction/contract_extractor.py
from typing import Dict, List, Optional
import re
import logging
from .base_extractor import BaseExtractor

logger = logging.getLogger(__name__)


class ContractExtractor(BaseExtractor):
    """
    Extract structured data from contracts
    
    Extracts:
    - Contract number/ID
    - Effective date, expiration date
    - Parties involved
    - Contract value
    - Key terms and clauses
    - Signatures
    """
    
    def __init__(self):
        """Initialize contract-specific patterns"""
        super().__init__()
        
        # Contract number patterns
        self.contract_patterns = [
            r'contract\s*#?\s*:?\s*([A-Z0-9\-/]+)',
            r'agreement\s*#?\s*:?\s*([A-Z0-9\-/]+)',
            r'contract\s*number\s*:?\s*([A-Z0-9\-/]+)',
        ]
        
        logger.info("Initialized ContractExtractor")
    
    def extract(self, text: str, tables: List[Dict] = None) -> Dict:
        """
        Extract all contract fields
        
        Args:
            text: Raw extracted text
            tables: Extracted tables (optional)
            
        Returns:
            Dictionary with contract data
        """
        logger.info("Starting contract extraction...")
        
        contract_data = {
            # Identifiers
            "contract_number": self._extract_contract_number(text),
            "title": self._extract_title(text),
            
            # Dates
            "effective_date": self.extract_date(text, ['effective date', 'start date', 'commencement']),
            "expiration_date": self.extract_date(text, ['expiration', 'end date', 'termination date']),
            "execution_date": self.extract_date(text, ['executed', 'signed', 'dated']),
            
            # Parties
            "parties": self._extract_parties(text),
            
            # Financial
            "total_value": self._extract_contract_value(text),
            "currency": self._extract_currency(text),
            
            # Key terms
            "key_terms": self._extract_key_terms(text),
            "termination_clause": self._extract_termination_clause(text),
            
            # Governance
            "governing_law": self._extract_governing_law(text),
        }
        
        # Calculate confidence
        required_fields = ['contract_number', 'parties', 'effective_date']
        confidence = self.calculate_confidence(contract_data, required_fields)
        contract_data["confidence"] = confidence
        
        logger.info(f"✅ Contract extraction complete. Confidence: {confidence:.2%}")
        
        return contract_data
    
    def _extract_contract_number(self, text: str) -> Optional[str]:
        """Extract contract number"""
        for pattern in self.contract_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                contract_num = match.group(1).upper().strip()
                logger.debug(f"Found contract number: {contract_num}")
                return contract_num
        
        return None
    
    def _extract_title(self, text: str) -> Optional[str]:
        """Extract contract title"""
        lines = text.split('\n')
        
        # Title often appears in first few lines or after "AGREEMENT" keyword
        for i, line in enumerate(lines[:10]):
            line = line.strip()
            
            # Look for lines with "AGREEMENT" or "CONTRACT"
            if 'agreement' in line.lower() or 'contract' in line.lower():
                # This might be the title
                if len(line) > 10 and len(line) < 100:
                    logger.debug(f"Found title: {line}")
                    return line
        
        return None
    
    def _extract_parties(self, text: str) -> List[Dict]:
        """Extract contracting parties"""
        parties = []
        
        # Look for common party indicators
        party_keywords = [
            'between', 'party', 'parties', 'hereinafter', 
            'vendor', 'client', 'buyer', 'seller', 'contractor'
        ]
        
        lines = text.split('\n')
        
        for i, line in enumerate(lines):
            line_lower = line.lower()
            
            # Check for party introductions
            if any(keyword in line_lower for keyword in party_keywords):
                # Extract next few lines as potential party info
                party_lines = self.extract_lines_after_keyword(text, line, num_lines=3)
                
                if party_lines:
                    party = {
                        "name": party_lines[0] if len(party_lines) > 0 else "",
                        "address": ', '.join(party_lines[1:]) if len(party_lines) > 1 else "",
                        "role": self._determine_party_role(line_lower)
                    }
                    
                    if party["name"]:
                        parties.append(party)
                        logger.debug(f"Found party: {party['name']} ({party['role']})")
        
        return parties[:2]  # Usually just 2