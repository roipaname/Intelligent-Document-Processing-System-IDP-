# src/processing/extraction/contract_extractor.py
"""
Contract-specific field extraction
"""
import re
from typing import Dict, List, Optional
from datetime import datetime
import logging

from .base_extractor import BaseExtractor

logger = logging.getLogger(__name__)


class ContractExtractor(BaseExtractor):
    """
    Extract structured contract data from text

    Extracts:
    - Contract number
    - Title
    - Parties involved
    - Effective and expiration dates
    - Contract value
    - Key terms and clauses
    """

    def __init__(self):
        """Initialize contract-specific patterns"""
        super().__init__()

        # Contract number patterns
        self.contract_number_patterns = [
            r'contract\s*#?\s*:?\s*([A-Z0-9\-/]+)',
            r'agreement\s*#?\s*:?\s*([A-Z0-9\-/]+)',
            r'contract\s*number\s*:?\s*([A-Z0-9\-/]+)',
            r'agreement\s*number\s*:?\s*([A-Z0-9\-/]+)',
        ]

        # Currency map (kept consistent with InvoiceExtractor)
        self.currency_map = {
            'usd': 'USD', 'us$': 'USD', '$': 'USD', 'dollar': 'USD',
            'eur': 'EUR', '€': 'EUR', 'euro': 'EUR',
            'gbp': 'GBP', '£': 'GBP', 'pound': 'GBP',
            'zar': 'ZAR', 'rand': 'ZAR',
            'inr': 'INR', '₹': 'INR', 'rupee': 'INR',
            'jpy': 'JPY', '¥': 'JPY', 'yen': 'JPY',
        }

        logger.info("Initialized ContractExtractor")

    def extract(self, text: str, tables: List[Dict] = None, **kwargs) -> Dict:
        """
        Extract contract data from text

        Args:
            text: Raw text from OCR
            tables: Extracted tables (unused but kept for interface consistency)
            **kwargs: Additional parameters

        Returns:
            Dictionary with extracted contract fields
        """
        logger.info("Extracting contract data...")

        contract_data = {
            # Identifiers
            "contract_number": self._extract_contract_number(text),
            "title": self._extract_title(text),

            # Dates
            "effective_date": self.extract_date(
                text, ["effective date", "start date", "commencement"]
            ),
            "expiration_date": self.extract_date(
                text, ["expiration", "end date", "termination date"]
            ),

            # Parties
            "parties": self._extract_parties(text),

            # Financials
            "contract_value": self.extract_amount(
                text, ["contract value", "total contract", "agreement value", "amount"]
            ),
            "currency": self._extract_currency(text),

            # Terms & clauses
            "key_terms": self._extract_key_terms(text),
        }

        # Calculate confidence
        required_fields = ["title", "parties", "effective_date"]
        confidence = self.calculate_confidence(contract_data, required_fields)
        contract_data["overall_confidence"] = confidence

        logger.info(f"Contract extraction complete. Confidence: {confidence:.2%}")

        return contract_data

    def _extract_contract_number(self, text: str) -> Optional[str]:
        """Extract contract number"""
        number = self.find_pattern(text, self.contract_number_patterns)

        if number:
            number = number.strip().upper()
            logger.debug(f"Found contract number: {number}")
            return number

        logger.warning("Contract number not found")
        return None

    def _extract_title(self, text: str) -> Optional[str]:
        """Extract contract title"""
        lines = text.split('\n')

        # Titles usually appear at the top
        for line in lines[:10]:
            line = line.strip()

            if re.search(r'\b(agreement|contract)\b', line, re.IGNORECASE):
                if 5 < len(line) < 120:
                    logger.debug(f"Found contract title: {line}")
                    return line

        return None

    def _extract_parties(self, text: str) -> List[Dict]:
        """
        Extract parties involved in the contract

        Returns:
            [
                {"name": "...", "role": "party_a"},
                {"name": "...", "role": "party_b"}
            ]
        """
        parties = []

        # Common legal phrasing: "between X and Y"
        between_pattern = (
            r'between\s+(.+?)\s+and\s+(.+?)'
            r'(?:\s+(?:dated|effective|as of|on)|\n)'
        )

        match = re.search(between_pattern, text, re.IGNORECASE | re.DOTALL)

        if match:
            party_a = match.group(1).strip().strip(',').strip()
            party_b = match.group(2).strip().strip(',').strip()

            parties.append({"name": party_a, "role": "party_a"})
            parties.append({"name": party_b, "role": "party_b"})

            logger.debug(f"Found parties: {party_a} AND {party_b}")
            return parties

        # Fallback: look for "Party A / Party B" or "Client / Provider"
        fallback_patterns = [
            r'(client|customer)\s*:?\s*(.+)',
            r'(provider|vendor|supplier)\s*:?\s*(.+)',
        ]

        for pattern in fallback_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                role = match.group(1).lower()
                name = match.group(2).split('\n')[0].strip()

                if name and len(name) < 150:
                    parties.append({"name": name, "role": role})

        if parties:
            logger.debug(f"Found parties using fallback patterns: {parties}")

        return parties

    def _extract_currency(self, text: str) -> str:
        """Extract currency code, defaults to USD"""
        text_lower = text.lower()

        for key, value in self.currency_map.items():
            if re.search(r'\b' + re.escape(key) + r'\b', text_lower):
                logger.debug(f"Found currency: {value}")
                return value

        logger.debug("Currency not found, defaulting to USD")
        return "USD"

    def _extract_key_terms(self, text: str) -> List[str]:
        """
        Extract key terms and clauses

        Looks for common clause headers and important contractual phrases.
        """
        key_terms = []

        clause_keywords = [
            "confidentiality",
            "termination",
            "governing law",
            "liability",
            "indemnification",
            "payment terms",
            "force majeure",
            "intellectual property",
            "non-disclosure",
            "warranty",
        ]

        lines = text.split('\n')

        for line in lines:
            clean_line = line.strip()
            lower_line = clean_line.lower()

            if any(kw in lower_line for kw in clause_keywords):
                if 10 < len(clean_line) < 300:
                    key_terms.append(clean_line)

        # Deduplicate while preserving order
        seen = set()
        unique_terms = []
        for term in key_terms:
            if term not in seen:
                seen.add(term)
                unique_terms.append(term)

        logger.debug(f"Extracted {len(unique_terms)} key terms")
        return unique_terms


# ============ TESTING ============

if __name__ == "__main__":
    sample_text = """
    MASTER SERVICES AGREEMENT

    This Agreement is made between ACME Corporation and Beta Solutions Ltd
    effective as of March 1, 2024.

    Contract Number: MSA-2024-009

    Termination:
    Either party may terminate this Agreement with 30 days written notice.

    Governing Law:
    This Agreement shall be governed by the laws of the Republic of South Africa.

    Total Contract Value: ZAR 1,200,000
    """

    extractor = ContractExtractor()
    result = extractor.extract(sample_text)

    print("\n" + "=" * 70)
    print("CONTRACT EXTRACTION TEST")
    print("=" * 70)

    for key, value in result.items():
        if value:
            print(f"{key:25}: {value}")

    print("\n" + "=" * 70)
