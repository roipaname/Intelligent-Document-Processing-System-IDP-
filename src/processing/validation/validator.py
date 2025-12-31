# src/processing/validation/validator.py
"""
Business rule validation for extracted documents
"""
from typing import Dict, List
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class ValidationResult:
    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def is_valid(self) -> bool:
        return len(self.errors) == 0

    def to_dict(self) -> Dict:
        return {
            "is_valid": self.is_valid(),
            "errors": self.errors,
            "warnings": self.warnings,
        }


class BusinessValidator:
    """
    Applies business rules to extracted data
    """

    def validate_invoice(self, invoice: Dict) -> ValidationResult:
        result = ValidationResult()

        # Required fields
        if not invoice.get("invoice_number"):
            result.errors.append("Missing invoice number")

        if not invoice.get("vendor_name"):
            result.errors.append("Missing vendor name")

        if not invoice.get("total_amount"):
            result.errors.append("Missing total amount")

        # Date consistency
        invoice_date = invoice.get("invoice_date")
        due_date = invoice.get("due_date")

        if invoice_date and due_date:
            if due_date < invoice_date:
                result.errors.append("Due date is earlier than invoice date")

        # Amount consistency
        subtotal = invoice.get("subtotal")
        tax = invoice.get("tax_amount")
        total = invoice.get("total_amount")

        if subtotal and tax and total:
            if round(subtotal + tax, 2) != round(total, 2):
                result.warnings.append(
                    "Subtotal + tax does not equal total amount"
                )

        logger.debug(f"Invoice validation result: {result.to_dict()}")
        return result

    def validate_contract(self, contract: Dict) -> ValidationResult:
        result = ValidationResult()

        if not contract.get("title"):
            result.errors.append("Missing contract title")

        if not contract.get("parties"):
            result.errors.append("No contract parties detected")

        # Date logic
        start = contract.get("effective_date")
        end = contract.get("expiration_date")

        if start and end and end < start:
            result.errors.append("Expiration date precedes effective date")

        # Financial sanity
        value = contract.get("contract_value")
        if value is not None and value <= 0:
            result.warnings.append("Contract value is zero or negative")

        logger.debug(f"Contract validation result: {result.to_dict()}")
        return result
