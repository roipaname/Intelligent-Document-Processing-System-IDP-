from pydantic import BaseModel,Field,validator
from typing import Optional,List
from datetime import datetime
from enum import Enum


#==== ENUMS ====#
class DocumentType(str,Enum):
    INVOICE="invoice"
    CONTRACT="contract"
    RECEIPT="receipt"
    OTHER="other"

class DocumentStatus(str,Enum):
    PENDING="pending"
    PROCESSING="processing"
    COMPLETED="completed"
    FAILED="failed"

class ValidationStatus(str,Enum):
    VALID="valid"
    INVALID="invalid"
    NEEDS_REVIEW="needs_review"

# ============ REQUEST SCHEMAS ============
class DocumentUploadResponse(BaseModel):
    documment_id:str
    filename:str
    file_size:int
    document_type:Optional[DocumentType]
    status:DocumentStatus
    uploaded_at:datetime
    message:str

class DocumentProcessRequest(BaseModel):
    document_type:Optional[DocumentType]=None
    extract_tables:bool=True
    run_validation:bool=True


# ============ RESPONSE SCHEMAS ============
class BoundingBox(BaseModel):
    x:int
    y:int
    width:int
    height:int
    page:int=1

class ExtractedField(BaseModel):
    field_name:str
    value:str
    confidence:float=Field(...,ge=0,le=1)
    bounding_box:Optional[BoundingBox]=None
    extraction_method:str=None
class LineItem(BaseModel):
    description:str
    quantity:float
    unit_price:float
    total:float
class InvoiceData(BaseModel):
    vendor_name:str
    vendor_address:Optional[str]=None
    invoice_number:str
    invoice_date:datetime
    due_date:Optional[datetime]=None
    po_number:Optional[str]=None
    total_amount:float
    currency:str="ZAR"
    tax_amount:Optional[float]=0.0
    line_items:List[LineItem]=[]
    overall_confidence:float=Field(...,ge=0,le=1)
    subtotal:Optional[float]=None

    @validator('total_amount')
    def validate_total(cls,v,values):
        subtotal=values.get('subtotal',0.0)
        tax=values.get('tax_amount',0.0)
        if subtotal is not None and (subtotal + tax)!=v:
            raise ValueError("Total amount does not match subtotal + tax amount")
        if v and v<0:
            raise ValueError("Total amount must be non-negative")
        return v
class ValidationRule(BaseModel):
    rule_name:str
    passed:bool
    message:Optional[str]=None
    severity:Optional[str]=None
    field_name:Optional[str]=None


class DocumentDetailResponse(BaseModel):
    document_id: str
    filename: str
    document_type: Optional[DocumentType]
    status: DocumentStatus
    uploaded_at: datetime
    processed_at: Optional[datetime]
    processing_time_ms: Optional[int]
    
    # Extracted data
    extracted_fields: List[ExtractedField] = []
    invoice_data: Optional[InvoiceData]
    
    # Quality metrics
    overall_confidence: Optional[float]
    validation_results: List[ValidationRule] = []
    validation_status: Optional[ValidationStatus]
    
    class Config:
        from_attributes = True  # For SQLAlchemy models

class DocumentListResponse(BaseModel):
    total:int
    page:int
    page_size:int
    documents:List[DocumentDetailResponse]

# ============ ERROR SCHEMAS ============

class ErrorResponse(BaseModel):
    error: str
    detail: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)