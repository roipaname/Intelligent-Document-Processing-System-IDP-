from fastapi import FastAPI,UploadFile,File,HTTPException,Depends,Query,BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse,FileResponse
from sqlalchemy.orm import Session
from sqlalchemy import text
from src.models.database import get_db,Document,ProcessingLog
from typing import List,Optional
import shutil
from pathlib import Path
import uuid
from datetime import datetime
import time
from src.api.schemas import (
    DocumentUploadResponse, DocumentDetailResponse, 
    DocumentListResponse, ErrorResponse, DocumentType, DocumentStatus
)



# ============ APP INITIALIZATION ============
app=FastAPI(
    title="Intelligent Document Processing System (IDP) API",
    description="API for uploading, processing, and retrieving documents using Intelligent Document Processing.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"

)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"])

# File Upload Config
UPLOAD_DIR=Path("./data/uploads/")
UPLOAD_DIR.mkdir(parents=True,exist_ok=True)
MAX_FILE_SIZE=10*1024*1024  # 10 MB
ALLOWED_EXTENSIONS=[".pdf",".png",".jpg",".jpeg",".tiff",".docx"]


# ============ APP INITIALIZATION ============

@app.middleware("http")
async def log_requests(request,call_next):
    start_time=time.time()
    response= await call_next(request)
    process_time=time.time()-start_time
    formatted_process_time=f"{process_time:.4f}s"
    print(f"{request.method} {request.url.path} completed_in={formatted_process_time} ")
    return response


# ============ EXCEPTION HANDLERS ============
@app.exception_handler(HTTPException)
async def http_exception_handler(request,exc):
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(error=exc.detail,detail=str(exc)).dict()
    )

@app.exception_handler(Exception)
async def general_exception_handler(request,exc):
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(error="Internal Server Error",detail=str(exc)).dict()
    )

# ============ HELPER FUNCTIONS ============

def validate_file(file: UploadFile) -> None:
    """Validate uploaded file"""
    # Check extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            400, 
            f"File type not supported. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    # Check size (if available)
    if hasattr(file, 'size') and file.size > MAX_FILE_SIZE:
        raise HTTPException(
            400,
            f"File too large. Maximum size: {MAX_FILE_SIZE / 1024 / 1024}MB"
        )
    
def save_upload_file(file: UploadFile, document_id: str) -> Path:
    """Save uploaded file to destination"""
    file_ext=Path(file.filename).suffix.lower()
    dest_path=UPLOAD_DIR / f"{document_id}{file_ext}"

    try:
        with dest_path.open("wb") as buffer:
            shutil.copyfileobj(file.file,buffer)
    finally:
        file.file.close()

    return dest_path
    
async def process_document_async(document_id: str, db: Session):
    """Background task to process document"""
    # Import here to avoid circular imports
    from src.processing.pipeline import DocumentProcessor
    
    try:
        doc = db.query(Document).filter_by(id=document_id).first()
        if not doc:
            return
        
        # Update status
        doc.status = DocumentStatus.PROCESSING
        db.commit()
        
        # Log start
        log = ProcessingLog(
            document_id=document_id,
            stage="processing",
            status="started"
        )
        db.add(log)
        db.commit()
        
        # Process document
        processor = DocumentProcessor()
        start_time = time.time()
        result = processor.process(doc.file_path)
        duration_ms = int((time.time() - start_time) * 1000)
        
        # Update document
        doc.status = DocumentStatus.COMPLETED
        doc.processed_at = datetime.utcnow()
        
        # Log completion
        log.status = "completed"
        log.duration_ms = duration_ms
        log.details = {"fields_extracted": len(result)}
        
        db.commit()
        
    except Exception as e:
        doc.status = DocumentStatus.FAILED
        doc.error_message = str(e)
        
        log.status = "failed"
        log.details = {"error": str(e)}
        
        db.commit()

# ============ ROUTES ============

@app.get("/",tags=["Root"])
def root():
    """API root endpoint"""
    return {
        "name": "Intelligent Document Processing System (IDP) API",
        "version": "1.0.0",
        "description": "API for uploading, processing, and retrieving documents using Intelligent Document Processing.",
        "documentation": "/docs",
        "endpoints":{
            "health": "/health",
            "upload":"api/v1/documents/upload",
            "list":"api/v1/documents",
            "details":"api/v1/documents/{document_id}"

        }
    }
@app.get("/health",tags=["Health"])
def health_check(db: Session = Depends(get_db)):
    """
    Health check endpoint
    Checks:
    - API status
    - Database connectivity
    - File system access
    """
    health_status= {"status":"healthy","timestamp":datetime.utcnow().isoformat(),"checks":{}}

    try:
        db.execute(text("SELECT 1"))
        health_status["checks"]["database"]="connected"
    except Exception as e:
        health_status["checks"]["database"]=f"error: {str(e)}"
        health_status["status"]="unhealthy"

    #check file system
    try:
        UPLOAD_DIR.exists()
        health_status["checks"]["file_system"]="accessible"
    except Exception as e:
        health_status["checks"]["file_system"]=f"error: {str(e)}"
        health_status["status"]="unhealthy"

    return health_status

@app.post(
    "/api/v1/documents/upload",
    response_model=DocumentUploadResponse,
    status_code=201,
    tags=["Documents"],
    summary="Upload a document for processing"
)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="PDF or image file to process"),
    document_type: Optional[DocumentType] = Query(None, description="Type of document (invoice, contract, receipt)"),
    auto_process: bool = Query(True, description="Automatically start processing after upload"),
    db: Session = Depends(get_db)
):
    """
    Upload a document for OCR and data extraction
    
    - **file**: PDF or image file (max 10MB)
    - **document_type**: Optional hint about document type
    - **auto_process**: If True, processing starts immediately in background
    
    Returns document ID and upload status
    """
    # Validate file
    validate_file(file)
    
    # Generate unique document ID
    doc_id = str(uuid.uuid4())
    
    # Save file to disk
    try:
        file_path = save_upload_file(file, doc_id)
        file_size = file_path.stat().st_size
    except Exception as e:
        raise HTTPException(500, f"Failed to save file: {str(e)}")
    
    # Create database record
    document = Document(
        id=doc_id,
        filename=file.filename,
        file_path=str(file_path),
        file_size_bytes=file_size,
        mime_type=file.content_type,
        document_type=document_type.value if document_type else None,
        status=DocumentStatus.PENDING.value
    )
    
    try:
        db.add(document)
        db.commit()
        db.refresh(document)
    except Exception as e:
        # Clean up file if database insert fails
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(500, f"Database error: {str(e)}")
    
    # Start background processing
    if auto_process:
        background_tasks.add_task(process_document_async, doc_id, db)
        message = "Document uploaded successfully and processing started"
    else:
        message = "Document uploaded successfully. Call /process endpoint to start processing"
    
    return DocumentUploadResponse(
        document_id=doc_id,
        filename=file.filename,
        file_size=file_size,
        document_type=document_type,
        status=DocumentStatus.PENDING if not auto_process else DocumentStatus.PROCESSING,
        uploaded_at=document.uploaded_at,
        message=message
    )

@app.get(
    "api/v1/documents/{document_id}",
    response_model=DocumentDetailResponse,
    tags=["Documents"],
    summary="Get detailed information about a specific document"
)
def get_document(
    document_id: str,
    db: Session = Depends(get_db)
):
    """
    Retrieve detailed information about a specific document
    
    - **document_id**: UUID of the document
    
    Returns:
    - Document metadata
    - Processing status
    - Extracted data (if processing completed)
    - Validation results
    """
    document = db.query(Document).filter_by(id=document_id).first()
    
    if not document:
        raise HTTPException(404, f"Document with ID {document_id} not found")
    
    # Calculate processing time
    processing_time = None
    if document.processed_at and document.uploaded_at:
        delta = document.processed_at - document.uploaded_at
        processing_time = int(delta.total_seconds() * 1000)
    
    # Get extracted data
    extracted_fields = []
    for data in document.extracted_data:
        extracted_fields.append({
            "field_name": data.field_name,
            "value": data.field_value,
            "confidence": data.confidence_score,
            "bounding_box": data.bounding_box,
            "extraction_method": data.extraction_method
        })
    
    # Get invoice data if applicable
    invoice_data = None
    if document.invoice:
        inv = document.invoice
        invoice_data = {
            "invoice_number": inv.invoice_number,
            "invoice_date": inv.invoice_date,
            "due_date": inv.due_date,
            "vendor_name": inv.vendor_name,
            "vendor_address": inv.vendor_address,
            "subtotal": float(inv.subtotal) if inv.subtotal else None,
            "tax_amount": float(inv.tax_amount) if inv.tax_amount else None,
            "total_amount": float(inv.total_amount) if inv.total_amount else None,
            "currency": inv.currency,
            "line_items": inv.line_items or []
        }
    
    # Get validation results
    validation_results = []
    for validation in document.validations:
        validation_results.append({
            "rule_name": validation.rule_name,
            "passed": validation.passed,
            "severity": validation.severity,
            "message": validation.message,
            "field_name": validation.field_name
        })
    
    return DocumentDetailResponse(
        document_id=document.id,
        filename=document.filename,
        document_type=document.document_type,
        status=document.status,
        uploaded_at=document.uploaded_at,
        processed_at=document.processed_at,
        processing_time_ms=processing_time,
        extracted_fields=extracted_fields,
        invoice_data=invoice_data,
        overall_confidence=document.invoice.overall_confidence if document.invoice else None,
        validation_results=validation_results,
        validation_status=document.invoice.validation_status if document.invoice else None
    )

@app.get(
    "/api/v1/documents",
    response_model=DocumentListResponse,
    tags=["Documents"],
    summary="List uploaded documents with optional filters"
)
def list_documents(
    page:int =Query(1,ge=1,description="Page Number"),
    page_size:int = Query(20,ge=1,le=100,description="Number of documents per page"),
    status: Optional[DocumentStatus] = Query(None,description="Filter by document status"),
    document_type:Optional[DocumentType] = Query(None,description="Filter by document type"),
    sort_by:str= Query("uploaded_at", description="Field to sort by"),
    sort_order:str = Query("desc",description="`sort` order, either `asc` or `desc`"),
    db:Session= Depends(get_db)
                          
):
    """
    List uploaded documents with pagination and optional filters
    
    - **page**: Page number (default: 1)
    - **page_size**: Documents per page (default: 20, max: 100)
    - **status**: Filter by document status
    - **document_type**: Filter by document type
    - **sort_by**: Field to sort by (default: uploaded_at)
    - **sort_order**: Sort order, either asc or desc (default: desc)
    
    Returns paginated list of documents
    """
    query = db.query(Document)
    
    # Apply filters
    if status:
        query = query.filter(Document.status == status.value)
    if document_type:
        query = query.filter(Document.document_type == document_type.value)
    
    # Get total count
    total_documents = query.count()
    
    # Apply sorting
    sort_column = getattr(Document, sort_by, None)
    if not sort_column:
        raise HTTPException(400, f"Invalid sort_by field: {sort_by}")
    
    if sort_order.lower() == "asc":
        query = query.order_by(sort_column.asc())
    else:
        query = query.order_by(sort_column.desc())
    
    # Apply pagination
    documents = query.offset((page - 1) * page_size).limit(page_size).all()
    
    # Prepare response
    document_list = []
    for doc in documents:
        document_list.append(DocumentDetailResponse(
            document_id=doc.id,
            filename=doc.filename,
            document_type=doc.document_type,
            status=doc.status,
            uploaded_at=doc.uploaded_at,
            processed_at=doc.processed_at
        ))
    
    return DocumentListResponse(
        total=total_documents,
        page=page,
        page_size=page_size,
        documents=document_list
    )

@app.post(
    "api/v1/documents/{document_id}/process",
    tags=["Documents"],
    summary="Trigger processing for an uploaded document"
)
async def process_document(
    document_id:str,
    background_tasks:BackgroundTasks,
    db:Session=Depends(get_db),

):
    """
    Docstring for process_document
    
    :param document_id: Description
    :type document_id: str
    :param background_tasks: Description
    :type background_tasks: BackgroundTasks
    :param db: Description
    :type db: Session
    """
    document=db.query(Document).filter_by(id=document_id).first()

    if not document:
        raise HTTPException(404,f"Document with Id {document_id} not found")
    if document.status== DocumentStatus.PROCESSING.value:
        raise HTTPException(400,f"Document with Id {document_id} is laready being processed")
    if document.status == DocumentStatus.COMPLETED.value:
        raise HTTPException(400, f"Document with Id {document_id} has already been processed ")
    
    background_tasks.add_task(process_document_async,document_id,db)
    document.status=DocumentStatus.PROCESSING.value
    db.commit()

    return {"message":f"Processing started for document {document_id}" , "status":DocumentStatus.PROCESSING , "document_id":document_id }

@app.post(
    "/api/v1/documents/{document_id}/reprocess",
    tags=["Documents"],
    summary="Reprocess an already processed document"
)
async def reprocess_document(
    document_id: str,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Reprocess a document (useful for testing or if processing failed)
    """
    document = db.query(Document).filter_by(id=document_id).first()
    
    if not document:
        raise HTTPException(404, f"Document with ID {document_id} not found")
    
    # Reset status
    document.status = DocumentStatus.PROCESSING.value
    document.processed_at = None
    document.error_message = None
    db.commit()
    
    # Start background processing
    background_tasks.add_task(process_document_async, document_id, db)
    
    return {
        "document_id": document_id,
        "message": "Reprocessing started",
        "status": DocumentStatus.PROCESSING.value
    }

@app.delete(
    "/api/v1/documents/{document_id}",
    status_code=204,
    tags=["Documents"],
    summary="Delete a document"
)
def delete_document(
    document_id: str,
    db: Session = Depends(get_db)
):
    """
    Delete a document and its associated file
    
    This will also delete:
    - Extracted data
    - Validation results
    - Processing logs
    - The physical file from disk
    """
    document = db.query(Document).filter_by(id=document_id).first()
    
    if not document:
        raise HTTPException(404, f"Document with ID {document_id} not found")
    
    # Delete physical file
    try:
        file_path = Path(document.file_path)
        if file_path.exists():
            file_path.unlink()
    except Exception as e:
        print(f"Warning: Could not delete file: {e}")
    
    # Delete from database (cascades to related tables)
    db.delete(document)
    db.commit()
    
    return None

@app.get(
    "/api/v1/documents/{document_id}/download",
    tags=["Documents"],
    summary="Download original document file"
)
def download_document(
    document_id: str,
    db: Session = Depends(get_db)
):
    """
    Download the original uploaded document file
    """
    document = db.query(Document).filter_by(id=document_id).first()
    
    if not document:
        raise HTTPException(404, f"Document with ID {document_id} not found")
    
    file_path = Path(document.file_path)
    
    if not file_path.exists():
        raise HTTPException(404, "File not found on disk")
    
    return FileResponse(
        path=str(file_path),
        filename=document.filename,
        media_type=document.mime_type or "application/octet-stream"
    )

@app.get(
    "/api/v1/stats",
    tags=["Statistics"],
    summary="Get system statistics"
)
def get_statistics(db: Session = Depends(get_db)):
    """
    Get overall system statistics
    
    Returns:
    - Total documents
    - Documents by status
    - Documents by type
    - Average processing time
    """
    from sqlalchemy import func
    
    total_documents = db.query(func.count(Document.id)).scalar()
    
    # Count by status
    status_counts = db.query(
        Document.status,
        func.count(Document.id)
    ).group_by(Document.status).all()
    
    # Count by type
    type_counts = db.query(
        Document.document_type,
        func.count(Document.id)
    ).group_by(Document.document_type).all()
    
    # Average processing time
    avg_processing_time = db.query(
        func.avg(
            func.extract('epoch', Document.processed_at - Document.uploaded_at) * 1000
        )
    ).filter(
        Document.processed_at.isnot(None)
    ).scalar()
    
    return {
        "total_documents": total_documents,
        "by_status": {status: count for status, count in status_counts},
        "by_type": {doc_type: count for doc_type, count in type_counts},
        "average_processing_time_ms": int(avg_processing_time) if avg_processing_time else None,
        "generated_at": datetime.utcnow().isoformat()
    }

@app.get(
    "/api/v1/documents/{document_id}/logs",
    tags=["Documents"],
    summary="Get processing logs for a document"
)
def get_document_logs(
    document_id: str,
    db: Session = Depends(get_db)
):
    """
    Get detailed processing logs for debugging
    """
    document = db.query(Document).filter_by(id=document_id).first()
    
    if not document:
        raise HTTPException(404, f"Document with ID {document_id} not found")
    
    logs = db.query(ProcessingLog).filter_by(document_id=document_id).order_by(
        ProcessingLog.created_at.asc()
    ).all()
    
    return {
        "document_id": document_id,
        "filename": document.filename,
        "logs": [
            {
                "stage": log.stage,
                "status": log.status,
                "duration_ms": log.duration_ms,
                "details": log.details,
                "timestamp": log.created_at.isoformat()
            }
            for log in logs
        ]
    }

# ============ RUN SERVER ============

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )