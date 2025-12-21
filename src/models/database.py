from sqlalchemy import create_engine, Column, String, DateTime, Float, Integer, Boolean, Text, JSON, DECIMAL, ForeignKey, CheckConstraint
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.dialects.postgresql import UUID, JSONB
from datetime import datetime
import uuid
import os
from dotenv import load_dotenv

load_dotenv()

Base = declarative_base()

# ============ MODELS ============

class Document(Base):
    __tablename__ = "documents"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filename = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False, unique=True)
    file_size_bytes = Column(Integer)
    mime_type = Column(String(100))
    document_type = Column(String(50))
    status = Column(String(50), default="pending")
    error_message = Column(Text)
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    processed_at = Column(DateTime)
    created_by = Column(String(100))
    document_metadata = Column(JSONB)
    
    # Relationships
    extracted_data = relationship("ExtractedData", back_populates="document", cascade="all, delete-orphan")
    invoice = relationship("Invoice", back_populates="document", uselist=False, cascade="all, delete-orphan")
    validations = relationship("ValidationResult", back_populates="document", cascade="all, delete-orphan")
    logs = relationship("ProcessingLog", back_populates="document", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Document {self.filename} ({self.status})>"


class ExtractedData(Base):
    __tablename__ = "extracted_data"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    field_name = Column(String(100), nullable=False)
    field_value = Column(Text)
    normalized_value = Column(Text)
    confidence_score = Column(Float)
    extraction_method = Column(String(50))
    bounding_box = Column(JSONB)
    page_number = Column(Integer)
    extracted_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship
    document = relationship("Document", back_populates="extracted_data")
    
    __table_args__ = (
        CheckConstraint('confidence_score >= 0 AND confidence_score <= 1', name='valid_confidence'),
    )


class Invoice(Base):
    __tablename__ = "invoices"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), unique=True, nullable=False)
    
    # Vendor
    vendor_name = Column(String(255))
    vendor_address = Column(Text)
    vendor_tax_id = Column(String(50))
    vendor_email = Column(String(255))
    vendor_phone = Column(String(50))
    
    # Invoice details
    invoice_number = Column(String(100), unique=True)
    invoice_date = Column(DateTime)
    due_date = Column(DateTime)
    po_number = Column(String(100))
    
    # Amounts
    subtotal = Column(DECIMAL(12, 2))
    tax_amount = Column(DECIMAL(12, 2))
    total_amount = Column(DECIMAL(12, 2))
    currency = Column(String(10), default="USD")
    
    # Line items
    line_items = Column(JSONB)
    
    # Quality
    overall_confidence = Column(Float)
    validation_status = Column(String(50))
    validation_errors = Column(JSONB)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationship
    document = relationship("Document", back_populates="invoice")


class ValidationResult(Base):
    __tablename__ = "validation_results"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    rule_name = Column(String(100), nullable=False)
    passed = Column(Boolean, nullable=False)
    message = Column(Text)
    severity = Column(String(20))
    field_name = Column(String(100))
    expected_value = Column(Text)
    actual_value = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship
    document = relationship("Document", back_populates="validations")


class ProcessingLog(Base):
    __tablename__ = "processing_logs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"))
    stage = Column(String(50), nullable=False)
    status = Column(String(20), nullable=False)
    duration_ms = Column(Integer)
    details = Column(JSONB)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship
    document = relationship("Document", back_populates="logs")


# ============ DATABASE CONNECTION ============

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://localhost/idp_db")

engine = create_engine(
    DATABASE_URL,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,  # Verify connections before using
    echo=False  # Set to True for SQL query logging
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# ============ HELPER FUNCTIONS ============

def init_db():
    """Create all tables"""
    Base.metadata.create_all(bind=engine)
    print("✅ Database initialized successfully")

def get_db():
    """Dependency for FastAPI routes"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ============ USAGE EXAMPLE ============
if __name__ == "__main__":
    # Initialize database
    init_db()
    
    # Create a test document
    db = SessionLocal()
    try:
        doc = Document(
            filename="test_invoice.pdf",
            file_path="/uploads/test.pdf",
            document_type="invoice",
            status="pending",
            file_size_bytes=204800,
            mime_type="application/pdf",
            created_by="admin",
            document_metadata={"source": "email", "tags": ["invoice", "priority"]}
        )
        db.add(doc)
        db.commit()
        print(f"✅ Created document: {doc.id}")
        
        # Query it back
        retrieved = db.query(Document).filter_by(filename="test_invoice.pdf").first()
        print(f"✅ Retrieved: {retrieved}")
        
    finally:
        db.close()