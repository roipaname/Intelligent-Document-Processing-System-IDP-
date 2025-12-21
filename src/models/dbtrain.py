from sqlachemy import Column,create_engine,DateTime,String,Float,Boolean,Integer,Text,ForeignKey,JSON,DECIMAL,CheckConstraint
from sqlachemy.ext.declarative import declarative_base
from sqlachemy.orm import sessionmaker,relationship
from sqlachemy.dialects.postgresql import UUID,JSONB
from datetime import datetime
import uuid
import os
from dotenv import load_dotenv

load_dotenv()
Base=declarative_base()


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
    __tablename__="extracted_data"
    id=Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id=Column(UUID(as_uuid=True),ForeignKey("documents.id",ondelete="CASCADE"),nullable=False)

DATABASE_URL= os.getenv("DATABASE_URL")
engine=create_engine(DATABASE_URL,echo=True,pool_size=10,max_overflow=20,pool_pre_ping=True)
SessionLocal=sessionmaker(autocommit=False,autoflush=False,bind=engine)
Base.metadata.create_all(bind=engine)

db=SessionLocal()