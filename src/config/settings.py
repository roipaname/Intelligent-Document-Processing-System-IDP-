from pydantic_settings import BaseSettings
from typing import Optional
from pathlib import Path


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # API Configuration
    api_title: str = "Intelligent Document Processing API"
    api_version: str = "1.0.0"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_reload: bool = True
    
    # Database Configuration
    database_url: str = "postgresql://myuser:mypassword@localhost:5432/idp_db"
    database_pool_size: int = 10
    database_max_overflow: int = 20
    
    # File Storage
    upload_dir: Path = Path("./data/uploads")
    processed_dir: Path = Path("./data/processed")
    model_dir: Path = Path("./data/models")
    
    # File Upload Limits
    max_file_size_mb: int = 10
    allowed_extensions: list = [".pdf", ".png", ".jpg", ".jpeg", ".tiff"]
    
    # OCR Configuration
    tesseract_cmd: Optional[str] = None  # Auto-detect if None
    ocr_language: str = "eng"
    ocr_dpi: int = 300
    ocr_psm: int = 6  # Page segmentation mode
    
    # Poppler Configuration (for PDF to image)
    poppler_path: Optional[str] = None
    
    # Processing Configuration
    confidence_threshold: float = 0.75
    min_text_chars: int = 50  # For PDF text detection
    
    # Logging
    log_level: str = "INFO"
    log_file: Path = Path("./logs/idp.log")
    
    # Security
    secret_key: str = "your-secret-key-change-in-production"
    
    # Feature Flags
    enable_async_processing: bool = True
    enable_table_extraction: bool = True
    enable_entity_extraction: bool = True
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()


# Create required directories
def init_directories():
    """Create all required directories"""
    directories = [
        settings.upload_dir,
        settings.processed_dir,
        settings.model_dir,
        settings.log_file.parent
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    
    print("✅ Initialized directories")


if __name__ == "__main__":
    init_directories()
    print("\nCurrent Settings:")
    print(f"  Database: {settings.database_url}")
    print(f"  Upload Dir: {settings.upload_dir}")
    print(f"  OCR DPI: {settings.ocr_dpi}")