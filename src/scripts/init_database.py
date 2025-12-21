import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.database import init_db,SessionLocal,Document,Invoice,ExtractedData,ValidationResult,ProcessingLog
from sqlalchemy import text

def reset_database():
    """
    drop and recreate all tables (Caution: This will delete all data   )
    """
    from src.models.database import Base,engine
    print("⚠️ Dropping all tables...")
    Base.metadata.drop_all(bind=engine)
    print("✅ All tables dropped.")
    print("✅ Creating fresh tables...")
    Base.metadata.create_all(bind=engine)
    print("✅ Database reset successfully.")
def seed_sample_data():
    """
    Seed the database with sample data for testing
    """
    db=SessionLocal()
    try:
        doc=Document(
            filename="sample_invoice2.pdf",
            file_path="./data/uploads/sample_invoice2.pdf",
            document_type="invoice",
            status="completed",
            file_size_bytes=150000,
            mime_type="application/pdf",
            created_by="admin",
            document_metadata={"source":"scanner","tags":["sample","invoice"]}
        )
        db.add(doc)
        db.commit()
        db.refresh(doc)
        print(f"✅ Sample document created: {doc.id}")

        invoice=Invoice(
            document_id=doc.id,
            vendor_name="Sample_Vendor",
            invoice_number="INV-0012",
            invoice_date="2024-02-15",
            total_amount=750.00,
            currency="ZAR",
            tax_amount=112.50,
            line_items=[
                {"description":"Item A","quantity":3,"unit_price":100.00,"total_price":300.00},
                {"description":"Item B","quantity":5,"unit_price":90.00,"total_price":450.00}
            ],
            overall_confidence=0.96,
            validation_status="valid")
        db.add(invoice)
        db.commit()
        db.refresh(invoice)
        print(f"✅ Sample invoice created for document: {invoice.document_id} with id {invoice.id}")
    except Exception as e:
        db.rollback()
        print(f"❌ Error seeding sample data: {e}")
    finally:
        db.close()

if __name__=="__main__":
    import argparse
    parser= argparse.ArgumentParser(description="Initialize or reset the database ")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Drop and recreate all tables (Caution: This will delete all data)"
    )
    parser.add_argument(
        "--seed",
        action="store_true",
        help="Seed the database with sample data for testing"
    )
    args=parser.parse_args()
    if args.reset:
        confirm= input("Are you sure you want to reset the database? This will delete all data. (yes/no): ")
        if confirm.lower()=="yes" or confirm.lower()=="y":
            reset_database()
        else:
            print("Database reset cancelled.")
    else:
        seed_sample_data()
    if not args.reset and not args.seed:
        init_db()