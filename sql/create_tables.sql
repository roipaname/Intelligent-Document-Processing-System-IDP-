CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE TABLE IF NOT EXISTS documents(
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    filename VARCHAR NOT NULL,
    file_path VARCHAR NOT NULL UNIQUE,
    file_size_bytes INTEGER NOT NULL,
    mime_type VARCHAR NOT NULL,
    document_type VARCHAR CHECK (document_type IN ('invoice','contract','receipt','other','default')) DEFAULT 'default',
    status VARCHAR CHECK( status IN ('pending','processing','completed','failed')) DEFAULT 'pending',
    error_message TEXT,
    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    processed_at TIMESTAMP,
    created_by VARCHAR,
    document_metadata JSONB,

    CONSTRAINT  valid_dates CHECK (processed_at is NULL OR processed_at>=uploaded_at)
);

CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status);
CREATE INDEX IF NOT EXISTS idx_documents_type ON documents(document_type);
CREATE INDEX IF NOT EXISTS idx_documents_uploaded ON documents(uploaded_at DESC);
CREATE INDEX IF NOT EXISTS idx_documents_metadata ON documents USING gin(document_metadata);



-- ============ EXTRACTED DATA TABLE ============

CREATE TABLE IF NOT EXISTS extracted_data(
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    field_name VARCHAR Not NUll,
    field_value TEXT,
    normalized_value TEXT,
    confidence_score FLOAT CHECK(confidence_score>=0 AND confidence_score<=1),
    bounding_box JSONB,
    page_number INTEGER,
    extracted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,   
    CONSTRAINT unique_field_per_document UNIQUE(document_id, field_name )
);

CREATE INDEX IF NOT EXISTS idx_extracted_doc_ID ON extracted_data(document_id);
CREATE INDEX IF NOT EXISTS idx_extracted_confidence ON extracted_data(confidence_score DESC);
CREATE INDEX IF NOT EXISTS idx_extracted_field ON extracted_data(field_name);

-- ============ INVOICES TABLE (Structured Output) ============
CREATE TABLE IF NOT EXISTS invoices(
    ID UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,

     -- Vendor Information
    vendor_name VARCHAR,
    vendor_address TEXT,
    vendor_tax_id VARCHAR,
    vendor_email VARCHAR,
    vendor_phone VARCHAR(50),

    --Invoice Details
    invoice_number VARCHAR(100) UNIQUE NOT NULL,
    invoice_date DATE,
    due_date DATE,
    po_number VARCHAR(100),

    --Amounts
    subtotal DECIMAL(12,2),
    tax_amount DECIMAL(12,2),
    total_amount DECIMAL(12,2),
    currency VARCHAR(10) DEFAULT 'ZAR',

    --LINE ITEMS AS JSONB
    line_items JSONB,

    -- Quality Metrics
    overall_confidence FLOAT,
    validation_status VARCHAR(50) CHECK (validation_status IN ('valid', 'invalid', 'needs_review')),
    validation_errors JSONB,  -- Array of error messages
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT valid_amounts CHECK (total_amount >= 0 AND tax_amount >= 0),
    CONSTRAINT valid_invoice_dates CHECK (due_date IS NULL OR due_date >= invoice_date)

);


CREATE INDEX IF NOT EXISTS idx_invoices_number ON invoices(invoice_number);
CREATE INDEX IF NOT EXISTS idx_invoices_date ON invoices(invoice_date DESC);
CREATE INDEX IF NOT EXISTS idx_invoices_vendor ON invoices(vendor_name);
CREATE INDEX IF NOT EXISTS idx_invoices_amount ON invoices(total_amount);

-- ============ VALIDATION RESULTS TABLE ============
CREATE TABLE IF NOT EXISTS validation_results (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    rule_name VARCHAR(100) NOT NULL,
    passed BOOLEAN NOT NULL,
    message TEXT,
    severity VARCHAR(20) CHECK (severity IN ('error', 'warning', 'info')),
    field_name VARCHAR(100),  -- Which field failed validation
    expected_value TEXT,
    actual_value TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_validation_doc ON validation_results(document_id);
CREATE INDEX IF NOT EXISTS idx_validation_passed ON validation_results(passed);

-- ============ PROCESSING AUDIT LOG ============
CREATE TABLE IF NOT EXISTS processing_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    stage VARCHAR(50) NOT NULL,  -- 'ocr', 'extraction', 'validation'
    status VARCHAR(20) NOT NULL,  -- 'started', 'completed', 'failed'
    duration_ms INTEGER,
    details JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_logs_document ON processing_logs(document_id);
CREATE INDEX IF NOT EXISTS idx_logs_created ON processing_logs(created_at DESC);