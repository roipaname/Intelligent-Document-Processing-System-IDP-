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
    proccessed_at TIMESTAMP,
    created_by VARCHAR,
    metadata JSONB,

    CONSTRAINT  valid_dates CHECK (proccessed_at is NULL OR proccessed_at>=uploaded_at)
);

CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status);
CREATE INDEX IF NOT EXISTS idx_documents_type ON documents(document_type);
CREATE INDEX IF NOT EXISTS idx_documents_uploaded ON documents(uploaded_at DESC);
CREATE INDEX IF NOT EXISTS idx_documents_metadata ON documents USING gin(metadata);



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