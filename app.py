from chunking import fixed_size_chunking_with_overlap
from extract_text_from_pdf import extract_text_from_pdf
from ingest_to_vector_db import ingest_chunks_without_metadata_to_qdrant

if __name__ == "__main__":
  pdf_path = "my_document.pdf"
  CHUNK_SIZE = 500
  OVERLAP = 50
 
  pdf_text = extract_text_from_pdf(pdf_path)
 
  chunks = fixed_size_chunking_with_overlap(
    pdf_text,
    chunk_size=CHUNK_SIZE,
    overlap=OVERLAP
    )
 
  all_chunks = [{"chunk": c} for c in chunks]
 
  ingest_chunks_without_metadata_to_qdrant(all_chunks)
 
  print(f"Done. Ingested {len(all_chunks)} chunks from {pdf_path} into Qdrant.")