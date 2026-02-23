"""
Run: python -m data.ingest
This takes ~5-10 minutes on first run.
Creates a chroma_db/ directory with all USCIS documents indexed.
"""
from data.download_uscis import download_forms
from services.document_processor import USCISDocumentProcessor
from services.vector_store import VectorStoreManager

def main():
    # Step 1: Download PDFs if not already downloaded
    print("=== STEP 1: Downloading USCIS Form Instructions ===")
    download_forms()
    
    # Step 2: Process PDFs into chunks
    print("\n=== STEP 2: Processing PDFs into Chunks ===")
    processor = USCISDocumentProcessor()
    all_chunks = processor.process_all_forms()
    
    # Step 3: Build vector index
    print("\n=== STEP 3: Building Vector Index ===")
    manager = VectorStoreManager()
    manager.build_index(all_chunks)
    
    print("\n✅ Ingestion complete! Vector index is ready.")
    print("Run: uvicorn main:app --reload")

if __name__ == "__main__":
    main()