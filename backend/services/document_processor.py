from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from pathlib import Path
import re
import gc  # Garbage collector

class USCISDocumentProcessor:

    """
    Processes USCIS form instructions into chunks suitable for RAG.
    
    Why RecursiveCharacterTextSplitter?
    It tries to split on paragraph boundaries first, then sentences,
    then words. This keeps semantically related content together.
    
    Why 800 tokens with 150 overlap?
    - 800: enough context to answer a specific question
    - 150 overlap: prevents cutting off a sentence at chunk boundary

    """


    def __init__(self): # Initialize the text splitter with the desired chunk size and overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def load_and_chunk(self, pdf_path: str, form_number: str) -> list[Document]: # Note: returns a list of langchain_core Document objects, each with .page_content and .metadata
        """
        Load a PDF and split into chunks with rich metadata.
        The metadata is CRITICAL — it's how you tell users which
        document and page their answer came from.
        """
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        del loader  # Free memory immediately
        gc.collect()  # Force garbage collection

        chunks = []
        for page in pages:
            cleaned_text = self._clean_uscis_text(page.page_content)
            if len(cleaned_text.strip()) < 50:
                continue
            
            page_chunks = self.text_splitter.create_documents(
                texts=[cleaned_text],
                metadatas=[{
                    "form_number": form_number,
                    "page": page.metadata.get("page", 0) + 1,
                    "source": f"USCIS Form {form_number} Instructions, Page {page.metadata.get('page', 0) + 1}",
                    "pdf_path": pdf_path,
                }]
            )
            chunks.extend(page_chunks)
            del page_chunks, cleaned_text  # Free page memory
        
        del pages  # Free all pages
        gc.collect()
        print(f"  {form_number}: {len(chunks)} chunks")
        return chunks
    
    def _clean_uscis_text(self, text: str) -> str:
        """Clean up PDF extraction artifacts"""
        # Remove excessive whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'Form [A-Z]-\d+ Instructions \(\d{2}/\d{2}/\d{2}\)', '', text)
        return text.strip()
    
    def process_all_forms(self, pdf_dir: str = "data/uscis_pdfs") -> list[Document]:
        """Process all downloaded PDFs"""
        all_chunks = []
        pdf_files = list(Path(pdf_dir).glob("*.pdf"))
        
        print(f"Processing {len(pdf_files)} PDFs...")
        for pdf_file in pdf_files:
            form_number = pdf_file.stem.split("_")[0]
            chunks = self.load_and_chunk(str(pdf_file), form_number)
            all_chunks.extend(chunks)
            gc.collect()  # Free memory after each PDF
        
        print(f"Total: {len(all_chunks)} chunks")
        return all_chunks
