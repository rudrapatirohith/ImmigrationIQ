from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from pathlib import Path
import re

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

        chunks = []
        for page in pages:
            # Clean the text (PDFs often have weird formatting)
            cleanned_text = self._clean_uscis_text(page.page_content)

            if len(cleanned_text.strip()) < 50: # Skip very short pages (e.g. cover page)
                continue

            page_chunks =self.text_splitter.create_documents(
                texts=[cleanned_text],
                metadatas=[{
                    "form_number": form_number,
                    "page": page.metadata.get("page",0)+1, # Human-readable page number
                    "source": f"USCIS Form {form_number} Instructions, Page {page.metadata.get('page',0)+1}",
                    "pdf_path": pdf_path,
                }]
            )
            chunks.extend(page_chunks)

        print(f" {form_number}: {len(pages)} pages -> {len(chunks)} chunks")
        return chunks
    
    def _clean_uscis_text(self, text: str) -> str:
        """Clean up PDF extraction artifacts"""
        # Remove excessive whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        # Remove page numbers and headers that got extracted
        text = re.sub(r'Form [A-Z]-\d+ Instructions \(\d{2}/\d{2}/\d{2}\)', '', text)
        return text.strip()
    
    def process_all_forms(self, pdf_dir: str = "data/uscis_pdfs") -> list[Document]:
        """Process all downloaded PDFs"""
        all_chunks = []
        pdf_files = list(Path(pdf_dir).glob("*.pdf")) # Get all PDF files in the directory

        print(f"Processing {len(pdf_files)} USCIS instruction PDFs from {pdf_dir}...")
        for pdf_file in pdf_files:
            form_number = pdf_file.stem.split("_")[0] # Extract form number from filename, e.g. "I-485_instructions.pdf" -> "I-485"
            chunks = self.load_and_chunk(str(pdf_file), form_number)
            all_chunks.extend(chunks) # Add chunks to the list

        print(f"Total chunks created: {len(all_chunks)}")
        return all_chunks