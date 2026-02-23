from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
import atexit

class VectorStoreManager:
    """
    Manages the ChromaDB vector store for USCIS documents.
    
    Embeddings: sentence-transformers/all-MiniLM-L6-v2
    - 384 dimensions
    - Runs 100% locally, no API calls
    - ~80MB model, downloads once
    - Fast enough for real-time retrieval
    """
    
    # This downloads the model on first run (~80MB), then caches it
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.persist_directory = persist_directory
        self.embeddings = None  # Lazy load
        self._vectorstore = None
        atexit.register(self.cleanup)  # Auto-cleanup
    
    def _get_embeddings(self):
        if self.embeddings is None:
            print("Loading embeddings...")
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
        return self.embeddings
    
    def _get_vectorstore(self):
        if self._vectorstore is None:
            self._vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self._get_embeddings(),
                collection_name="uscis_documents"
            )
        return self._vectorstore
    
    def build_index(self, documents: list[Document]) -> Chroma:
        """
        Build vector index from documents. 
        This is the 'indexing phase' of RAG.
        Run once, takes a few minutes.
        """
         
        print(f"Building vector index from {len(documents)} chunks...")
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self._get_embeddings(),
            persist_directory=self.persist_directory,
            collection_name="uscis_documents"
        )
        self._vectorstore = vectorstore
        print(f"Index built. Saved to {self.persist_directory}")
        return vectorstore
    
    def load_index(self) -> Chroma:
        return self._get_vectorstore()
    
    def get_retriever(self, k: int = 4, form_filter: str = None):
        """
        Get a retriever configured for ImmigrationIQ queries.
        
        k=4: Retrieve 4 most relevant chunks per query.
        Why 4? Enough context without overloading the context window
        
        form_filter: Optional — only search within a specific form's docs (e.g. "I-485") for more targeted retrieval.
        """
        vectorstore = self._get_vectorstore()
        search_kwargs = {"k": k}
        if form_filter:
            # Add a filter to only retrieve chunks from the specified form
            search_kwargs["filter"] = {"form_number": form_filter}
        
        return vectorstore.as_retriever(
            search_type="mmr",  # Diverse results
            search_kwargs=search_kwargs
        )
    
    def similarity_search(self, query: str, k: int = 4, form_filter: str = None) -> list[Document]:
        """Direct search + full text (no truncation)."""
        retriever = self.get_retriever(k=k, form_filter=form_filter)
        results = retriever.invoke(query)
        print(f"\n🔍 {len(results)} chunks:")
        for i, doc in enumerate(results, 1):
            print(f"\n{i}. {doc.metadata.get('source', 'Unknown')}")
            print(f"{doc.page_content[:300]}...")
        return results
    
    def cleanup(self):
        """Clean up embeddings on exit"""
        self.embeddings = None
        self._vectorstore = None
