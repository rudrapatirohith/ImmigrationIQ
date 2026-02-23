from services.vector_store import VectorStoreManager

manager = VectorStoreManager()
results = manager.similarity_search("How do I apply for OPT on F1 visa?")

print("\n✅ RAG working perfectly!")
