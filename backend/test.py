from services.vector_store import VectorStoreManager
import gc

print("🧠 Testing RAG (low memory mode)...")
manager = VectorStoreManager()

try:
    results = manager.similarity_search("How do I apply for OPT on F1 visa?")

    print("\n✅ SUCCESS!")
finally:
    gc.collect()  # Force cleanup
    print("🧹 Memory cleaned.")
