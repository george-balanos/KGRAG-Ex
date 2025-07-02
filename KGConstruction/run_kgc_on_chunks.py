from pymongo import MongoClient
from pymongo.server_api import ServerApi
from config import *
from KnowledgeCollector import KnowledgeCollector

def fetch_all_chunks():
    client = MongoClient(MONGO_URI, server_api=ServerApi("1"))
    db = client[DB_CLIENT]
    collection = db[STATPEARLS_COLLECTION]

    print(DB_CLIENT)
    print(STATPEARLS_COLLECTION)

    documents = list(collection.find({"chunk_text": {"$exists": True, "$ne": ""}}))

    client.close()
    return [(doc["chunk_text"], doc.get("source_filename", "unknown"), str(doc["_id"])) for doc in documents]

def main():
    kc = KnowledgeCollector()
    chunks = fetch_all_chunks() 

    for idx, (chunk_text, source_filename, chunk_id) in enumerate(chunks):
        print(f"\n--- Processing Chunk {idx + 1} ---")
        extracted = kc.extract_data(chunk_text, source_filename, chunk_id)
        if extracted:
            print("✅ Extracted Data:", extracted)
        else:
            print("⚠️ No data extracted.")

if __name__ == "__main__":
    main()