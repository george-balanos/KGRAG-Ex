import os
import json
import hashlib
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from config import *
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import torch

class ChunkUploader:
    def __init__(self, directory, db_name=DB_CLIENT, collection_name=STATPEARLS_COLLECTION, chunk_size=600, overlap=100):
        """
        Initializes the ChunkUploader class with the given parameters.
        """

        print(DB_CLIENT)
        print(STATPEARLS_COLLECTION)

        self.directory = directory
        self.db_name = db_name
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.overlap = overlap

        try:
            self.client = MongoClient(MONGO_URI, server_api=ServerApi("1"))
            print("✅ MongoDB connection successful!")
        except Exception as e:
            print(f"❌ MongoDB connection failed: {e}")
            raise

        self.db = self.client[self.db_name]
        self.collection = self.db[self.collection_name]

        self.embedding_model = SentenceTransformer(SIMILARITY_MODEL)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding_model.to(self.device)

        self.ensure_text_index()

    def ensure_text_index(self):
        """Ensures a text index exists on the 'chunk_text' field for full-text search."""
        existing_indexes = self.collection.index_information()
        if "chunk_text_text" not in existing_indexes:
            self.collection.create_index([("chunk_text", "text")])
            print("✅ Created text index on 'chunk_text' field.")
        else:
            print("ℹ️ Text index already exists.")

    def split_text_with_recursive_splitter(self, text):
        """
        Splits text into chunks using RecursiveCharacterTextSplitter.
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.overlap,
            separators=["\n\n", "\n", " ", ""],
            length_function=len
        )
        return text_splitter.split_text(text)

    def get_chunk_hash(self, text):
        """
        Generates a SHA256 hash of the given text.
        """
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def upload_chunks(self):
        """
        Uploads JSONL chunks from a directory into MongoDB,
        generating embeddings for each chunk.
        """
        for filename in os.listdir(self.directory):
            if filename.endswith(".jsonl"):
                file_path = os.path.join(self.directory, filename)
                print(f"📂 Processing {file_path}...")

                with open(file_path, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            doc = json.loads(line)
                            # Correct key name
                            text_parts = [doc.get(k, "") for k in ("title", "abstract", "body")]
                            text = "\n\n".join([part for part in text_parts if part.strip()])

                            print(text)

                            if not text:
                                print(f"⚠️ Skipping document with no content: {doc.get('_id', 'No ID')}")
                                continue

                            chunks = self.split_text_with_recursive_splitter(text)
                            chunk_documents = []

                            for idx, chunk in enumerate(chunks):
                                chunk_id = self.get_chunk_hash(chunk)

                                embedding = self.embedding_model.encode(chunk, convert_to_tensor=False).tolist()

                                chunk_doc = {
                                    "_id": chunk_id,
                                    "chunk_index": idx,
                                    "chunk_text": chunk,
                                    "source_filename": filename,
                                    "chunk_embedding": embedding
                                }
                                chunk_documents.append(chunk_doc)

                            if chunk_documents:
                                try:
                                    self.collection.insert_many(chunk_documents, ordered=False)
                                    print(f"✅ Inserted {len(chunk_documents)} chunks.")
                                except Exception as insert_err:
                                    print(f"⚠️ Insert error (likely due to duplicate hashes): {insert_err}")

                        except json.JSONDecodeError as e:
                            print(f"⚠️ Skipping invalid JSON in {filename}: {e}")
                        except Exception as e:
                            print(f"❌ Unexpected error: {e}")

        print("✅ All files uploaded successfully.")


if __name__ == "__main__":
    chunks_dir = "statpearls_NBK430685/chunks"
    uploader = ChunkUploader(directory=chunks_dir, chunk_size=600, overlap=100)
    uploader.upload_chunks()
