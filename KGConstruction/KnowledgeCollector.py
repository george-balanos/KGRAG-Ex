import csv
from templates import EXTRACT_ENTITIES_TEMPLATE
from utils import run_model
import os

class KnowledgeCollector:
    def __init__(self):
        self.data = []

    def extract_data(self, text, source_filename, chunk_id):
        result_message, run_time = run_model("llama3", f"{EXTRACT_ENTITIES_TEMPLATE}\n{text}")
        
        if not result_message:
            return

        lines = result_message.split('\n')
        for line in lines:
            if "Answer:" in line:
                parts = line.split(":")
                
                if len(parts) == 6:
                    entity1, label1, relationship, entity2, label2 = parts[1:]
                    self.data.append([entity1.strip(), label1.strip(), relationship.strip(), entity2.strip(), label2.strip(), source_filename, chunk_id])

        print(self.data) 

        self.write_to_csv(self.data)

        extracted_data = self.data 
        self.data = []  

        return extracted_data

    def write_to_csv(self, data):
        os.makedirs("Graph", exist_ok=True)

        file_exists = os.path.exists("Graph/arch_kg.csv")

        with open("Graph/arch_kg.csv", "a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(["Entity1", "Label1", "Relationship", "Entity2", "Label2", "SourceFilename", "ChunkID"])  # ✅ header
            writer.writerows(data)


