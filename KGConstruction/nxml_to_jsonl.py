import os
import json
from lxml import etree
from glob import glob

def extract_text(elem):
    if elem is None:
        return ""
    return ''.join(elem.itertext()).strip()

def parse_nxml(file_path):
    try:
        tree = etree.parse(file_path)
        root = tree.getroot()
        ns = {'ns': root.nsmap.get(None, '')}

        article = {
            "filename": os.path.basename(file_path)
        }

        title_elem = root.find('.//ns:article-title', namespaces=ns)
        abstract_elem = root.find('.//ns:abstract', namespaces=ns)
        body_elem = root.find('.//ns:body', namespaces=ns)

        article["title"] = extract_text(title_elem)
        article["abstract"] = extract_text(abstract_elem)
        article["body"] = extract_text(body_elem)

        return article
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return None

def convert_to_individual_jsonl(input_dir):
    output_dir = os.path.join(input_dir, "chunks")
    os.makedirs(output_dir, exist_ok=True)

    nxml_files = glob(os.path.join(input_dir, '*.nxml'))
    for file_path in nxml_files:
        article = parse_nxml(file_path)
        if article:
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            jsonl_path = os.path.join(output_dir, base_name + '.jsonl')
            with open(jsonl_path, 'w', encoding='utf-8') as out_file:
                out_file.write(json.dumps(article, ensure_ascii=False) + '\n')
            print(f"✅ Saved: {jsonl_path}")

if __name__ == "__main__":
    input_dir = 'statpearls_NBK430685'
    convert_to_individual_jsonl(input_dir)
