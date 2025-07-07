#!/usr/bin/env bash

set -e

URL="https://ftp.ncbi.nlm.nih.gov/pub/litarch/3d/12/statpearls_NBK430685.tar.gz"
TARGET="statpearls_NBK430685.tar.gz"
DEST_DIR="statpearls_NBK430685"

if [ -f "$TARGET" ]; then
  echo "✔ $TARGET already exists. Skipping download."
else
  echo "⬇️  Downloading $URL ..."
  curl -L -o "$TARGET" "$URL"
  echo "✔ Download complete."
fi

mkdir -p "$DEST_DIR"

echo "📂 Extracting $TARGET into $DEST_DIR/ ..."
tar -xzf "$TARGET" -C "$DEST_DIR"
echo "✔ Extraction complete."

echo "📋 Contents of '$DEST_DIR':"
ls -l "$DEST_DIR"

echo "✅ Directory is set up!"

echo "🚀 Running nxml_to_jsonl.py ..."
python3 nxml_to_jsonl.py "$DEST_DIR"
echo "✅ Nxml to jsonl completed!"

echo "🚀 Running Chunker.py ..."
python3 Chunker.py
echo "✅ Chunking completed!"

echo "🚀 Running KnowledgeCollector on chunks ..."
python3 run_kgc_on_chunks.py
echo "✅ KGC completed!"

echo "🚀 Running KnowledgeGraphCreator ..."
python3 KnowledgeGraphCreator.py
echo "✅ KGC completed!"


