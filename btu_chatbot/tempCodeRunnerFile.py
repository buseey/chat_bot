import os
os.environ['TESSDATA_PREFIX'] = r"C:\Program Files\Tesseract-OCR\tessdata"

import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
from pdf2image import convert_from_path
from sentence_transformers import SentenceTransformer
import chromadb

# PDF'i resimlere çevir
pdf_path = "Öğrenci_Rehberi_BM_28.07.2022.pdf"
images = convert_from_path(pdf_path, poppler_path=r"C:\poppler\Library\bin")

print(f"{len(images)} sayfa bulundu. OCR başlıyor...")

texts = []
for i, img in enumerate(images):
    text = pytesseract.image_to_string(img, lang='tur')
    texts.append(text)
    print(f"{i+1}. sayfa tamamlandı.")

# Embedding modeli
model = SentenceTransformer("all-MiniLM-L6-v2")

# Chroma veritabanı (yeni sistemle)
client = chromadb.PersistentClient(path="./chroma")
collection = client.get_or_create_collection(name="btu_rehberi")

for i, metin in enumerate(texts):
    if metin.strip():
        collection.add(
            documents=[metin],
            metadatas=[{"kaynak": f"sayfa_{i+1}"}],
            ids=[f"doc_{i+1}"]
        )

print("✅ OCR ile vektör veritabanı başarıyla oluşturuldu.")
