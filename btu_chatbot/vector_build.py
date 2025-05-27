# vector_build.py
import nltk
nltk.download('punkt')
import fitz
from sentence_transformers import SentenceTransformer
import chromadb




from nltk.tokenize import sent_tokenize


# PDF'ten metin al
doc = fitz.open("Öğrenci_Rehberi_BM_28.07.2022.pdf")
metinler = []
for sayfa in doc:
    metinler.append(sayfa.get_text())

parcali_metinler = []
for metin in metinler:
    parcali_metinler.extend(sent_tokenize(metin))

# Embedding modeli
model = SentenceTransformer("all-MiniLM-L6-v2")

# Chroma DB
client = chromadb.PersistentClient(path="db") 
#client = chromadb.PersistentClient(path="./chroma")
collection = client.get_or_create_collection(name="btu_rehberi")

for i, metin in enumerate(parcali_metinler):
    collection.add(
        documents=[metin],
        metadatas=[{"kaynak": "ogrenci_rehberi"}],
        ids=[f"doc_{i}"]
    )

print("Vektör veritabanı oluşturuldu.")
