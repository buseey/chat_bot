# app.py (Gemini entegre)
from flask import Flask, request, jsonify, send_from_directory
from sentence_transformers import SentenceTransformer
import chromadb
import google.generativeai as genai
from flask_cors import CORS

# Google Gemini API anahtarını burada tanımla
genai.configure(api_key="AIzaSyAfuVvfdywAsGu2lxyzCZAvt4PrO_Yv0fI")

# Flask uygulaması
app = Flask(__name__)
CORS(app)

# Embedding modeli
model = SentenceTransformer("all-MiniLM-L6-v2")

# Chroma veritabanı
client = chromadb.PersistentClient(path="./chroma")
collection = client.get_collection(name="btu_rehberi")

# HTML arayüzü (isteğe bağlı)
@app.route("/")
def index():
    return send_from_directory(".", "chatbot.html")

# Chat endpoint
@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    soru = data["question"]

    # Soru embed'ini oluştur
    soru_embed = model.encode([soru])

    # En alakalı metni Chroma'dan getir
    sonuc = collection.query(query_embeddings=soru_embed, n_results=1)
    ilgili_metin = sonuc['documents'][0][0]

    # Gemini ile cevap oluştur
    prompt = f"""
Aşağıdaki metni oku ve soruyu anlamlı ve kısa bir şekilde yanıtla. 

Metin:
\"\"\"
{ilgili_metin}
\"\"\"

Soru:
{soru}

Cevap:
"""
    try:
        # Gemini modelini güncel isme göre başlat
        gemini_model = genai.GenerativeModel("models/gemini-1.5-flash-latest")

        # prompt ile içerik üret
        response = gemini_model.generate_content(prompt)

        # Cevap metni
        return jsonify({"answer": response.text})

    except Exception as e:
        return jsonify({"answer": f"Hata oluştu: {str(e)}"})

# Uygulama başlatma
if __name__ == "__main__":
    app.run(debug=True)
