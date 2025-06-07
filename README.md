#  BTÜ Chatbot – Üniversite Asistanı

BTÜ Chatbot, Bursa Teknik Üniversitesi öğrenci rehberinden bilgi alarak öğrencilere doğal dilde yanıt verebilen akıllı bir sohbet uygulamasıdır. Kullanıcı dostu arayüzü sayesinde, öğrencilerin sıkça sorduğu sorulara hızlı ve doğru yanıtlar sunar.

##  Özellikler

-  PDF tabanlı üniversite rehberinden bilgi çıkarımı (OCR destekli)
-  Google Gemini API ile doğal dilde yanıt üretimi
-  SentenceTransformer ile embedding modeli (MiniLM)
-  ChromaDB ile vektör tabanlı sorgulama
-  CORS destekli frontend erişimi

##  Kullanılan Teknolojiler

- Python 
- Google Generative AI (Gemini API)
- ChromaDB
- pdf2image & pytesseract (OCR)
- SentenceTransformer
- HTML/CSS frontend

## 🔧 Kurulum ve Çalıştırma

1. **Gerekli kütüphaneleri yükleyin:**

   ```bash
   pip install -r requirements.txt

2. **Vektörleri oluşturmak için OCR çalıştırın:**
   ```bash
   python vector_build_ocr.py
   
3. **Sunucuyu başlatın:**
   ```bash
   app.py   
