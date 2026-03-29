# Mevzuu AI Decision Debate System

Bu proje, PDF tabanli retrieval + cok ajanli tartisma + karar verme akisini tek bir sistemde birlestirir.

## Ozellikler

- Router + Research Agent
- Pro Agent / Contra Agent ile cok turlu tartisma
- Evidence Verifier + Debate Scoring
- Judge Agent ile nihai karar ve guven skoru
- Streamlit arayuzu

## Proje Yapisi

- app.py: Streamlit arayuzu
- main.py: CLI calistirma girisi
- multiagent_system.py: Tum ajanlari birlestiren orkestrator
- agents/: Research agent
- retrieval/: PDF yukleme, chunk, embedding, vector store
- router/: Soru yonlendirme katmani

## Kurulum

1. Sanal ortam olustur ve etkinlestir.
2. Bagimliliklari kur:

```bash
pip install -r requirements.txt
```

## Calistirma

### 1) Arayuz

```bash
streamlit run app.py
```

### 2) CLI

```bash
python main.py "Devletler yapay zeka gelisimini regule etmeli mi?" --build-index
```

Opsiyonel parametreler:

- --model: Kullanilacak Ollama modeli
- --rounds: Tartisma tur sayisi
- --top-k: Retrieval top-k
- --output: Cikti JSON dosyasi

## Notlar

- Ollama servisinin calisiyor olmasi gerekir.
- PDF dosyalarini data/pdfs klasorune koyduktan sonra --build-index ile index guncellenmelidir.
- JSON alan adlari sistemle uyum icin Ingilizce tutulur; metin icerigi Turkce uretilir.
