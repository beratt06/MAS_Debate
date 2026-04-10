# Mevzuu AI Decision Debate System

PDF tabanli bilgi toplama (retrieval), cok ajanli tartisma, kanit dogrulama ve nihai karar verme adimlarini tek bir akista birlestiren bir AI sistemidir.

## Neler Yapar

- Soru geldiginde router ve research adimi ile konu ozeti + bulgu listesi uretir.
- Pro ve Contra ajanlari ile cok turlu tartisma yurutur.
- Tartismadaki iddialari Evidence Verifier ve Debate Scoring adimlarindan gecirir.
- Judge Agent ile nihai oneriyi ve guven skorunu uretir.
- Hem Streamlit UI hem de CLI uzerinden calisir.

## Pipeline Akisi

1. Soru alinir.
2. Retrieval katmani PDF kaynaklardan baglam toplar.
3. Research agent konu ozetini ve temel bulgulari cikarir.
4. Debate loop (Pro <-> Contra) birden fazla tur calistirir.
5. Verifier + scoring adimlari secili iddialari degerlendirir.
6. Judge son karari ve guven skorunu verir.

## Mimari

- `multiagent_system.py`: Tum pipeline orkestrasyonu
- `app.py`: Streamlit arayuzu
- `main.py`: CLI girisi
- `agents/research_agent.py`: Arastirma adimi
- `router/router.py`: Soru yonlendirme ve state baslangici
- `retrieval/`: PDF yukleme, chunk, embedding, vector store
- `pro_agent.py` ve `contra_agent.py`: Tartisma ajanlari
- `debate_loop.py`: Tur bazli tartisma hafizasi ve transcript
- `last_part.py`: Evidence verifier, scoring, judge cagrilari
- `config.py`: Merkez konfig (model, top-k, dizinler)

## Gereksinimler

- Python 3.10+
- Ollama (lokalde kurulu ve calisiyor)
- Uygun bir model (varsayilan: `gpt-oss:120b-cloud`)

## Hizli Baslangic

1. Proje klasorune girin.
2. Sanal ortam olusturun.
3. Bagimliliklari kurun.
4. Ollama servisini ve modeli hazirlayin.
5. PDF index olusturup sistemi calistirin.

### 1) Ortam Kurulumu

```bash
python -m venv .venv
```

Windows PowerShell:

```bash
.\.venv\Scripts\Activate.ps1
```

Bagimliliklar:

```bash
pip install -r requirements.txt
```

### 2) Ollama Hazirligi

Ornek model indir:

```bash
ollama pull llama3
```

Servisi baslat:

```bash
ollama serve
```

Not: Varsayilan model adini `config.py` icindeki `OLLAMA_MODEL_NAME` ile degistirebilirsiniz.

### 3) PDF Kaynaklarini Hazirlama

- PDF dosyalarini `data/pdfs` altina koyun.
- Ilk calistirmada veya veri degistiginde index yenileyin (`--build-index`).

## Calistirma

### Streamlit UI

```bash
streamlit run app.py
```

### CLI

```bash
python main.py "Devletler yapay zeka gelisimini regule etmeli mi?" --build-index
```

### CLI Parametreleri

- `--model`: Kullanilacak Ollama modeli
- `--rounds`: Tartisma tur sayisi (varsayilan: 3)
- `--top-k`: Retrieval top-k chunk sayisi
- `--output`: Cikti JSON yolu (varsayilan: `multiagent_result.json`)
- `--build-index`: Calistirmadan once index olustur/guncelle

## Cikti

Sistem sonunda JSON formatinda su ana bolumleri uretir:

- `question`
- `research` (topic summary, facts, sources, retrieved context)
- `debate` (rounds + full history)
- `verifier_logs`
- `judge`

## Sorun Giderme

- Ollama baglanti hatasi:
	- `ollama serve` calisiyor mu kontrol edin.
	- Model adinin sistemde mevcut oldugunu dogrulayin.
- Retrieval sonucu zayif:
	- `data/pdfs` altina daha ilgili kaynak ekleyin.
	- `--top-k` degerini arttirin.
- Yavas yanit:
	- Daha hafif model deneyin.
	- Round sayisini dusurun.

## Gelistirme Notlari

- Kod tabani moduler oldugu icin yeni ajanlar eklemek kolaydir.
- `MultiAgentDebateSystem` sinifi hem UI hem CLI tarafinda tek giris noktasi olarak kullanilir.

## Lisans

Depodaki lisans dosyasina gore kullanin.
