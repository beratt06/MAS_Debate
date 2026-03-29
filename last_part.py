"""
==========================================================
 AI Decision Debate System - Oğuz'un Görev Modülü
 Görevler: Evidence Verifier + Judge Agent + Debate Scoring
==========================================================
Bu dosya, projenin üçüncü kısmını içerir.
Sudem (Router + Research) ve Berat (Pro + Contra + Debate Loop)
modülleriyle birleştirilecektir.
"""

import ollama
import json


# ============================================================
# ORTAK FONKSİYON - ask_ollama
# Tüm ajanların kullandığı LLM istemcisi.
# ============================================================
def ask_ollama(system_prompt, user_prompt, model_name="gpt-oss:120b-cloud"):
    """
    Tüm ajanların kullanacağı ortak LLM istemcisidir.
    Ollama'ya prompt gönderir ve JSON yanıtını ayıklayıp Python Dict objesi olarak döndürür.
    """
    try:
        response = ollama.chat(
            model=model_name,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt}
            ]
        )
        content = response['message']['content'].strip()

        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].strip()

        return json.loads(content)
    except Exception as e:
        return {"Hata": "JSON parse edilemedi", "HamYanit": content[:50] if 'content' in locals() else str(e)}


# ============================================================
# 1) EVIDENCE VERIFIER
# Görev: Tartışmada kullanılan iddiaların güvenilirliğini
#        kontrol etmek ve her iddia için kanıt kaynağı ile
#        güven seviyesi belirlemek.
# ============================================================
def evidence_verifier(claim):
    """
    Tartışmadaki bir iddianın güvenilirliğini sorgular.

    Parametreler:
        claim (str): Kontrol edilecek iddia metni.

    Döndürür (dict):
        {
            "Claim":           "Kontrol edilen iddia",
            "Evidence_Source":  "İddianın dayandığı kaynak türü",
            "Confidence":      "High / Medium / Low"
        }
    """
    sys_prompt = """Sen bir Evidence Verifier'sın.
Görevin: Tartışma sırasında ortaya atılan iddiaların güvenilir olup olmadığını analiz etmek.
Her iddia için kaynağını belirle ve güven seviyesi ver.

KESİNLİKLE sadece JSON formatında dön:
{
    "Claim": "Değerlendirilen iddia",
    "Evidence_Source": "Kanıt/Kaynak Türü (örn: Bilimsel Araştırma, İstatistiksel Veri, Uzman Görüşü, Varsayım, Anekdot vs.)",
    "Confidence": "High veya Medium veya Low"
}

Güven seviyesi kriterleri:
- High: İddia bilimsel verilerle veya güvenilir istatistiklerle destekleniyorsa
- Medium: İddia mantıklı ama kesin veri ile desteklenmiyorsa  
- Low: İddia varsayıma veya kişisel görüşe dayanıyorsa"""

    return ask_ollama(sys_prompt, f"Kontrol Edilecek İddia: {claim}")


# ============================================================
# 2) DEBATE SCORING
# Görev: Argümanların teknik gücünü puanlamak.
#        Her argüman mantıksal tutarlılık, kanıt kalitesi ve
#        argüman gücü kriterlerine göre 1-10 arası puanlanır.
# ============================================================
def debate_scoring(argument_text):
    """
    Bir argümanın teknik gücünü üç farklı kritere göre puanlar.

    Parametreler:
        argument_text (str): Değerlendirilecek argüman metni.

    Döndürür (dict):
        {
            "mantiksal_tutarlilik": 8,   (1-10)
            "kanit_kalitesi":       7,   (1-10)
            "arguman_gucu":         8    (1-10)
        }
    """
    sys_prompt = """Sen bir Debate Scoring ajanısın.
Görevin: Tartışmadaki argümanların teknik gücünü objektif olarak puanlamak.

Her argümanı şu üç kritere göre 1'den 10'a kadar (10 en iyi) puanla:
- mantiksal_tutarlilik: Argüman mantıksal olarak tutarlı mı? İç çelişki var mı?
- kanit_kalitesi: Argüman somut veri veya kanıtlara dayanıyor mu?
- arguman_gucu: Argüman ikna edici mi? Tartışmayı ne kadar etkiliyor?

KESİNLİKLE sadece JSON formatında dön:
{
    "mantiksal_tutarlilik": 8,
    "kanit_kalitesi": 7,
    "arguman_gucu": 8
}"""

    return ask_ollama(sys_prompt, f"Değerlendirilecek Argüman: {argument_text}")


# ============================================================
# 3) JUDGE AGENT
# Görev: Tüm tartışmayı analiz etmek ve final kararı üretmek.
#        Pro/Contra argümanlarını, evidence verifier raporlarını
#        ve debate scoring sonuçlarını değerlendirerek nihai
#        kararı oluşturur.
# ============================================================
def judge_agent(question, research, debate_history, verifier_logs):
    """
    Sistemin tüm verilerini analiz ederek nihai kararı üretir.

    Parametreler:
        question       (str): Kullanıcının orijinal sorusu.
        research       (str): Research Agent'ın ürettiği bilgi (JSON string).
        debate_history (str): Tüm turların tartışma geçmişi.
        verifier_logs  (str): Evidence Verifier raporları (JSON string).

    Döndürür (dict):
        {
            "Nihai_Karar":  "Tartışmanın genel özeti ve varılan sonuç",
            "Oneri":        "Destekle / Reddet / Belirsiz",
            "Reasoning":    ["Gerekçe 1", "Gerekçe 2", "Gerekçe 3"],
            "Guven_Skoru":  85   (0-100 arası)
        }
    """
    sys_prompt = """Sen bir Judge Agent'sın (Hakim).
Görevin: Sistemin tüm araştırma, kanıt doğrulama ve pro/contra tartışmalarını
objektif olarak analiz edip bir karara bağlamak.

Karar verirken şunları dikkate al:
1. Research Agent'ın sunduğu temel gerçekler
2. Pro Agent'ın argümanlarının güçlü ve zayıf yönleri
3. Contra Agent'ın karşı argümanlarının güçlü ve zayıf yönleri
4. Evidence Verifier'ın güvenilirlik raporları
5. Debate Scoring sonuçları (mantıksal tutarlılık, kanıt kalitesi, argüman gücü)

KESİNLİKLE sadece JSON formatında dön:
{
    "Nihai_Karar": "Tartışmanın genel bir özeti ve varılan sonuç",
    "Oneri": "SADECE Destekle, Reddet veya Belirsiz kullanabilirsin",
    "Reasoning": [
        "Gerekçe 1 (Örn: Pro tarafın X argümanı güçlü kanıtlarla desteklendi)",
        "Gerekçe 2 (Örn: Contra tarafın Y riski ciddi ve göz ardı edilemez)",
        "Gerekçe 3 (Örn: Evidence Verifier raporlarına göre Pro taraf daha güvenilir)"
    ],
    "Guven_Skoru": 85
}

Güven Skoru Kriterleri:
- 80-100: Tartışma net bir yöne işaret ediyor, güçlü kanıtlar var
- 50-79:  Her iki tarafın da güçlü argümanları var, kesin bir sonuç zor
- 0-49:   Tartışma yetersiz veya kanıtlar zayıf, karar belirsiz"""

    user_prompt = (
        f"Soru: {question}\n\n"
        f"Araştırma: {research}\n\n"
        f"Tartışma Geçmişi:\n{debate_history}\n\n"
        f"Evidence Verifier Raporları:\n{verifier_logs}"
    )

    return ask_ollama(sys_prompt, user_prompt)


# ============================================================
# TEST / DEMO
# Bu bölüm sadece bu dosya doğrudan çalıştırıldığında aktif
# olur. Entegrasyon sırasında import ile kullanılacaktır.
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print(" Oğuz Modülü - Bağımsız Test")
    print("=" * 60)

    # ---- Evidence Verifier Testi ----
    print("\n[1] Evidence Verifier Testi")
    print("-" * 40)
    test_claim = "Yapay zeka, 2030 yılına kadar tüm rutin işleri otomatize edecek."
    ev_result = evidence_verifier(test_claim)
    print(json.dumps(ev_result, indent=4, ensure_ascii=False))

    # ---- Debate Scoring Testi ----
    print("\n[2] Debate Scoring Testi")
    print("-" * 40)
    test_argument = "Yapay zeka sağlık sektöründe tanı doğruluğunu %40 artırmıştır."
    ds_result = debate_scoring(test_argument)
    print(json.dumps(ds_result, indent=4, ensure_ascii=False))

    # ---- Judge Agent Testi ----
    print("\n[3] Judge Agent Testi")
    print("-" * 40)
    test_question = "Yapay zeka eğitim sisteminde öğretmenlerin yerini almalı mı?"
    test_research = json.dumps({
        "Konu_Ozeti": "Yapay zekanın eğitimdeki kullanımı hızla artmaktadır.",
        "Gercekler": [
            "AI tabanlı eğitim platformları öğrenci başarısını artırabiliyor.",
            "Öğretmenler duygusal zeka ve mentorluk sağlar.",
            "Birçok ülke AI destekli eğitim pilot programları başlattı."
        ]
    }, ensure_ascii=False)
    test_debate = (
        "Tur 1 - Pro: AI kişiselleştirilmiş eğitim sunabilir.\n"
        "Tur 1 - Contra: AI empati ve motivasyon sağlayamaz.\n"
        "Tur 2 - Pro: AI 7/24 erişilebilir ve ölçeklenebilir.\n"
        "Tur 2 - Contra: Dijital uçurum eşitsizliği artırır.\n"
        "Tur 3 - Pro: Hibrit model en iyi çözümdür.\n"
        "Tur 3 - Contra: Tam otomasyon riskli, düzenleme gerekli."
    )
    test_verifier = json.dumps([
        {"Tur": 1, "Taraf": "Pro",    "Doğrulama": {"Confidence": "High"},   "Skor": {"arguman_gucu": 8}},
        {"Tur": 1, "Taraf": "Contra", "Doğrulama": {"Confidence": "Medium"}, "Skor": {"arguman_gucu": 7}},
        {"Tur": 2, "Taraf": "Pro",    "Doğrulama": {"Confidence": "High"},   "Skor": {"arguman_gucu": 9}},
        {"Tur": 2, "Taraf": "Contra", "Doğrulama": {"Confidence": "High"},   "Skor": {"arguman_gucu": 8}},
        {"Tur": 3, "Taraf": "Pro",    "Doğrulama": {"Confidence": "Medium"}, "Skor": {"arguman_gucu": 7}},
        {"Tur": 3, "Taraf": "Contra", "Doğrulama": {"Confidence": "Medium"}, "Skor": {"arguman_gucu": 7}},
    ], ensure_ascii=False)

    judge_result = judge_agent(test_question, test_research, test_debate, test_verifier)
    print(json.dumps(judge_result, indent=4, ensure_ascii=False))

    print("\n" + "=" * 60)
    print(" Test Tamamlandı!")
    print("=" * 60)
