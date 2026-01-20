import pandas as pd
import sqlite3
from sklearn.ensemble import RandomForestClassifier

# 1. SQL'den zenginleştirilmiş veriyi çek
conn = sqlite3.connect('sigorta_guncel.db')
df = pd.read_sql("SELECT * FROM musteriler", conn)
conn.close()

# 2. Risk Modelini Eğit (Olasılık tahmini için)
X = df[['Yas', 'Ehliyet_Yili', 'Arac_Degeri', 'Sehir_Kodu', 'Arac_Tipi']]
y = df['Hasar_Kaydi']
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# 3. Aktüeryal Prim Hesaplama Fonksiyonu
def prim_hesapla(yas, ehliyet, arac_degeri, sehir, tip):
    # Modelden hasar olasılığını al
    girdi = [[yas, ehliyet, arac_degeri, sehir, tip]]
    olasilik = model.predict_proba(girdi)[0][1] # Hasar yapma ihtimali (0 ile 1 arası)
    
    # Aktüeryal Varsayım: Ortalama hasar maliyeti araç değerinin %10'udur
    ortalama_maliyet = arac_degeri * 0.10
    
    # Saf Prim (Giderler ve kar marjı eklenmemiş hali)
    saf_prim = olasilik * ortalama_maliyet
    
    # Şirket Karı ve Operasyonel Gider (Saf primin %20'si)
    toplam_prim = saf_prim * 1.20
    
    return olasilik, toplam_prim

# --- TEST EDELİM ---
# Senaryo 1: Riskli Müşteri (20 yaş, İstanbul, Spor Araç)
riskli_olasilik, riskli_prim = prim_hesapla(20, 1, 1000000, 0, 0)

# Senaryo 2: Güvenli Müşteri (45 yaş, Ankara, Sedan)
guvenli_olasilik, guvenli_prim = prim_hesapla(45, 25, 1000000, 1, 1)

print(f"RISKLI MUSTERI -> Olasilik: %{riskli_olasilik*100:.1f} | Teklif Edilen Prim: {riskli_prim:,.2f} TL")
print(f"GUVENLI MUSTERI -> Olasilik: %{guvenli_olasilik*100:.1f} | Teklif Edilen Prim: {guvenli_prim:,.2f} TL")