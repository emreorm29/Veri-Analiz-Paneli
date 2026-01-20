import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.title("Emre Orman - Dinamik Veri Analiz Paneli")

# 1. Veriyi Yükle
df = pd.read_csv('silver_prices_forecast_2026.csv') 

# Sütun isimlerini tamamen temizle (gizli karakterler vs. gitsin)
df.columns = df.columns.str.strip().str.replace('"', '').str.replace("'", "")

# 2. Hangi sütunu kullanacağımızı otomatik seçelim
# Sadece sayısal (float veya int) sütunları bul
sayisal_sutunlar = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

if len(sayisal_sutunlar) > 0:
    # 'Close' varsa onu al, yoksa ilk sayısal sütunu al
    hedef_sutun = 'Close' if 'Close' in sayisal_sutunlar else sayisal_sutunlar[0]
    
    st.info(f"Analiz edilen sütun: {hedef_sutun}")

    # 3. Hesaplamalar
    ortalama_deger = df[hedef_sutun].mean()
    df['Hedef_Sinif'] = (df[hedef_sutun] > ortalama_deger).astype(int)

    # Modeli hazırla
    X = df[sayisal_sutunlar].drop(columns=[hedef_sutun], errors='ignore')
    y = df['Hedef_Sinif']

    # 4. Model Eğit
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X, y)

    # 5. Görselleştirme
    st.line_chart(df[hedef_sutun])
    st.success(f"Model {hedef_sutun} verisine göre başarıyla eğitildi!")
else:
    st.error("Dosyada sayısal veri bulunamadı. Lütfen CSV dosyasını kontrol et.")
# 1. Günlük Getiri ve Volatilite Hesabı
df['Daily_Return'] = df[hedef_sutun].pct_change()
volatilite = df['Daily_Return'].std() * (252**0.5) # Yıllıklandırılmış standart sapma

st.subheader("Aktüeryal Risk Analizi")
st.metric(label="Yıllıklandırılmış Volatilite (Risk Skoru)", value=f"%{volatilite*100:.2f}")

if volatilite > 0.30:
    st.warning("Bu varlık yüksek risk grubunda! (Yüksek Volatilite)")
else:
    st.success("Bu varlık düşük/orta risk grubunda.")