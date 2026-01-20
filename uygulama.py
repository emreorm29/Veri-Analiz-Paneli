import streamlit as st
import pandas as pd
import sqlite3
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from datetime import timedelta
from fpdf import FPDF
import os

# Sayfa Ayarları
st.set_page_config(page_title="Emre Orman - Finans & Aktuerya", layout="wide")

# --- PDF OLUSTURMA FONKSIYONLARI ---
def gumus_pdf_olustur(fiyat, tahminler):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="Gumus Analiz Raporu", ln=True, align='C')
    pdf.set_font("Arial", size=12)
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Guncel Fiyat: ${fiyat:.2f}", ln=True)
    pdf.ln(5)
    pdf.cell(200, 10, txt="Gelecek 5 Gunluk Tahminler:", ln=True)
    for i in range(5):
        pdf.cell(200, 10, txt=f"Gun {i+1}: ${tahminler[i]:.2f}", ln=True)
    pdf.output("gumus_rapor.pdf")
    return "gumus_rapor.pdf"

def sigorta_pdf_olustur(yas, olasilik, prim):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="Sigorta Teklif Formu", ln=True, align='C')
    pdf.set_font("Arial", size=12)
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Musteri Yas: {yas}", ln=True)
    pdf.cell(200, 10, txt=f"Hasar Olasiligi: %{olasilik*100:.1f}", ln=True)
    pdf.cell(200, 10, txt=f"Onerilen Yillik Prim: {prim:,.2f} TL", ln=True)
    pdf.ln(10)
    pdf.set_font("Arial", 'I', 10)
    pdf.cell(200, 10, txt="Bu belge Emre Orman tarafindan hazirlanmistir.", ln=True)
    pdf.output("sigorta_teklif.pdf")
    return "sigorta_teklif.pdf"

# --- MODEL HAZIRLIGI ---
@st.cache_resource
def modelleri_hazirla():
    # Gumus
    conn_g = sqlite3.connect('finans_verileri.db')
    df_g = pd.read_sql("SELECT * FROM hisseler", conn_g)
    conn_g.close()
    df_g['Date'] = pd.to_datetime(df_g['Date'])
    df_g['Ordinal'] = df_g['Date'].map(lambda x: x.toordinal())
    mod_g = RandomForestRegressor(n_estimators=100, random_state=42).fit(df_g[['Ordinal']], df_g['Close'])
    # Sigorta
    conn_s = sqlite3.connect('sigorta_guncel.db')
    df_s = pd.read_sql("SELECT * FROM musteriler", conn_s)
    conn_s.close()
    X_s = df_s[['Yas', 'Ehliyet_Yili', 'Arac_Degeri', 'Sehir_Kodu', 'Arac_Tipi']]
    mod_s = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_s, df_s['Hasar_Kaydi'])
    return df_g, mod_g, mod_s

df_gumus, model_gumus, model_sigorta = modelleri_hazirla()

# Sekmeler
tab1, tab2 = st.tabs(["🥈 Gumus Tahmini", "🛡️ Sigorta Prim Hesaplama"])

# --- TAB 1: GUMUS ---
with tab1:
    st.header("Gumus Fiyat Analizi")
    st.line_chart(df_gumus.set_index('Date')['Close'])
    
    if st.button('Tahmin Et ve PDF Hazirla', key='g_pdf_btn'):
        son_tarih = df_gumus['Date'].max()
        gelecek_df = pd.DataFrame([son_tarih.toordinal() + i for i in range(1, 31)], columns=['Ordinal'])
        tahminler = model_gumus.predict(gelecek_df)
        
        st.line_chart(tahminler)
        pdf_yolu = gumus_pdf_olustur(df_gumus['Close'].iloc[-1], tahminler)
        with open(pdf_yolu, "rb") as f:
            st.download_button("📊 Gumus Raporunu Indir", f, file_name="Gumus_Analiz.pdf")

# --- TAB 2: SIGORTA ---
with tab2:
    st.header("Sigorta Risk Analizi")
    c1, c2, c3 = st.columns(3)
    yas = c1.number_input("Yas", 18, 80, 30)
    ehliyet = c2.number_input("Ehliyet Yili", 0, 50, 10)
    deger = c3.number_input("Arac Degeri (TL)", 100000, 5000000, 500000)
    sehir = st.selectbox("Sehir", [0,1,2,3], format_func=lambda x: ["Istanbul","Ankara","Izmir","Diger"][x])
    tip = st.selectbox("Arac Tipi", [0,1,2], format_func=lambda x: ["Spor","Sedan","SUV"][x])

    if st.button("Risk Analizi ve PDF Teklif Al"):
        girdi = pd.DataFrame([[yas, ehliyet, deger, sehir, tip]], columns=['Yas', 'Ehliyet_Yili', 'Arac_Degeri', 'Sehir_Kodu', 'Arac_Tipi'])
        olasilik = model_sigorta.predict_proba(girdi)[0][1]
        prim = olasilik * (deger * 0.10) * 1.25
        
        st.metric("Hasar Olasiligi", f"%{olasilik*100:.1f}")
        st.metric("Onerilen Prim", f"{prim:,.2f} TL")
        
        pdf_yolu = sigorta_pdf_olustur(yas, olasilik, prim)
        with open(pdf_yolu, "rb") as f:
            st.download_button("📄 Sigorta Teklifini Indir", f, file_name="Sigorta_Teklif.pdf")
    # --- TAB 2 İÇİNE VEYA YENİ BİR SEKME OLARAK EKLEYEBİLİRSİN ---
st.divider()
st.subheader("📊 Şirket Geneli Portföy Risk Analizi (Yönetici Özeti)")

if st.button("Tüm Portföyü Analiz Et"):
    conn = sqlite3.connect('sigorta_guncel.db')
    df_all = pd.read_sql("SELECT * FROM musteriler", conn)
    conn.close()
    
    # Tüm portföy için olasılıkları tahmin et
    X_all = df_all[['Yas', 'Ehliyet_Yili', 'Arac_Degeri', 'Sehir_Kodu', 'Arac_Tipi']]
    tahminler_all = model_sigorta.predict_proba(X_all)[:, 1]
    
    # Toplam Beklenen Hasar (Olasılık * Ortalama Hasar Maliyeti)
    df_all['Beklenen_Hasar'] = tahminler_all * (df_all['Arac_Degeri'] * 0.10)
    
    toplam_risk = df_all['Beklenen_Hasar'].sum()
    ortalama_risk = df_all['Beklenen_Hasar'].mean()
    en_yuksek_risk = df_all['Beklenen_Hasar'].max()
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Toplam Portföy Riski", f"{toplam_risk:,.2f} TL")
    m2.metric("Müşteri Başı Ortalama Risk", f"{ortalama_risk:,.2f} TL")
    m3.metric("En Yüksek Tekil Risk", f"{en_yuksek_risk:,.2f} TL")
    
    # Risk Dağılım Grafiği
    st.write("### Şirketin Maruz Kaldığı Hasar Dağılımı")
    st.bar_chart(df_all['Beklenen_Hasar'].head(50)) # İlk 50 müşteri örneği
    st.info("Bu grafik, şirketin kasasından çıkması muhtemel hasarların müşterilere göre dağılımını gösterir.")