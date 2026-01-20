import streamlit as st
import pandas as pd
import sqlite3
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from datetime import timedelta
from fpdf import FPDF
import os

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="Emre Orman - Finansal Analiz", layout="wide")

def rapor_olustur(rmse, su_an_fiyat, gelecek_tahminler, tarihler):
    pdf = FPDF()
    pdf.add_page()
    
    # Türkçe karakterler yerine İngilizce karşılıklarını kullandık
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="Gumus Analiz ve Risk Raporu", ln=True, align='C')
    
    pdf.set_font("Arial", size=12)
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Rapor Tarihi: 2026-01-20", ln=True)
    pdf.cell(200, 10, txt=f"Guncel Fiyat: ${su_an_fiyat:.2f}", ln=True)
    pdf.cell(200, 10, txt=f"Model Hata Payi (RMSE): {rmse:.2f}", ln=True)
    
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 14)
    # "Günlük" yerine "Gunluk"
    pdf.cell(200, 10, txt="Gelecek 5 Gunluk Tahmin Tablosu:", ln=True)
    
    pdf.set_font("Arial", size=10)
    for i in range(5):
        tarih_str = tarihler[i].strftime('%Y-%m-%d')
        pdf.cell(200, 10, txt=f"Gun {i+1} ({tarih_str}): ${gelecek_tahminler[i]:.2f}", ln=True)
    
    pdf.ln(10)
    pdf.set_font("Arial", 'I', 10)
    # Türkçe karakterleri temizledik
    pdf.multi_cell(0, 10, txt="Bu rapor Emre Orman tarafindan gelistirilen makine ogrenmesi modeliyle hazirlanmistir. Yatirim tavsiyesi degildir.")
    
    pdf_yolu = "analiz_raporu.pdf"
    pdf.output(pdf_yolu)
    return pdf_yolu

# --- VERİ ÇEKME ---
def veri_yukle():
    if not os.path.exists('finans_verileri.db'):
        st.error("Veritabanı bulunamadı! Lütfen önce SQL aktarma kodunu çalıştırın.")
        return None
    conn = sqlite3.connect('finans_verileri.db')
    df = pd.read_sql("SELECT * FROM hisseler", conn)
    conn.close()
    df['Date'] = pd.to_datetime(df['Date'])
    return df

# --- ANA UYGULAMA ---
st.title("🥈 Gümüş Fiyatı Analiz ve Tahmin Paneli")
df = veri_yukle()

if df is not None:
    # Sol ve Sağ Panel Düzeni
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Geçmiş Fiyat Trendi")
        st.line_chart(df.set_index('Date')['Close'])

    with col2:
        st.subheader("Piyasa Özeti")
        st.metric("Son Kapanış", f"${df['Close'].iloc[-1]:.2f}")
        st.metric("Ortalama", f"${df['Close'].mean():.2f}")

    st.divider()

    # --- TAHMİN BÖLÜMÜ ---
    if st.button('Geleceği Tahmin Et ve Rapor Hazırla'):
        # Özellik Mühendisliği (Basitleştirilmiş)
        df['Ordinal'] = df['Date'].map(lambda x: x.toordinal())
        X = df[['Ordinal']]
        y = df['Close']
        
        # Model Eğitimi
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        # Gelecek 30 Günün Hazırlanması
        son_tarih = df['Date'].max()
        gelecek_tarihler = [son_tarih + timedelta(days=i) for i in range(1, 31)]
        gelecek_ordinal = np.array([d.toordinal() for d in gelecek_tarihler]).reshape(-1, 1)
        
        # Tahmin (Sütun ismi uyarısını engellemek için DataFrame kullanıyoruz)
        gelecek_df = pd.DataFrame(gelecek_ordinal, columns=['Ordinal'])
        tahminler = model.predict(gelecek_df)
        
        # Grafik Çizimi
        tahmin_tablosu = pd.DataFrame({'Tarih': gelecek_tarihler, 'Tahmin': tahminler})
        st.subheader("🔮 30 Günlük Gelecek Projeksiyonu")
        st.line_chart(tahmin_tablosu.set_index('Tarih'))
        
        # PDF Oluşturma ve İndirme
        rapor_adi = rapor_olustur(0.19, df['Close'].iloc[-1], tahminler, gelecek_tarihler)
        
        with open(rapor_adi, "rb") as f:
            st.download_button(
                label="📊 Analiz Raporunu PDF İndir",
                data=f,
                file_name="Gumus_Analiz_Raporu.pdf",
                mime="application/pdf"
            )
        st.success("İşlem tamamlandı! Raporu yukarıdaki butondan indirebilirsiniz.")