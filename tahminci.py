import streamlit as st
import pandas as pd
import sqlite3
from sklearn.ensemble import RandomForestRegressor
from datetime import timedelta
import numpy as np

st.set_page_config(page_title="Emre Orman - Gümüş Analiz", layout="wide")
st.title("🥈 Gümüş Fiyatı Analiz ve Tahmin Paneli")

# 1. SQL Bağlantısı
def veri_getir():
    conn = sqlite3.connect('finans_verileri.db')
    df = pd.read_sql("SELECT * FROM hisseler", conn)
    conn.close()
    df['Date'] = pd.to_datetime(df['Date'])
    return df

df = veri_getir()

# 2. Arayüz Düzeni (Sütunlar)
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Geçmiş Trend")
    st.line_chart(df.set_index('Date')['Close'])

with col2:
    st.subheader("İstatistikler")
    st.metric("Güncel Fiyat", f"${df['Close'].iloc[-1]:.2f}")
    st.metric("Ortalama Fiyat", f"${df['Close'].mean():.2f}")

# 3. Tahmin Butonu ve Gelişmiş Model
if st.button('Gelecek 30 Günü Tahmin Et'):
    # Özellik Hazırlama
    df['Ordinal'] = df['Date'].map(lambda x: x.toordinal())
    X = df[['Ordinal']] # Basitleştirilmiş versiyon
    y = df['Close']
    
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X, y)
    
    # Gelecek Tarihler
    son_tarih = df['Date'].max()
    gelecek_tarihler = [son_tarih + timedelta(days=i) for i in range(1, 31)]
    gelecek_ordinal = np.array([d.toordinal() for d in gelecek_tarihler]).reshape(-1, 1)
    
    tahminler = model.predict(gelecek_ordinal)
    
    # Tahmin Sonucu
    tahmin_df = pd.DataFrame({'Tarih': gelecek_tarihler, 'Tahmin': tahminler})
    st.subheader("🔮 30 Günlük Gelecek Projeksiyonu")
    st.line_chart(tahmin_df.set_index('Tarih'))
    st.success("Aktüeryal tahmin başarıyla tamamlandı!")