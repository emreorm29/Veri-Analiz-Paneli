import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime
from sklearn.linear_model import LinearRegression

# Dosya adını tırnak içine tam yaz
df = pd.read_csv('silver_prices_forecast_2026.csv') 

# İlk 5 satırı görelim
print(df.head())
# Verideki sayısal sütunların ortalamasını, min ve max değerlerini gösterir
print(df.describe())

# Hangi sütunda kaç tane boş (NaN) değer var?
print(df.isnull().sum())
# Tarih sütununu Python'ın anlayacağı tarih formatına çevirelim
df['Date'] = pd.to_datetime(df['Date'])

# Grafik oluşturma
plt.figure(figsize=(10, 5))
plt.plot(df['Date'], df['Upper_Bound'], label='Üst Sınır', color='orange')
plt.title('Zamana Göre Değer Değişimi')
plt.xlabel('Tarih')
plt.ylabel('Değer')
plt.legend()
plt.grid(True)
plt.show()
en_yuksek_gun = df[df['Upper_Bound'] == df['Upper_Bound'].max()]
print("En yüksek değerin görüldüğü gün:")
print(en_yuksek_gun)
# 1. Tarihleri sayısal bir değere çevirelim (Ordinal format)
df['Date_Ordinal'] = df['Date'].map(datetime.date.toordinal)

# X (Girdi: Tarihler), y (Çıktı: Değerler)
X = df['Date_Ordinal'].values.reshape(-1, 1)
y = df['Upper_Bound'].values

# 2. Modeli kuralım ve eğitelim
model = LinearRegression()
model.fit(X, y)

# 3. Gelecek için bir tarih belirleyelim (Örn: 2026-01-15)
import datetime
gelecek_tarih = datetime.date(2026, 1, 15).toordinal()
tahmin = model.predict([[gelecek_tarih]])

print(f"2026-01-15 tarihi için tahmini değer: {tahmin[0]:.2f}")
# Mevcut veriler üzerine tahmin çizgisini çizdirelim
plt.figure(figsize=(10, 5))
plt.scatter(df['Date'], df['Upper_Bound'], color='blue', label='Gerçek Veri')
plt.plot(df['Date'], model.predict(X), color='red', linewidth=2, label='Trend Çizgisi (Tahmin)')
plt.title('Zaman Serisi Trend Analizi')
plt.legend()
plt.show()
from sklearn.metrics import r2_score

# Modelin tahminlerini al
y_tahmin = model.predict(X)

# Başarı puanını hesapla
skor = r2_score(y, y_tahmin)

print(f"Modelin Başarı Puanı (R-Squared): {skor:.4f}")