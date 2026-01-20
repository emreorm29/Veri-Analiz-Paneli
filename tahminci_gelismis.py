import pandas as pd
import numpy as np
import sqlite3
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from datetime import timedelta
from sklearn.metrics import mean_squared_error



# 1. SQL'den veriyi çek
conn = sqlite3.connect('finans_verileri.db')
df = pd.read_sql("SELECT Date, Close FROM hisseler", conn)
conn.close()

df['Date'] = pd.to_datetime(df['Date'])

# 2. Özellik Mühendisliği (Feature Engineering)
# Modele zamanın ruhunu öğretiyoruz
df['Day'] = df['Date'].dt.day
df['Month'] = df['Date'].dt.month
df['DayOfWeek'] = df['Date'].dt.dayofweek
df['Ordinal'] = df['Date'].map(lambda x: x.toordinal())

# 3. Modeli Eğit
X = df[['Ordinal', 'Day', 'Month', 'DayOfWeek']]
y = df['Close']

# ... (Veri çekme ve özellik hazırlama kısımları yukarıda kalıyor)

# 1. Önce modeli tanımla
model = RandomForestRegressor(n_estimators=100, random_state=42)

# 2. Modeli eğit (Bu satırı atlamış olabilirsin)
model.fit(X, y)

# 3. ŞİMDİ tahmin yapabilirsin
gecmis_tahminler = model.predict(X)
rmse = np.sqrt(mean_squared_error(y, gecmis_tahminler))

print(f"Modelin Ortalama Hata Payı (RMSE): {rmse:.2f}")

# 4. Gelecek 30 Günü Hazırla
son_tarih = df['Date'].max()
gelecek_tarihler = [son_tarih + timedelta(days=i) for i in range(1, 31)]

gelecek_ozellikler = []
for d in gelecek_tarihler:
    gelecek_ozellikler.append([d.toordinal(), d.day, d.month, d.dayofweek])

# 5. Tahmin Yap
tahminler = model.predict(gelecek_ozellikler)

# 6. Görselleştir
plt.figure(figsize=(12, 6))
plt.plot(df['Date'].tail(100), df['Close'].tail(100), label='Son 100 Gün Gerçek Veri')
plt.plot(gelecek_tarihler, tahminler, label='Gelişmiş 30 Günlük Tahmin', color='green', marker='o', markersize=4)
plt.title('Gümüş Fiyatı Gelişmiş Tahmin Modeli (Random Forest)')
plt.legend()
plt.grid(True)
plt.show()