import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. Kendi verini oku (df burada tanımlanıyor)
df = pd.read_csv('silver_prices_forecast_2026.csv') 

# 2. Sınıflandırma için hedef değişkeni oluştur
ortalama = df['Upper_Bound'].mean()
df['Hedef_Sinif'] = (df['Upper_Bound'] > ortalama).astype(int)

# 3. Model için veriyi hazırla
# Tarih sütunu varsa onu şimdilik eğitimden çıkaralım çünkü metin formatındadır
X = df.drop(['Date', 'Hedef_Sinif'], axis=1, errors='ignore') 
y = df['Hedef_Sinif']

# 4. Eğitim ve Test bölmesi
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Model kurma ve eğitme
model = RandomForestClassifier()
model.fit(X_train, y_train)

# 6. Sonuç
tahminler = model.predict(X_test)
print(f"Kendi verin üzerindeki başarı puanı: %{accuracy_score(y_test, tahminler)*100:.2f}")