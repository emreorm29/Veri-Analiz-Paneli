import pandas as pd
import sqlite3
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# 1. SQL'den veriyi çek
conn = sqlite3.connect('sigorta.db')
df = pd.read_sql("SELECT * FROM musteriler", conn)
conn.close()

# 2. Özellikleri ve Hedefi Belirle
X = df[['Yas', 'Ehliyet_Yili', 'Arac_Degeri']] # Giriş verileri
y = df['Hasar_Kaydi'] # Tahmin edilecek olan (0: Hasar yok, 1: Hasar var)

# 3. Eğitim ve Test Setine Ayır (%20 test için)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Aktüeryal Model (Sınıflandırıcı) Kur
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# 5. Başarıyı Ölç
tahminler = model.predict(X_test)
basari = accuracy_score(y_test, tahminler)

print(f"Modelin Tahmin Basarisi: %{basari*100:.2f}")

# 6. Örnek Bir Müşteri İçin Tahmin Yap
# Örn: 22 yaşında, 1 yıllık ehliyetli, 1.5M TL'lik aracı olan biri
yeni_musteri = [[22, 1, 1500000]]
tahmin_sonuc = model.predict(yeni_musteri)
olasilik = model.predict_proba(yeni_musteri)

print(f"\nYeni Musteri Tahmini: {'Hasar Yapar' if tahmin_sonuc[0] == 1 else 'Hasar Yapmaz'}")
print(f"Hasar Yapma Olasiligi: %{olasilik[0][1]*100:.2f}")
# 1. Özellik önem derecelerini al
onem_dereceleri = model.feature_importances_
ozellikler = ['Yas', 'Ehliyet_Yili', 'Arac_Degeri']

# 2. Görselleştirme
plt.figure(figsize=(10, 6))
plt.barh(ozellikler, onem_dereceleri, color='skyblue')
plt.xlabel('Etki Oranı (Önem Derecesi)')
plt.ylabel('Müşteri Özellikleri')
plt.title('Hasar Riskini Belirleyen En Önemli Faktörler')
plt.grid(axis='x', linestyle='--', alpha=0.7)

plt.show()