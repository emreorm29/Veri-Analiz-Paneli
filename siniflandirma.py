import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. Hazır veri setini yükle
iris = load_iris()
X = iris.data  # Çiçek özellikleri (boy, en)
y = iris.target # Çiçek türleri (0, 1, 2)

# 2. Veriyi Böl (Eğitim ve Test olarak)
# Verinin %20'sini modeli test etmek için ayırıyoruz
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Model Seçimi: Random Forest (Rastgele Orman)
# Sektörde en çok kullanılan, çok güçlü bir algoritmadır
model = RandomForestClassifier()

# 4. Eğitim
model.fit(X_train, y_train)

# 5. Tahmin ve Başarı Ölçümü
tahminler = model.predict(X_test)
basari = accuracy_score(y_test, tahminler)

print(f"Modelin Sınıflandırma Başarısı: %{basari*100:.2f}")

# Örnek bir tahmin yapalım: Yeni bir çiçek bulduk diyelim
yeni_cicek = [[5.1, 3.5, 1.4, 0.2]]
tahmin_turu = model.predict(yeni_cicek)
print(f"Yeni çiçeğin tahmini türü: {iris.target_names[tahmin_turu][0]}")
import matplotlib.pyplot as plt

# Hangi özelliğin ne kadar önemli olduğunu alalım
onem_dereceleri = pd.Series(model.feature_importances_, index=iris.feature_names)

# Görselleştirelim
onem_dereceleri.nlargest(4).plot(kind='barh', color='teal')
plt.title('Model Karar Verirken Hangi Özelliğe Baktı?')
plt.xlabel('Önem Skoru')
plt.show()
# Ortalamayı bul
ortalama = df['Upper_Bound'].mean()

# Yeni bir sınıf sütunu oluştur: Ortalamadan büyükse 1, küçükse 0
df['Hedef_Sinif'] = (df['Upper_Bound'] > ortalama).astype(int)

print(df[['Upper_Bound', 'Hedef_Sinif']].head())