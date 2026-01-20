import pandas as pd
import numpy as np
import sqlite3

# 1. Genişletilmiş veri simülasyonu
n = 1500 # Veriyi biraz büyütelim
data = {
    'Yas': np.random.randint(18, 75, n),
    'Ehliyet_Yili': np.random.randint(0, 50, n),
    'Arac_Degeri': np.random.randint(300000, 3000000, n),
    # Kategorik veriler: 0: İstanbul (Riskli), 1: Ankara, 2: İzmir, 3: Diğer
    'Sehir_Kodu': np.random.choice([0, 1, 2, 3], size=n, p=[0.4, 0.2, 0.2, 0.2]),
    # 0: Spor (Hızlı), 1: Sedan, 2: SUV
    'Arac_Tipi': np.random.choice([0, 1, 2], size=n, p=[0.2, 0.5, 0.3])
}

df = pd.DataFrame(data)

# 2. Hasar Kaydı Mantığını "Gerçekçileştirelim"
# (Eğer araç spor ise ve yaş küçükse hasar ihtimali artsın)
df['Hasar_Ihtimali'] = (df['Arac_Tipi'] == 0).astype(int) * 0.3 + \
                       (df['Yas'] < 25).astype(int) * 0.2 + \
                       (df['Sehir_Kodu'] == 0).astype(int) * 0.1

df['Hasar_Kaydi'] = df['Hasar_Ihtimali'].apply(lambda x: 1 if np.random.random() < x + 0.1 else 0)

# 3. SQL'e Yaz
conn = sqlite3.connect('sigorta_guncel.db')
df.drop(columns=['Hasar_Ihtimali']).to_sql('musteriler', conn, if_exists='replace', index=False)
conn.close()

print("Veri seti güncellendi: Şehir ve Araç Tipi özellikleri eklendi!")