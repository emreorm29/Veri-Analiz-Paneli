import pandas as pd
import numpy as np
import sqlite3

# 1. 1000 kişilik yapay sigorta verisi oluşturalım
data = {
    'Yas': np.random.randint(18, 70, 1000),
    'Ehliyet_Yili': np.random.randint(0, 50, 1000),
    'Arac_Degeri': np.random.randint(200000, 2000000, 1000),
    'Hasar_Kaydi': np.random.choice([0, 1], size=1000, p=[0.8, 0.2]) # %20 hasar ihtimali
}

df = pd.DataFrame(data)

# 2. Bu veriyi SQL'e atalım
conn = sqlite3.connect('sigorta.db')
df.to_sql('musteriler', conn, if_exists='replace', index=False)
conn.close()

print("Sigorta verisi SQL'e yüklendi!")