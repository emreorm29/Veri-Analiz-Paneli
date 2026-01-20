import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sqlite3

conn = sqlite3.connect('sigorta_guncel.db')
df = pd.read_sql("SELECT * FROM musteriler", conn)
conn.close()

# Isı haritası (Heatmap) oluşturma
plt.figure(figsize=(10, 8))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Değişkenlerin Hasar Kaydı Üzerindeki Etkisi (Korelasyon)')
plt.show()