import sqlite3
import pandas as pd

# 1. Veri tabanına bağlan
conn = sqlite3.connect('finans_verileri.db')

# 2. Mevcut gümüş verini oku
# Klasöründeki gerçek dosya adını yazdık
df = pd.read_csv('silver_prices_historical.csv')

# Sütun isimlerini temizleyelim
df.columns = df.columns.str.strip()

# 3. Veriyi SQL tablosuna aktar
df.to_sql('hisseler', conn, if_exists='replace', index=False)

print("Veriler SQL veri tabanına başarıyla aktarıldı!")

# 4. SQL Sorgusu (Örnek: Kapanış fiyatı 25'ten büyük olan ilk 5 veri)
# Gümüş fiyatları genelde 20-30 bandında olduğu için 25 yaptık
sorgu = "SELECT Date, Close FROM hisseler WHERE Close > 25 LIMIT 5"
sonuc_df = pd.read_sql(sorgu, conn)

print("\n--- SQL Sorgu Sonucu (Fiyatı 25'ten Büyük Olanlar) ---")
print(sonuc_df)

conn.close()