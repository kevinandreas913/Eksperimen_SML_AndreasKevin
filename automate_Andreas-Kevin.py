import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import sys

def kolomoutlier(df):
    outlier_columns = []
    for col in df.select_dtypes(include=[np.number]).columns:
        q25, q75 = np.percentile(df[col], 25), np.percentile(df[col], 75)
        iqr = q75 - q25
        
        cut_off = iqr * 1.5
        minimum, maximum = q25 - cut_off, q75 + cut_off

        outliers = df[(df[col] < minimum) | (df[col] > maximum)]
        if not outliers.empty:  # Jika ada outlier, tambahkan ke list
            outlier_columns.append(col)
    
    return outlier_columns

def remove_outliers(df, columns):
    df_clean = df.copy()
    for col in columns:
        q25, q75 = np.percentile(df_clean[col], 25), np.percentile(df_clean[col], 75)
        iqr = q75 - q25

        if iqr == 0:
            continue
            
        cut_off = iqr * 1.5
        minimum, maximum = q25 - cut_off, q75 + cut_off

        df_clean = df_clean[(df_clean[col] >= minimum) & (df_clean[col] <= maximum)]
    
    return df_clean

def clean_and_normalize_data(df_raw):
    """
    Menggabungkan proses cleaning dan normalisasi.
    """
    print(f"Jumlah data asli: {len(df_raw)}")
    
    columns_with_outliers = kolomoutlier(df_raw)
    print(f"Kolom dengan outlier terdeteksi: {columns_with_outliers}")
    
    df_clean = remove_outliers(df_raw, columns_with_outliers)
    print(f"Jumlah data setelah menghapus outlier: {len(df_clean)}")
    
    if df_clean.empty:
        print("Semua data terhapus setelah pembersihan outlier.")
        return None
    
    original_columns = df_clean.columns
    
    print("Menerapkan normalisasi MinMaxScaler ke seluruh data...")
    scaler = MinMaxScaler()
    df_scaled_data = scaler.fit_transform(df_clean)
    
    # Ubah kembali menjadi DataFrame dengan nama kolom yang benar
    df_final = pd.DataFrame(df_scaled_data, columns=original_columns)
    
    print("Pembersihan dan normalisasi selesai.")
    return df_final

if __name__ == "__main__":
    dataset = "Diabetes.csv"
    dataset_bersih = "diabetes_cleaned.csv"
    
    try:
        df_diabetes_raw = pd.read_csv(dataset, sep=",")
        print(f"Dataset '{dataset}' berhasil dimuat.")
        
        df_processed = clean_and_normalize_data(df_diabetes_raw)
        
        if df_processed is not None:
            df_processed.to_csv(dataset_bersih, index=False)
            
            print(f"\nData bersih telah disimpan ke: {dataset_bersih}")
            print("5 baris pertama data bersih:")
            print(df_processed.head())
            print(f"\nUkuran Data Final: {df_processed.shape}")

    except FileNotFoundError:
        print(f"File '{dataset}' tidak ditemukan.", file=sys.stderr)
    except Exception as e:
        print(f"Terjadi error yang tidak terduga: {e}", file=sys.stderr)