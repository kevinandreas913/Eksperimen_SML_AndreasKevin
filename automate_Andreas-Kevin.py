import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import warnings

def kolomoutlier(df):
    outlier_columns = []
    for col in df.select_dtypes(include=[np.number]).columns:
        if col == 'Outcome':
            continue
            
        q25, q75 = np.percentile(df[col], 25), np.percentile(df[col], 75)
        iqr = q75 - q25
        
        if iqr == 0:
            continue
            
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

def preprocess_data(df_raw, test_size=0.2, random_state=42):
    warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)
    
    # 1. Menghapus Outlier (sesuai logika Anda)
    print(f"Jumlah data asli: {len(df_raw)}")
    columns_with_outliers = kolomoutlier(df_raw)
    print(f"Kolom dengan outlier terdeteksi: {columns_with_outliers}")
    
    df_clean = remove_outliers(df_raw, columns_with_outliers)
    print(f"Jumlah data setelah menghapus outlier: {len(df_clean)}")
    
    if df_clean.empty:
        print("Semua data terhapus setelah pembersihan outlier.")
        return None, None, None, None

    # 2. Memisahkan Fitur (X) dan Target (y)
    X = df_clean.drop(['Outcome'], axis=1)
    y = df_clean['Outcome']

    # 3. Membagi Data Menjadi Train dan Test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    print(f"Data dibagi menjadi train set ({X_train.shape[0]} sampel) dan test set ({X_test.shape[0]} sampel).")

    # 4. Normalisasi (Scaling)
    scaler = MinMaxScaler()
    
    X_train_scaled = scaler.fit_transform(X_train)
    
    X_test_scaled = scaler.transform(X_test)
    
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X.columns)
    
    print("Preprocessing Selesai")
    
    return X_train_scaled_df, X_test_scaled_df, y_train, y_test

if __name__ == "__main__":
    
    try:
        df_diabetes_raw = pd.read_csv("Diabetes.csv", sep=",")
        print("Dataset 'Diabetes.csv' berhasil dimuat.")
        
        X_train_final, X_test_final, y_train_final, y_test_final = preprocess_data(df_diabetes_raw)
        
        if X_train_final is not None:
            print("\n--- Hasil Akhir Siap untuk Model ---")
            print("\nX_train (data latih fitur) (5 baris pertama):")
            print(X_train_final.head())
            
            print(f"\nUkuran X_train: {X_train_final.shape}")
            print(f"Ukuran y_train: {y_train_final.shape}")
            print(f"Ukuran X_test: {X_test_final.shape}")
            print(f"Ukuran y_test: {y_test_final.shape}")
            
            print("\nDeskripsi X_train (setelah scaling):")
            print(X_train_final.describe())

    except FileNotFoundError:
        print("File 'Diabetes.csv' tidak ditemukan.")
        print("Silakan ganti 'Diabetes.csv' dengan path yang benar ke file Anda.")
    except Exception as e:
        print(f"Terjadi error: {e}")
