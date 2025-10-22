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
    # stratify=y penting untuk klasifikasi
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
    RAW_DATA_PATH = "Diabetes.csv" 
    
    try:
        # Kita asumsikan 'Diabetes.csv' ada di root, sama seperti skrip ini
        df_diabetes_raw = pd.read_csv(RAW_DATA_PATH, sep=",")
        print(f"Dataset '{RAW_DATA_PATH}' berhasil dimuat.")
        
        X_train_final, X_test_final, y_train_final, y_test_final = preprocess_data(df_diabetes_raw)
        
        if X_train_final is not None:
            # 1. Reset index pada semua bagian. 
            X_train_reset = X_train_final.reset_index(drop=True)
            y_train_reset = y_train_final.reset_index(drop=True)
            X_test_reset = X_test_final.reset_index(drop=True)
            y_test_reset = y_test_final.reset_index(drop=True)

            # 2. Gabungkan fitur dan target untuk TRAIN set (axis=1 artinya gabung menyamping)
            train_set = pd.concat([X_train_reset, y_train_reset], axis=1)
            
            # 3. Gabungkan fitur dan target untuk TEST set
            test_set = pd.concat([X_test_reset, y_test_reset], axis=1)

            # 4. Simpan menjadi DUA file
            train_set.to_csv("train_set.csv", index=False)
            test_set.to_csv("test_set.csv", index=False)
            
            print(train_set.head())
            
            print(f"\nUkuran Train Set: {train_set.shape}")
            print(f"Ukuran Test Set: {test_set.shape}")

    except FileNotFoundError:
        print(f"File '{RAW_DATA_PATH}' tidak ditemukan.")
        print(f"Pastikan file '{RAW_DATA_PATH}' ada di direktori yang sama dengan skrip ini.")
    except Exception as e:
        print(f"Terjadi error: {e}")