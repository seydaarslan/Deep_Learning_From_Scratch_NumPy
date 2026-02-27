"""
    Bu dosya, proje için gerekli olan HIGGS ve RCV1/Reuters veri setlerini 
    indirmek ve modellerin eğitimi için hazır hale getirmek amacıyla hazırlandı.
"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_rcv1
from sklearn.feature_extraction.text import TfidfVectorizer

# Reuters veri seti için Keras kontrolü yapıldı.
try:
    from keras.datasets import reuters
except ImportError:
    print("Hata: Keras yüklü değil. '!pip install tensorflow keras' komutunu çalıştırın.")

"""
    Bu fonksiyon HIGGS verisini indirir, temizler ve döndürür.
    NOT: Veri seti harici olarak indirilip "HIGGS.csv.gz" ismiyle proje klasörüne eklenirse kodlar çok daha hızlı çalışacaktır.
"""
def prepare_higgs_subset(total_samples=20000, test_size=0.2, random_state=42):
    
    # Dosya kaynağı belirlendi.
    fileName = "HIGGS.csv.gz"
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz"
    
    if os.path.exists(fileName):
        print("Dosya bulundu diskten okunuyor")
        source = fileName
    else: 
        print("Dosya bulunamadı sunucudan indiriliyor")
        source = url
    
    # Sıkıştırılmış formatta veriyi okur.
    try:
        raw_df = pd.read_csv(source, compression='gzip', header=None, nrows=int(total_samples * 1.5))
    except Exception as e:
        print(f"Hata: Veri çekilemedi. Detay: {e}")
        return None, None, None, None

    # Veri ayrıştırıldı.
    data = raw_df.to_numpy()
    y_raw = data[:, 0].astype(int)
    X_raw = data[:, 1:]
    
    X_subset, _, y_subset, _ = train_test_split(
        X_raw, y_raw, train_size=total_samples, stratify=y_raw, random_state=random_state
    )

    # Ölçeklendirme (Standardizasyon) yapıldı.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_subset)

    # Eğitim ve test bölümlendi.
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_subset, test_size=test_size, random_state=random_state, stratify=y_subset)
    
    # Boyutlar modele uygun hale getirildi.
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    print("[DATA_HANDLER] HIGGS hazırlığı tamamlandı.")
    return X_train, X_test, y_train, y_test

"""
    Bu fonksiyon RCV1/Reuters veri setini indirir, temizler ve döndürür.
"""
def prepare_rcv1_subset(n_samples=None, test_size=None):
    
    print("[DATA_HANDLER] Reuters verisi hazırlanıyor.")
    
    # Veri yüklendi.
    (x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=10000)

    # Kelime indeksleri metne çevirildi.
    word_index = reuters.get_word_index()
    index_word = {v + 3: k for k, v in word_index.items()}
    index_word[0] = "<PAD>"
    index_word[1] = "<START>"
    index_word[2] = "<UNK>"

    def decode_review(seq):
        return " ".join(index_word.get(i, "?") for i in seq)

    print("Metinler decode ediliyor.")
    x_train_text = [decode_review(x) for x in x_train]
    x_test_text  = [decode_review(x) for x in x_test]

    # Etiket dönüştürme yapıldı.
    CCAT_LIKE_TOPICS = {3, 35, 36, 42, 43, 44, 45}

    y_train_bin = np.array([1 if y in CCAT_LIKE_TOPICS else 0 for y in y_train])
    y_test_bin  = np.array([1 if y in CCAT_LIKE_TOPICS else 0 for y in y_test])

    # TF-IDF dönüştürme yapıldı.
    print("TF-IDF dönüşümü yapılıyor.")
    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1,2),
        min_df=2
    )

    # Eğitim setine fit, test setine sadece transform uygulandı.
    X_train_tfidf = vectorizer.fit_transform(x_train_text)
    X_test_tfidf  = vectorizer.transform(x_test_text)

    print(f"[DATA_HANDLER] Reuters hazırlığı tamamlandı.")
    
    return X_train_tfidf, X_test_tfidf, y_train_bin, y_test_bin