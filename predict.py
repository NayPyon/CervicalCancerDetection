import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.keras.utils import load_img, img_to_array
import sys

# 1. Menentukan Variabel Utama
IMG_SIZE = (224, 224)
MODEL_PATH = 'cervical_cancer_model_v2.keras' 

CLASS_NAMES = ['HSIL', 'LSIL', 'NORMAL'] 
print(f"--- Program Prediksi Kanker Serviks ---")
print(f" Kelas telah dikonfigurasi (0, 1, 2): {CLASS_NAMES}")

# 2. Muat model yang sudah dilatih
try:
    # Kita muat model .keras (yang sudah berisi augmentasi & preprocessing)
    model = tf.keras.models.load_model(MODEL_PATH) 
    print(f" Model berhasil dimuat dari {MODEL_PATH}")
except Exception as e:
    print(f"Error: Tidak dapat memuat model '{MODEL_PATH}'.")
    print(f"Pastikan Anda sudah menjalankan 'train.py' (vUpgrade).")
    print(f"Detail error: {e}")
    sys.exit()

# 3. Fungsi memproses gambar
def proses_gambar(image_path): 
    """Muat dan proses gambar untuk prediksi."""
    try:
        img = load_img(image_path, target_size=IMG_SIZE)
        img_array = img_to_array(img)
        img_batch = np.expand_dims(img_array, axis=0)
        
        original_img_display = mpimg.imread(image_path)
        
        return img_batch, original_img_display
    except Exception as e:
        print(f"Error saat memproses gambar '{image_path}': {e}")
        return None, None
    
# 4. Fungsi prediksi
def prediksi_dan_tampilkan(image_path): 
    processed_img_batch, display_img = proses_gambar(image_path) 
    if processed_img_batch is None:
        return
    
    prediksi = model.predict(processed_img_batch)
    prediksi_scores = prediksi[0] 
    prediksi_indeks = np.argmax(prediksi_scores) 
    hasil = CLASS_NAMES[prediksi_indeks]
    confidence = np.max(prediksi_scores) * 100

    print(f"--- Hasil Prediksi ---")
    print(f"Skor mentah (softmax): {prediksi_scores}")
    print(f"Indeks Prediksi: {prediksi_indeks}, Kelas: {hasil}")

    plt.figure(figsize=(6,7))
    plt.imshow(display_img)
    title_text = f"Prediksi: {hasil}\nKepercayaan: {confidence:.2f}%"
    plt.title(title_text) 
    plt.axis('off')
    plt.show()

# 5. Jalankan Program
if __name__ == "__main__": 
    try:
        path_input = input("Masukkan path gambar tes anda (e.g., tes.bmp):")
        prediksi_dan_tampilkan(path_input) 
    except KeyboardInterrupt:
        print("\nProgram dihentikan oleh pengguna.")