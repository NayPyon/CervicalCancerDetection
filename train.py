import tensorflow as tf
import os
import sys
from tensorflow.keras.callbacks import EarlyStopping

# 1. Tentukan Variabel Utama
TRAIN_DIR = 'data/train'
VAL_DIR = 'data/validation'
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

try:
    NUM_CLASSES = len(os.listdir(TRAIN_DIR))
except FileNotFoundError:
    print(f"Error: Direktori data training '{TRAIN_DIR}' tidak ditemukan.")
    sys.exit()

if NUM_CLASSES == 0:
    print(f"Error: Tidak ada folder kelas di {TRAIN_DIR}. Folder ini kosong.")
    sys.exit()

print(f"--- Persiapan Dimulai ---")
print(f"Ditemukan {NUM_CLASSES} kelas di data training.")

# 2. Muat Data
print("--- Memuat Data Training & Validasi ---")
try:
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='categorical'
    )

    validation_dataset = tf.keras.utils.image_dataset_from_directory(
        VAL_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='categorical'
    )
except Exception as e:
    print(f"Error saat memuat data: {e}")
    sys.exit()

class_names = train_dataset.class_names
print(f"Nama Kelas Ditemukan (URUTAN INI PENTING): {class_names}")

train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

# 3. Bangun Arsitektur Model (Dengan Augmentasi)
print("--- Membangun Arsitektur Model (Dengan Augmentasi) ---")

base_model = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

# PENAMBAHAN LAPISAN AUGMENTASI DATA 
# Layer-layer ini hanya akan aktif saat .fit() (latihan),
# dan non-aktif saat .predict() (prediksi).
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2)
], name="augmentasi_data")

# PERUBAHAN ARSITEKTUR MODEL 
inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
# 1. Data masuk ke Augmentasi (0-255)
x = data_augmentation(inputs)
# 2. Data yang sudah di-augmentasi di-preprocess (-1 s/d 1)
x = tf.keras.applications.mobilenet_v2.preprocess_input(x) 
# 3. Data masuk ke model dasar
x = base_model(x, training=False) 

x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x) 
outputs = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')(x)

model = tf.keras.Model(inputs, outputs)

# 4. Compile Model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.summary() 

# 5. Latih Model
print("--- Memulai Pelatihan Model (Lebih Lama) ---")
EPOCHS = 20 # Bebas mau beraapa epochs, semakin tinggi semakin lama dan semakin baik (biasanya)

# "Penjaga Pintar" (Early Stopping)
# Ini akan memonitor 'val_loss' (seberapa baik model di data validasi)
# 'patience=3' berarti: "Tunggu 3 epoch. Jika val_loss tidak membaik
# (tidak turun) selama 3 epoch, BERHENTI LATIHAN."
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS,
    # Tambahkan 'callbacks' kita ke .fit()
    callbacks=[early_stop] 
)
print("--- Pelatihan Selesai ---")

# 6. Simpan Model
MODEL_NAME = 'cervical_cancer_model_v2.keras' 
model.save(MODEL_NAME)
print(f"Model telah disimpan sebagai '{MODEL_NAME}'")