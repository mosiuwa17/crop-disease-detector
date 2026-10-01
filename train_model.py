import tensorflow as tf
import os

# ========== MEMORY SETTINGS ==========
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# ========== CONFIGURATION ==========
TRAIN_DIR = r'C:\Users\motla\PycharmProjects\Crop_Disease_AI_Detector\plantvillage\train'
VAL_DIR = r'C:\Users\motla\PycharmProjects\Crop_Disease_AI_Detector\plantvillage\val'

IMG_SIZE = (224, 224)
BATCH_SIZE = 16          # Reduced from 32
EPOCHS = 5               # Reduced from 10 for faster test

# ========== 1. LOAD DATA ==========
print("Loading training data...")
train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

print("Loading validation data...")
val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names
print(f"Found {len(class_names)} classes.")

# ========== 2. OPTIMIZE DATA PIPELINE ==========
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.shuffle(100).prefetch(buffer_size=AUTOTUNE)   # Removed .cache()
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)                     # Removed .cache()

# ========== 3. DATA AUGMENTATION ==========
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal'),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
])

# ========== 4. BUILD THE MODEL ==========
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)

base_model.trainable = False

model = tf.keras.Sequential([
    data_augmentation,
    tf.keras.layers.Rescaling(1./127.5, offset=-1),
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(len(class_names), activation='softmax')
])

# ========== 5. COMPILE AND TRAIN ==========
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("Starting training...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

# ========== 6. SAVE THE MODEL ==========
model.save('crop_disease_model.h5')
print("✅ Model saved as crop_disease_model.h5")