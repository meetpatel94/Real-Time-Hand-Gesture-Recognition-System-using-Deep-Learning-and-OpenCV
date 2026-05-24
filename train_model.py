import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPool2D,
    Flatten,
    Dense,
    Dropout
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import matplotlib.pyplot as plt

train_path = "dataset/train"

# ==========================
# DATA PREPROCESSING
# ==========================

datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    zoom_range=0.15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    validation_split=0.2
)

# Train data
train_batches = datagen.flow_from_directory(
    train_path,
    target_size=(64,64),
    batch_size=32,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

# Validation data
val_batches = datagen.flow_from_directory(
    train_path,
    target_size=(64,64),
    batch_size=32,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

num_classes = len(train_batches.class_indices)

print("\nDetected Classes:")
print(train_batches.class_indices)

# ==========================
# MODEL
# ==========================

model = Sequential([

    Input(shape=(64,64,3)),

    Conv2D(32,(3,3),activation='relu'),
    MaxPool2D(2,2),

    Conv2D(64,(3,3),activation='relu'),
    MaxPool2D(2,2),

    Conv2D(128,(3,3),activation='relu'),
    MaxPool2D(2,2),

    Flatten(),

    Dense(256,activation='relu'),

    Dropout(0.4),

    Dense(num_classes,activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=2
)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

history = model.fit(
    train_batches,
    epochs=15,
    validation_data=val_batches,
    callbacks=[reduce_lr,early_stop]
)

model.save("best_model.h5")

print("\n✅ Model saved successfully")

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

plt.legend(["Train","Validation"])
plt.show()