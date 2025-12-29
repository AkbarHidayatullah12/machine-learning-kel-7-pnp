import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

base_dir = 'dataset'
datagen = ImageDataGenerator(rescale=1./255)
generator = datagen.flow_from_directory(
    base_dir,
    class_mode='binary',
    batch_size=1,
    shuffle=False
)

print("\n--- CLASS INDICES ---")
print(generator.class_indices)
print("---------------------")
