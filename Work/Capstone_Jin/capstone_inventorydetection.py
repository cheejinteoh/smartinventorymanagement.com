import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import matplotlib.pyplot as plt

# Paths
base_path = os.path.dirname(__file__)
dataset_path = os.path.join(base_path, 'dataset')
test_images_path = os.path.join(base_path, 'test_images')

# Verify paths
if not os.path.exists(dataset_path):
    print(f"Error: Dataset path not found: {dataset_path}")
    exit()
if not os.path.exists(test_images_path):
    print(f"Error: Test images path not found: {test_images_path}")
    exit()

# Data augmentation and preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # Split dataset into 80% training and 20% validation
)

# Training data generator
train_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',  # Binary classification
    subset='training'
)

# Validation data generator
validation_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

# Debugging the data structure
for data, labels in train_generator:
    print(f"Training Data shape: {data.shape}")
    print(f"Training Labels shape: {labels.shape}")
    break

# Initialize the MobileNetV2 model without pre-trained weights
base_model = MobileNetV2(weights=None, include_top=False, input_shape=(224, 224, 3))

# Add custom layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)  # Dropout to prevent overfitting
predictions = Dense(1, activation='sigmoid')(x)  # Binary classification

# Create the final model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer=Adam(learning_rate=1e-5), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=20  # Adjust epochs as needed
)

# Save the trained model
model.save('defect_detection_model.h5')

# Plot training metrics
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Model Accuracy')
plt.show()

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Model Loss')
plt.show()

# Function to predict and show an image
def predict_and_show_image(img_path, model, threshold=0.5):
    """Predict and display the image with its prediction."""
    try:
        # Load and preprocess the image
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        # Make prediction
        prediction = model.predict(img_array)[0][0]
        label = "Good" if prediction >= threshold else "Defective"

        # Display the image with its prediction
        plt.figure()
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"Prediction: {label}")
        plt.show()

        # Print prediction result
        print(f"Image: {img_path} - Prediction: {label}")
    except Exception as e:
        print(f"Error processing {img_path}: {e}")

# Function to predict and show all images in a folder
def predict_and_show_images_in_folder(folder_path, model, threshold=0.5):
    """Predict and display all images in a folder with their predictions."""
    print(f"Predicting results for images in folder: {folder_path}")
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        if os.path.isfile(img_path):
            predict_and_show_image(img_path, model, threshold)

# Load the trained model
model = tf.keras.models.load_model('defect_detection_model.h5')

# Test on the 'test_images' folder and display results
predict_and_show_images_in_folder(test_images_path, model)
