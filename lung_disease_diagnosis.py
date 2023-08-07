import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Set seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Set data path
data_dir = '/content/drive/MyDrive/Disease/COVID19_Pneumonia_Normal_Chest_Xray_PA_Dataset'  # Path to the main dataset directory

# Set hyperparameters
batch_size = 64
epochs = 10  # Increase the number of epochs for more training
image_shape = (128, 128)  # Adjust the size as needed

# Data Augmentation and train-test split
data_gen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2)  # Split 20% of data for validation (test)

# Load and preprocess the data (both train and test)
train_generator = data_gen.flow_from_directory(
    data_dir,
    target_size=image_shape,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True,
    classes=['normal', 'pneumonia', 'covid'],
    subset='training')  # Use the training data (80% of the data)

validation_generator = data_gen.flow_from_directory(
    data_dir,
    target_size=image_shape,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False,
    classes=['normal', 'pneumonia', 'covid'],
    subset='validation')  # Use the validation data (20% of the data)

# Build the CNN model with transfer learning (using a larger model like VGG16)
base_model = tf.keras.applications.VGG16(
    input_shape=(image_shape[0], image_shape[1], 3),
    include_top=False,
    weights='imagenet')

base_model.trainable = False

model = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(3, activation='softmax')
])

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001),  # Use a smaller learning rate
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Implement learning rate schedule
def lr_schedule(epoch):
    if epoch < 10:
        return 0.0001
    else:
        return 0.00001

lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_schedule)

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=len(validation_generator),
    callbacks=[lr_scheduler],  # Use the learning rate scheduler
)

# Plot training and validation accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training and validation loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

model.save("Model.h5")
print("Model saved successfully!")