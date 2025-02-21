import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import glob
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Dropout, Dense, Conv2D, BatchNormalization, Activation
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.python.client import device_lib
from sklearn.model_selection import train_test_split

def plot_accuracy(history):
    """
    Plots training and validation accuracy.
    """
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig('/home/medanomaly/ResNET/Result/accuracy_plot_owen.png')

def plot_loss(history):
    """
    Plots training and validation loss.
    """
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('/home/medanomaly/ResNET/Result/loss_plot_owen.png')


print(device_lib.list_local_devices())
if tf.config.list_physical_devices('GPU'):
    print("GPU is available and will be used")
else:
    print("Running on CPU")

# Define categories based on file names
categories = ["glioma", "meningioma", "metastasis"]

def get_label_from_filename(filename):
    """
    Given a filename, this function returns the corresponding label based on file name patterns.
    """
    if 'GLI' in filename:
        return 0  # glioma
    elif 'MEN' in filename:
        return 1  # meningioma
    elif 'MET' in filename:
        return 2  # metastasis
    elif 'GOO' in filename: 
        return 'good'
    else:
        return None  # unknown class

# Define directories for training, validation, and testing
data_dir = '/home/medanomaly/ResNET/anomaly_mask/'

# Gather all image file paths
all_files = glob.glob(os.path.join(data_dir, '*.png'))  # Assuming files are PNGs; change if different

# Split the dataset into training and validation
train_files, other_files = train_test_split(all_files, test_size=0.2, random_state=42)
test_files, valid_files = train_test_split(other_files, test_size=0.5, random_state=42)
print("Training data amount: ", len(train_files))
print("Validation data amount: ", len(valid_files))
print("Testing data amount: ", len(test_files))


batch_size = 32
target_size = (369, 369, 3)

# Extract labels for training and validation sets
train_labels = [get_label_from_filename(f) for f in train_files]
valid_labels = [get_label_from_filename(f) for f in valid_files]
test_labels = [get_label_from_filename(f) for f in test_files]

# Create custom data generators for training, validation, and testing
# train_generator = custom_data_generator(train_files, train_labels, batch_size=batch_size, target_size=target_size)
# valid_generator = custom_data_generator(valid_files, valid_labels, batch_size=batch_size, target_size=target_size)
# test_generator = custom_data_generator(test_files, test_labels, batch_size=batch_size, target_size=target_size, shuffle = False)

# Convert file paths and labels into DataFrames for ImageDataGenerator
train_df = pd.DataFrame({'filename': train_files, 'label': train_labels})
valid_df = pd.DataFrame({'filename': valid_files, 'label': valid_labels})
test_df = pd.DataFrame({'filename': test_files, 'label': test_labels})

# Convert labels to strings for compatibility with sparse mode
train_df['label'] = train_df['label'].astype(str)
valid_df['label'] = valid_df['label'].astype(str)
test_df['label'] = test_df['label'].astype(str)

# Create an ImageDataGenerator with augmentation for training
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

# For validation and testing, use only preprocessing (no augmentation)
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col='filename',
    y_col='label',
    target_size=target_size[:2],
    batch_size=batch_size,
    class_mode='sparse',  # Use 'sparse' because your labels are integers
    shuffle=True  # Shuffle for training
)

# Set up the validation generator (no augmentation)
valid_generator = test_datagen.flow_from_dataframe(
    dataframe=valid_df,
    x_col='filename',
    y_col='label',
    target_size=target_size[:2],
    batch_size=batch_size,
    class_mode='sparse',
    shuffle=False  # No shuffling for validation
)

# Set up the test generator (no augmentation, no shuffling)
test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    x_col='filename',
    y_col='label',
    target_size=target_size[:2],
    batch_size=batch_size,
    class_mode='sparse',
    shuffle=False  # No shuffling for testing
)


# Calculate class weights
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
class_weights = dict(enumerate(class_weights))

# Load the pre-trained ResNet50 model without the top layer
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(369, 369, 3))

# Freeze the base model layers
for layer in base_model.layers:
    layer.trainable = False

num_layers_to_unfreeze = 10
# Unfreeze the last 'num_layers_to_unfreeze' layers
for layer in base_model.layers[-num_layers_to_unfreeze:]:
    layer.trainable = True

# Build the model
# model = Sequential([
#     base_model,
#     Conv2D(64, (3, 3), activation='swish', kernel_regularizer=l2(0.01)),
#     BatchNormalization(),
#     Activation('swish'),
#     GlobalAveragePooling2D(),
#     Dropout(0.1),  # Dropout to prevent overfitting
#     Dense(128, activation='swish', kernel_regularizer=l2(0.01)),  # Using Swish activation
#     Dropout(0.3),
#     Dense(64, activation='swish', kernel_regularizer=l2(0.01)),
#     Dense(3, activation='softmax', kernel_regularizer=l2(0.01))  
# ])

model = Sequential([
    base_model,
    Conv2D(64, (3, 3), activation='swish', kernel_regularizer=l2(0.02)),
    BatchNormalization(),
    Activation('swish'),
    GlobalAveragePooling2D(),
    Dropout(0.2),  # Adjusted dropout to prevent overfitting
    Dense(128, kernel_regularizer=l2(0.02)),  # Using increased regularization
    BatchNormalization(),
    Activation('swish'),
    Dropout(0.3),
    Dense(64, kernel_regularizer=l2(0.01)),
    BatchNormalization(),
    Activation('swish'),
    Dense(3, activation='softmax', kernel_regularizer=l2(0.01))
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Define a learning rate scheduler
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

# Count the number of images for calculating steps_per_epoch and validation_steps
train_image_count = len(train_files)  # Number of training images
valid_image_count = len(valid_files)  # Number of validation images
# test_image_count = len(test_files)

steps_per_epoch = train_image_count // batch_size
validation_steps = valid_image_count // batch_size
# validation_steps = test_image_count // batch_size

# Adjust the model fit function to include class weights
# history = model.fit(
#     train_generator,
#     steps_per_epoch=steps_per_epoch,
#     epochs=5,
#     validation_data=valid_generator,
#     validation_steps=validation_steps,  # Corrected validation_steps
#     class_weight=class_weights  # Pass the computed class weights
# )

history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=20,
    validation_data=valid_generator,
    validation_steps=len(valid_generator),
    class_weight=class_weights,  # Use the computed class weights
    callbacks=[lr_scheduler]
)

# Save the model
model.save('brain_tumor_resnet50_custom_labels.keras')


def evaluate_test_set(model, test_generator, test_labels):
    # Calculate steps to cover the entire test dataset
    steps = len(test_labels) // batch_size + (1 if len(test_labels) % batch_size != 0 else 0)
    
    # Evaluate the model on the test set
    test_loss, test_acc = model.evaluate(test_generator, steps=steps)
    print(f'Test accuracy (from evaluate): {test_acc}, Test loss: {test_loss}')
    
    # Make predictions on the test dataset
    predictions = model.predict(test_generator, steps=steps, workers=0)

    # Determine predicted classes
    if predictions.shape[1] == 1:  # Binary classification
        predicted_classes = [1 if x[0] >= 0.5 else 0 for x in predictions]
    else:  # Multi-class
        predicted_classes = np.argmax(predictions, axis=1)

    # Check alignment of predictions and labels
    if len(test_labels) != len(predicted_classes):
        print(f"Warning: Length mismatch (test_labels: {len(test_labels)}, predictions: {len(predicted_classes)})")
        return

    # Compute manual accuracy
    manual_accuracy = np.sum(np.array(test_labels) == np.array(predicted_classes)) / len(test_labels)
    print(f'Manual accuracy (from comparison): {manual_accuracy}')

    # Generate classification report
    print(classification_report(test_labels, predicted_classes, target_names=categories))
    
    # Create and display the confusion matrix
    cm = confusion_matrix(test_labels, predicted_classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=categories)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix (Test Set)')
    plt.savefig('/home/medanomaly/ResNET/Result/test_evaluation_owen.png')


# Plot accuracy
plot_accuracy(history)

# Plot loss
plot_loss(history)

test_image_count = len(test_files)
# steps = np.ceil(test_image_count / batch_size).astype(int)
# steps = test_image_count // batch_size

# Test the model on the test set
# evaluate_test_set(model, test_generator, test_labels, steps=steps)
evaluate_test_set(model, test_generator, test_labels)