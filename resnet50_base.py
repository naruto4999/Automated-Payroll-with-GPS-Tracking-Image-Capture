import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from keras_vggface.vggface import VGGFace
import mtcnn
import tensorflow as tf



# Define the path to your test dataset directory
test_dataset_directory = "/run/media/naruto/disk_e/Kaushal/machine_learning/dataset/transfer_learning"

# Define the target image size
image_size = (224, 224)

# Initialize lists to store images and labels
images = []
labels = []

# Create a dictionary to map labels to integers
label_to_int = {}
next_label_int = 0

# Create a reverse mapping to store label names as strings
int_to_label = {}

# Loop through the subdirectories in the test dataset directory
for label in os.listdir(test_dataset_directory):
    label_directory = os.path.join(test_dataset_directory, label)
    if os.path.isdir(label_directory):
        for filename in os.listdir(label_directory):
            if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
                image_path = os.path.join(label_directory, filename)
                image = Image.open(image_path)
                
                # Resize the image to the specified size
                image = image.resize(image_size)
                
                # Convert the image to a NumPy array
                image = np.array(image)
                images.append(image)
                
                # Map the label to an integer (0-based)
                if label not in label_to_int:
                    label_to_int[label] = next_label_int
                    int_to_label[next_label_int] = label
                    next_label_int += 1
                labels.append(label_to_int[label])


X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Convert the lists to NumPy arrays
X_train = np.array(X_train)
y_train = np.array(y_train)

#Pre-trained Model
vggface_resnet_base = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3))

#Data Augmentation
data_augmentation = tf.keras.Sequential([tf.keras.layers.RandomFlip('horizontal'), tf.keras.layers.RandomRotation(0.2),])

nb_class = 15 # Number of new people + 1 for unknown/Invalid

# Freeze the base model
vggface_resnet_base.trainable = False
last_layer = vggface_resnet_base.get_layer('avg_pool').output

# Build up the new model
inputs = tf.keras.Input(shape=(224, 224, 3))
x = data_augmentation(inputs)
x = vggface_resnet_base(x)
x = tf.keras.layers.Flatten(name='flatten')(x)
out = tf.keras.layers.Dense(nb_class, name='classifier', activation='sigmoid')(x)
custom_vgg_model = tf.keras.Model(inputs, out)

#Summary
custom_vgg_model.summary()

#Training
base_learning_rate = 0.001
custom_vgg_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
history = custom_vgg_model.fit(X_train, y_train, epochs=20)

# Convert the lists to NumPy arrays
X_test = np.array(X_test)
y_test = np.array(y_test)

custom_vgg_model.evaluate(X_test, y_test)