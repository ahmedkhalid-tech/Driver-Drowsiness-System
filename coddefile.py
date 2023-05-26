import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Load the dataset
dataset_path = 'path_to_dataset_folder/drowsiness_dataset.csv'
data = pd.read_csv(dataset_path)

# Preprocess the data
labels = data['label']
features = data.drop('label', axis=1)

# Convert labels to numerical values
le = LabelEncoder()
labels = le.fit_transform(labels)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Normalize the input features
X_train = X_train / 255.0
X_test = X_test / 255.0

# Reshape the input features
X_train = X_train.values.reshape(-1, 24, 24, 1)
X_test = X_test.values.reshape(-1, 24, 24, 1)

# Convert the labels to categorical values
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
num_classes = y_test.shape[1]

# Create the model
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(24, 24, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32)

# Evaluate the model
_, test_acc = model.evaluate(X_test, y_test, verbose=0)
print("Test Accuracy: {:.2f}%".format(test_acc * 100))
