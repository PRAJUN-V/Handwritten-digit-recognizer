import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input, BatchNormalization
from keras import backend as K

# Load the MNIST data, split between training and testing datasets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(f"Training data shape: {x_train.shape}, Labels shape: {y_train.shape}")

# Reshape data to fit the model: (samples, 28, 28, 1) to represent grayscale images
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)

# Convert labels to one-hot encoded format
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# Convert data to float32 and normalize to the range [0, 1]
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Set training parameters
batch_size = 128
num_classes = 10
epochs = 20  # Increased epochs

# Define the Sequential CNN model
model = Sequential()
model.add(Input(shape=input_shape))

# Convolutional Layers with Batch Normalization
model.add(Conv2D(32, kernel_size=(5, 5), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten the convolutional feature maps into a 1D vector
model.add(Flatten())

# Fully connected layers (dense layers)
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))  # Increased dropout to prevent overfitting

model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))

# Output layer (10 classes for the digits 0-9)
model.add(Dense(num_classes, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',  # Changed optimizer
              metrics=['accuracy'])

# Train the model
hist = model.fit(x_train, y_train,
                 batch_size=batch_size,
                 epochs=epochs,
                 verbose=1,
                 validation_data=(x_test, y_test))

print("The model has successfully trained")

# Evaluate the model on the test dataset
score = model.evaluate(x_test, y_test, verbose=0)
print(f'Test loss: {score[0]}')
print(f'Test accuracy: {score[1]}')

# Save the model to disk
model.save('mnist.h5')
print("Saving the model as mnist.h5")
