# Improved Handwritten Digit Recognition with CNN

This project implements an improved convolutional neural network (CNN) for recognizing handwritten digits from the MNIST dataset. The model is designed to improve accuracy and reduce overfitting through techniques like data augmentation and dropout layers.

## Requirements

To run this project, you will need the following libraries:

- `tensorflow` (includes Keras)
- `numpy`
- `matplotlib`

You can install the required dependencies using pip

pip install tensorflow numpy matplotlib
Dataset
The MNIST dataset is used for training and testing the model. It contains 60,000 training images and 10,000 test images of handwritten digits (0-9). The dataset is loaded directly using TensorFlow's keras.datasets.mnist.

Model Overview
The model architecture consists of the following layers:

Convolutional Layers: Three convolutional layers with increasing filter sizes (32, 64, and 128) followed by batch normalization and max pooling.
Dropout Layer: A dropout layer with a rate of 0.5 to prevent overfitting.
Fully Connected Layers: A dense layer with 128 units and a softmax output layer to classify the digits.
Data Augmentation
The model utilizes data augmentation to improve generalization. The following transformations are applied to the training data:

Random rotations (up to 10 degrees)
Random horizontal and vertical shifts (up to 10%)
Random zoom (up to 10%)
Training
The model is trained using the Adam optimizer and sparse categorical crossentropy loss for 20 epochs. The augmented data is fed to the model through the ImageDataGenerator class.

Results
After training, the model is evaluated on the test dataset and the accuracy is printed.

Saving the Model
The trained model is saved to a file named improved_cnn_digit_recognition.h5.

Visualizing Predictions
The model makes predictions on the test dataset. A test image is displayed along with the predicted label for visual verification.

How to Use
To use the model for predictions, load the trained model and provide an input image. The model will predict the digit, and you can visualize it using matplotlib.

from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('improved_cnn_digit_recognition.h5')

# Make a prediction
prediction = model.predict(x_test[0:1])
print(f"Predicted Label: {np.argmax(prediction)}")
