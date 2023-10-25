# Cifar10 CNN Classifier

This project is an implementation of a Convolutional Neural Network (CNN) for image classification using the Cifar10 dataset. The goal is to train a model that can accurately classify 10 different classes of images.

## Dataset

The data used in this project is the Cifar10 dataset, which consists of 60,000 images in the training set and 10,000 images in the test set. The images are of size 32x32 pixels and are divided into 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck. All images are normalized to the range of 0 to 1.

## Model Architecture

The model used for image classification consists of three convolutional layers, followed by normalization, max-pooling, and dropout layers. After the convolutional layers, the data is flattened into a vector and goes through two dense layers with ReLU activation, where the last layer uses softmax activation to output probabilities for each class.

## Training

The model is trained using the Adam optimizer and the "sparse categorical crossentropy" loss function. Training is performed for 50 epochs, with a batch size of 32. Additionally, data augmentation techniques are applied, including random horizontal flipping and random shifting along the x and y axes.

## Results

After 50 epochs, the model achieves an accuracy of around 93% on the training set and around 89% on the test set. Loss and accuracy curves are plotted for both the training and validation sets. Furthermore, a confusion matrix is displayed to show how the model classified each class.
 
