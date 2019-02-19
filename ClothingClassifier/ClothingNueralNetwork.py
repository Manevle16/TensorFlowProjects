from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

#load testing and training data in
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

#create list of categories corresponding with the numbers in the labels
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#Shows number of images in training set that has size of 28 by 28 pixels
print(train_images.shape)

#Shows number of lables
print(len(train_labels))

'''
#Create figure to plot something on
plt.figure()
#Show the matrix of pixels
plt.imshow(train_images[1])
#Add bar to show what values the color represents
plt.colorbar()
plt.grid(False)
#Show plot to user
plt.show()
'''

#Divide all pixel values by 255 to normalize value range between 0 and 1
train_images = train_images/255.0
test_images = test_images/255.0
'''
#Create figure of size 10 inches by 10 inches
plt.figure(figsize=(10,10))
for i in range(25):
    #Place in a 5 by 5 subplot iteratively
    plt.subplot(5,5,i+1)
    #Remove x and y tick marks from graphs
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    #Make pixel color black and white
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    #Label each subplot by the category it belongs in
    plt.xlabel(class_names[train_labels[i]])
plt.show()
'''
#Create nueral network model of layers
model = keras.Sequential([
    #Converts 2d array of pixels into long 1d array
    #Flattened layers are used for input
    keras.layers.Flatten(input_shape=(28,28)),
    #Dense layers are the densely-connected inner layers
    #relu = Rectified Linear Unit
    #just y = max(0, x)
    #Good for already standardized input
    keras.layers.Dense(128, activation=tf.nn.relu),
    #Good for digit classification more than two
    #Just finds the max and uses that as the answer
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

#optimizer: a procedure to optimize the effectiveness and efficiency of
#building the model
#Adam: a classical stochastic gradient descent procedure that efficiently,
#and effectively updates network weights based on the training datasets

#loss: The lower the closer the predictions are to the true labels
#sparse_categorical_crossentropy: Allows target to stay as integers to
#be catogorized

#metrics: what the model should be testing for and outputting

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#epochs: number of runs through training data
model.fit(train_images, train_labels, epochs=5)

#Test model on test data and output accuracy of it
test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)
print('Test loss:', test_loss)

#Outputs numpy of every images prediction array of values for each category
predictions = model.predict(test_images)
print(predictions[0])

def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    #Use blue label if correct and red label if incorrect
    #argmax: gets index with the largest value
    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = "blue"
    else:
        color = "red"

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    #Plot 10 bars using the predictions made
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

#Create subplot to plot multiple images with their associated bar graph
#for a better look of how the data performed on predicting
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions, test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions, test_labels)
plt.show()
