import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# keras is API for tensorflow

# data loading starts here

data = keras.datasets.fashion_mnist

# divide data into training and test data
# 80 - 90 % data for training, the rest will be for testing data

(train_images, train_labels), (test_images, test_labels) = data.load_data()

# this data has 10 labels

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress',
               'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# the value of output will be
# 0 for T-shirt/top
# 1 for Trouser
# 2 for Pullover
# and so on, hence we've got 10 nodes on the output

train_images = train_images / 255.0
test_images = test_images / 255.0


print(train_images[7])

plt.imshow(train_images[7], cmap=plt.cm.binary)
plt.show()

# data loading finished here

# our input data is 28 x 28 pixel (2 dimensional array)

# now we want to make it into 1 dimensional array and becomes 1 x 784, where 784 is 28 x 28

# thus, we will have 784 neurons as input

# hidden layer should be 15 - 20 % amounts of nodes of the input

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])

# it can be seen that we have three layers in building the model
# the first one is the input_shape that is 28 x 28
# the second one is the hidden layer with 128 nodes, with relu activation function
# the last one is our output that has softmax activation function with 10 nodes

# relu : rectify linear unit, a type of activation function
# the activation function will increase the complexity of model prediction
# n improve accuracy
# softmax = all the values in these neurons will add up to 1
# for example: 0.12 ankle boot, 0.28 pants, 0.59 trouser, n so on.

# now we are going to set up parameters for the models

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.fit(train_images, train_labels, epochs=5)

# epoch : how many times the model will see this information (the input)
# it gives the same images in different order 5 times to increase accuracy of the model
# we can play with epochs or tweak it to see the best accuracy

# evaluate model
test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Tested Acc: ', test_acc)
print('Tested Loss: ', test_loss)
