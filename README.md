# NN-Digits
A neural network which analyses a small sample of the MNIST handwritten digits using openCV-4.5.1. The sample size for the digits.png has a total of 5000 digits.

This neural network was created by Dr Andrew Watson; my university lecturer. It was given to the students to demonstrate how neural networks function. 

It was to show us how:
- different amounts of neurons in the hidden layers,
- adding more or less hidden layers,
- using different activation functions,
- updating the weights of each neuron in each hidden layer
- changing the training and testing ratios of the datasets

As a secondary objective we were tasked to find another dataset to test. For my task I chose to use the MNIST fashion dataset. It was a small sample of the dataset as the original dataset is large. The sample size for the fashion.png has a total of 900 images.

On lines 87 and 88, use comments to select digits.png or fashion.png

On lines 23 and 24, use comments to select the tile sizes to digits.png or fashion.png respective to the dataset being used. (digits = 20, fashion = 28)

The CLASS_N is 10 as there are 10 classes in each image.

- The program loads the whole image,
- Breaks the individual images up using a function named mosaic,
- Uses a shuffle function to shuffle those images into a random order,
- A deskew function is applied to correct any abnormalities, (This deskew function seems to work well with the digits. With the fashion, I suspect it is having adverse effects)
- Sets the training and testing ratios,
- Creates an image for the training images,
- Creates an image for the testing images,
- Converts the training array of images into 32bit floating points,
- Converts the testing array of images into 32bit floating points,
- Converts the output results to 32bit floating points with an output=1.0 for correct digits,
- Sets paramaters for the NN, (runtime, neurons in each hidden layer, layer size, topology of the network)
- Creates the NN, (applies the layer size, activation function, term criteria (when to terminate), training methods (backprop, update weights), training data,
- Training the network,
- Tests the network,
- Tests the trained network
- Checks for errors of each epoch and saves the lowest error achieved,
- When finished, the lowest error is printed to the console.
- The results are saved to a .xml file.

On lines 245-248, use comments to set the desired saved .xml file respective to the dataset being tested.
