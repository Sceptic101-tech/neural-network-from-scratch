import sys, time
import numpy as np
import struct
from array import array
from os.path import join
import random
#import matplotlib.pyplot as plt

#
# MNIST Data Loader Class
#
class MnistDataloader(object):
    def __init__(self, training_images_filepath, training_labels_filepath,
                 test_images_filepath, test_labels_filepath):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath

    def read_images_labels(self, images_filepath, labels_filepath):
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())

        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i][:] = img

        return images, labels

    def load_data(self):
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        return (np.array(x_train), np.array(y_train)), (np.array(x_test), np.array(y_test))



input_path = 'F:/py_proj/projects/MNIST_classification/'
training_images_filepath = join(input_path, 'train-images.idx3-ubyte')
training_labels_filepath = join(input_path, 'train-labels.idx1-ubyte')
test_images_filepath = join(input_path, 't10k-images.idx3-ubyte')
test_labels_filepath = join(input_path, 't10k-labels.idx1-ubyte')

#
# Load MINST dataset
#
mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
(x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

#function defining

def relu(x):
    return (x >= 0) * x

def relu_deriv(x):
    return x > 0

def tanh(x):
    return np.tanh(x)

def tanh_deriv(x):
    return 1 - x**2

def softmax(x):
    temp = np.exp(x)
    return temp / np.sum(temp, axis=1, keepdims=True)


alpha = 0.05
train_len = 20000
hidden_size = 25
hidden_count = 2
pixels_per_image = 784
num_labels = 10
iterations = 300
batch_size = 30

layers = [0] * (hidden_count + 2)
weights = [0] * (len(layers) - 1)
layers_delta = [0] * (hidden_count + 1)
dropout_masks = [0] * hidden_count


images, labels = (x_train[0:train_len].reshape(train_len, 28*28) / 255, y_train[0:train_len]) #vector 1000x784, vector 1000x1
one_hot_labels = np.zeros((len(labels), 10))


for i,j in enumerate(labels):
    one_hot_labels[i][j] = 1
labels = one_hot_labels #матрица 10x10 с единственным значением единицы в строке. Далее с помощью функции argmax вытянем индекс максимального элемента. Это и будет цифра с картинки

test_images = x_test.reshape(len(x_test), 28*28) / 255
test_labels = np.zeros((len(y_test), 10))
for i,j in enumerate(y_test):
    test_labels[i][j] = 1


np.random.seed(42)
weights[0] = 0.002*np.random.random((pixels_per_image, hidden_size)) - 0.001 #input layer
for i in range(0, hidden_count - 1):
    weights[i+1] = 0.02*np.random.random((hidden_size, hidden_size)) - 0.01 #hidden layers
weights[-1] = 0.02*np.random.random((hidden_size, num_labels)) - 0.01 #output layer


min_error, max_correct = (100.0, 0)
for j in range(iterations):
    error, correct_count = (0.0, 0)
    for i in range(int(len(images) / batch_size)):
        batch_start, batch_end = ((i*batch_size), ((i+1)*batch_size))
        layers[0] = images[batch_start:batch_end]
        for h in range(1, len(layers) - 1):
            layers[h] = relu(np.dot(layers[h-1], weights[h-1]))
            dropout_masks[h-1] = np.random.randint(2, size=layers[h].shape)
            layers[h] *= dropout_masks[h-1] * 2 #умножение на 2 для усиления сигнала(умножение на обратную величину вероятности отключения p нейронов в слое)
        layers[-1] = softmax(np.dot(layers[-2], weights[-1]))

        error += np.sum((labels[batch_start:batch_end] - layers[-1])**2)

        for k in range(batch_size):
            correct_count += int(np.argmax(layers[-1][k:k+1]) == np.argmax(labels[batch_start+k:batch_start+k+1]))

    #Backpropogation
        layers_delta[-1] = (layers[-1] - labels[batch_start:batch_end]) / batch_size #vector 1x10
        for h in range(2, hidden_count + 2):
            layers_delta[-h] = (np.dot(layers_delta[-h+1], weights[-h+1].T)) * dropout_masks[-h+1] * relu_deriv(layers[-h])

    #weights adjustments
        for w in range(len(layers) - 1):
            weights[w] -= alpha * (np.dot(layers[w].T, layers_delta[w]))

#evaluating accuracy
    if j % 10 == 0:
        print(f'iteration {j+1} error {str(error / len(images))[:6]} correct_percent {str(correct_count/train_len * 100)[:8]}%')

        error_test, correct_count_test = (0.0, 0)
        for i in range(len(x_test)):
            layers[0] = test_images[i:i+1]
            #h for hidden layers
            for h in range(1, len(layers) - 1):
                layers[h] = relu(np.dot(layers[h-1], weights[h-1]*0.5))# Делим на 2, поскольку выкидывали нейроны при обучении. Компенсируем этот момент

            layers[-1] = softmax(np.dot(layers[-2], weights[-1]*0.5))

            error_test += np.sum((test_labels[i:i+1] - layers[-1])**2)
            correct_count_test += int(np.argmax(test_labels[i:i+1]) == np.argmax(layers[-1]))
            min_error = min(min_error, error_test)
            max_correct = max(correct_count_test, max_correct)
        print(f"average test error {str(error_test / len(test_images))[:10]}, test correct percent {str(correct_count_test/len(test_images) * 100)[:8]}")
print('-'*40)
print(f"min_error = {min_error}, max_correct = {max_correct/len(test_images) * 100}")

