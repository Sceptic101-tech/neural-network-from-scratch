import numpy as np
import struct
from array import array
from os.path import join
from Perceptron import PerceptronClf

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

input_path = '/media/konstantin/143CD8253CD803A0/Files/Datasets/MNIST'
training_images_filepath = join(input_path, 'train-images.idx3-ubyte')
training_labels_filepath = join(input_path, 'train-labels.idx1-ubyte')
test_images_filepath = join(input_path, 't10k-images.idx3-ubyte')
test_labels_filepath = join(input_path, 't10k-labels.idx1-ubyte')

alpha = 0.02
train_len = 30000
hidden_size = 30
hidden_count = 2
num_inputs = 784
num_labels = 10
iterations = 150
batch_size = 40

#
# Load MINST dataset
#

mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
(x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()


X_train, y_train = (x_train[0:train_len].reshape(train_len, 28*28) / 255, y_train[0:train_len]) #vector 1000x784, vector 1000x1
one_hot_labels = np.zeros((len(y_train), 10))


for i,j in enumerate(y_train):
    one_hot_labels[i][j] = 1
#Y_train = one_hot_labels #матрица 10x10 с единственным значением единицы в строке. Далее с помощью функции argmax вытянем индекс максимального элемента. Это и будет цифра с картинки

X_test = x_test.reshape(len(x_test), 28*28) / 255
#y_test = np.zeros((len(y_test), 10))
#for i,j in enumerate(y_test):
    #test_labels[i][j] = 1

clf = PerceptronClf(alpha=alpha, train_len=train_len, hidden_size=hidden_size, hidden_count=hidden_count, num_inputs=num_inputs, num_labels=num_labels,\
                    iterations=iterations, batch_size=batch_size, activation='relu', record_freq=10)

clf.fit(X_train, y_train)
scores = clf.train_score()

y = clf.predict(X_test)
print(clf.score(y_test, y))
