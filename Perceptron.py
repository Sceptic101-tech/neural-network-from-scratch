import numpy as np
import random

#Переделать вычисление ошибки(сейчас из числа вычитается массив в методе fit)

class _InternalPerceptronClf:

    def __init__(self, alpha, train_len, hidden_size, hidden_count, num_input, num_labels, iterations, batch_size, activation, record_freq):
        self.alpha = alpha
        self.train_len = train_len
        self.hidden_size = hidden_size
        self.hidden_count = hidden_count
        self.num_input = num_input
        self.num_labels = num_labels
        self.iterations = iterations
        self.batch_size = batch_size
        self.activation = activation
        self.record_freq = record_freq
        self.train_scores = []
        np.random.seed(42)
    
    #activation functions defining

    def relu(self, x):
        return (x >= 0) * x

    def relu_deriv(self, x):
        return x > 0

    def tanh(self, x):
        return np.tanh(x)

    def tanh_deriv(self, x):
        return 1 - x**2

    def softmax(self, x):
        temp = np.exp(x)
        return temp / np.sum(temp, axis=1, keepdims=True)
    

    def _init_network(self):
        np.random.seed(42)
        layers = [0] * (self.hidden_count + 2)
        weights = [0] * (len(layers) - 1)
        layers_delta = [0] * (self.hidden_count + 1)
        dropout_masks = [0] * self.hidden_count

        weights[0] = 0.002*np.random.random((self.num_input, self.hidden_size)) - 0.001 #input layer
        for i in range(0, self.hidden_count - 1):
            weights[i+1] = 0.02*np.random.random((self.hidden_size, self.hidden_size)) - 0.01 #hidden layers
        weights[-1] = 0.02*np.random.random((self.hidden_size, self.num_labels)) - 0.01 #output layer
        return layers, weights, layers_delta, dropout_masks

    def _backpropogation(self, batch_start, batch_end, y):
        self.layers_delta[-1] = (self.layers[-1] - y[batch_start:batch_end]) / self.batch_size
        for h in range(2, self.hidden_count + 2):
            self.layers_delta[-h] = (np.dot(self.layers_delta[-h+1], self.weights[-h+1].T)) * self.dropout_masks[-h+1] * self.relu_deriv(self.layers[-h])
    
    def _weights_adjust(self):
        for w in range(len(self.layers) - 1):
            self.weights[w] -= self.alpha * (np.dot(self.layers[w].T, self.layers_delta[w]))

    def _fit(self, X, y):
        self.layers, self.weights, self.layers_delta, self.dropout_masks = self._init_network()
        for j in range(self.iterations):
            error, correct_count = (0.0, 0)
            for i in range(int(len(X) / self.batch_size)):
                batch_start, batch_end = ((i*self.batch_size), ((i+1)*self.batch_size))
                self.layers[0] = X[batch_start:batch_end]
                for h in range(1, len(self.layers) - 1):
                    self.layers[h] = self.relu(np.dot(self.layers[h-1], self.weights[h-1]))
                    self.dropout_masks[h-1] = np.random.randint(2, size=self.layers[h].shape)
                    self.layers[h] *= self.dropout_masks[h-1] * 2 #умножение на 2 для усиления сигнала(умножение на обратную величину вероятности отключения p нейронов в слое)
                self.layers[-1] = self.softmax(np.dot(self.layers[-2], self.weights[-1]))

                error -= np.sum(y[batch_start:batch_end] * np.log(self.layers[-1] + 1e-10))

                for k in range(self.batch_size):
                    correct_count += int(np.argmax(self.layers[-1][k:k+1]) == np.argmax(y[batch_start+k:batch_start+k+1]))

                self._backpropogation(batch_start, batch_end, y)
            
                self._weights_adjust()

            if j % self.record_freq == 0:
                self.train_scores.append((j, error/len(X), correct_count/len(X)))
                print(f'iteration {j} error {str(error / len(X))[:6]} correct_rate {str(correct_count/len(X))[:6]}')


    def _predict(self, X, is_proba=False):
        self.pred = np.zeros(len(X))
        for i in range(len(X)):
            self.layers[0] = X[i:i+1]
            for h in range(1, len(self.layers) - 1):
                self.layers[h] = self.relu(np.dot(self.layers[h-1], self.weights[h-1]*0.5))# Делим на 2, поскольку выкидывали нейроны при обучении. Компенсируем этот момент
            self.layers[-1] = self.softmax(np.dot(self.layers[-2], self.weights[-1]*0.5))
            if not is_proba:
                self.pred[i] = np.argmax(self.layers[-1])
            else:
                self.pred[i] = max(self.layers[-1])
        return self.pred
    
    def _score(self, label, pred):
        correct_count = 0
        for i in range(len(label)):
            correct_count += (label[i:i+1] == pred[i:i+1])
        return correct_count/len(label)

class PerceptronClf:
    def __init__(self, alpha=0.05, train_len=20000, hidden_size=10, hidden_count=1, num_inputs=0, num_labels=0,\
                 iterations=200, batch_size=1, activation='relu', record_freq=10):
        self.num_labels = num_labels
        self.__internal_perceptron = _InternalPerceptronClf(alpha=alpha, train_len=train_len, hidden_size=hidden_size,\
                                                           hidden_count=hidden_count, num_input=num_inputs, num_labels=num_labels,\
                                                            iterations=iterations, batch_size=batch_size, activation=activation, record_freq=record_freq)
    
    def print_hi(self):
        print('Hi')

    def fit(self, X, y):
        print('fit called')
        self.trans_y = np.zeros((len(X), self.num_labels))
        for ind, num in enumerate(y):
            self.trans_y[ind][num] = 1
        self.__internal_perceptron._fit(X,self.trans_y)

    def predict(self, X):
        print('return ndarray y')
        return self.__internal_perceptron._predict(X, is_proba=False)
    
    def get_params(self):
        print('imagine here is dict of model params')

    def preditc_proba(self, X):
        print('return ndarray y')
        return self.__internal_perceptron._predict(X, is_proba=True)
    
    def train_score(self):
        return self.__internal_perceptron.train_scores
    
    def score(self, label, pred):
        print('return float score of the model')
        return self.__internal_perceptron._score(label, pred)
    
__all__ = ['PerceptronClf']