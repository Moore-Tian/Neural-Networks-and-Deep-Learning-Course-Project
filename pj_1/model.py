import numpy as np
import sys
import pickle
from tqdm import tqdm
from utils import *


class back_propagation():
    def __init__(self, layers, dropout=0, classifacation=False,
                 activate='leakyrelu'):
        self.layers = layers
        self.num_layers = len(layers)
        self.dropout = dropout
        self.classifacation = classifacation
        if self.classifacation is True:
            self.loss = 'cross_entropy'
        else:
            self.loss = 'mean_squared_error'
        self.activate = activate
        self.Weights = []
        self.Biases = []
        self.gradients_W = []
        self.gradients_b = []
        self.momentum_W = []
        self.momentum_b = []
        self.initialize_weights_and_biases()

    def initialize_weights_and_biases(self):
        for i in range(self.num_layers - 1):
            Weight = np.random.randn(self.layers[i+1], self.layers[i])*(np.sqrt(2/(self.layers[i+1] + self.layers[i])))
            Bias = np.zeros((self.layers[i+1], 1))
            weight_momentum = np.zeros((self.layers[i+1], self.layers[i]))
            bias_momentum = np.zeros((self.layers[i+1], 1))
            self.Weights.append(Weight)
            self.Biases.append(Bias)
            self.momentum_W.append(weight_momentum)
            self.momentum_b.append(bias_momentum)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def leakyrelu(self, x):
        return np.where(x > 0, x, x * 0.001)

    def softmax(self, X):
        X_exp = np.exp(X - np.max(X, axis=1, keepdims=True))
        row_sums = np.sum(X_exp, axis=1, keepdims=True)
        softmax = X_exp / row_sums
        return softmax

    # 用于将对某一层的输出向量随机置零
    def Dropout(self, X, iftest=0):
        if iftest == 1:
            return X
        else:
            return X * np.random.binomial(1, 1-self.dropout, size=X.shape)

    def forward_propogation(self, X, iftest=0):
        activations = [X]
        if self.activate == 'sigmoid':
            for i in range(self.num_layers - 2):
                activations.append(self.sigmoid(self.Dropout((np.dot(self.Weights[i], activations[i].T) + self.Biases[i]).T, iftest=iftest)))
        elif self.activate == 'leakyrelu':
            for i in range(self.num_layers - 2):
                activations.append(self.leakyrelu(self.Dropout((np.dot(self.Weights[i], activations[i].T) + self.Biases[i]).T, iftest=iftest)))

        activations.append((np.dot(self.Weights[-1], activations[-1].T) + self.Biases[-1]).T)

        if self.classifacation is True:
            activations[-1] = self.softmax(activations[-1])

        self.activations = activations
        return activations[-1]

    def loss_compute(self, y_true, y_pred):
        l2_loss = 0
        if self.l2_regularization != 0:
            for i in range(self.num_layers - 1):
                l2_loss += np.sum(np.square(self.Weights[i]))
            l2_loss = self.l2_regularization * l2_loss / 2
        if self.loss == 'mean_squared_error':
            return self.square_error(y_true, y_pred)[0] + l2_loss, y_pred - y_true
        else:
            return self.cross_entropy(y_true, y_pred)[0] + l2_loss, y_pred - y_true

    def square_error(self, y_true, y_pred):
        return np.sum((y_pred - y_true)**2, axis=1)/2

    def cross_entropy(self, y_true, y_pred):
        return np.sum(np.multiply(-y_true, np.log(y_pred + sys.float_info.epsilon)), axis=1)

    # 该函数是为了得到计算梯度时所需要的一个部分
    def get_deltas(self):
        deltas = []
        if self.activate == 'sigmoid':
            for i in range(self.num_layers - 2):
                deltas.append(np.multiply(self.activations[i + 1], 1 - self.activations[i + 1]))
        if self.activate == 'leakyrelu':
            for i in range(self.num_layers - 2):
                deltas.append(np.where(self.activations[i+1] >= 0, 1, 0.001))

        return deltas

    def backward_propagation(self, y):
        _, loss_grad = self.loss_compute(y, self.activations[-1])
        gradients_W = []
        gradients_b = []

        deltas = self.get_deltas()

        # 第一步的梯度计算（第一步放在循环外是为了防止在循环内部写if语句）
        gradient_b = loss_grad
        gradients_b.insert(0, gradient_b)
        gradients_W.insert(0, gradient_b[:, :, np.newaxis] * self.activations[-2][:, np.newaxis, :] + self.l2_regularization * self.Weights[-1])
        for i in range(2, self.num_layers):
            gradient_b = np.multiply(np.dot(gradient_b, self.Weights[-i+1]), deltas[-i+1])
            gradients_b.insert(0, gradient_b)
            gradients_W.insert(0, gradient_b[:, :, np.newaxis] * self.activations[-i-1][:, np.newaxis, :] + self.l2_regularization * self.Weights[-i])

        self.gradients_W = gradients_W
        self.gradients_b = gradients_b

    # 这里是使用了 momentum 的版本
    def update_weights(self, lr, batch_size, full_bias):
        for i in range(self.num_layers-1):
            new_momentum_W = self.momentum_strength * self.momentum_W[i] - lr * (np.sum(self.gradients_W[i], axis=0) / batch_size)
            if full_bias is True or i == 0:
                new_momentum_b = self.momentum_strength * self.momentum_b[i] - lr * (np.sum(self.gradients_b[i], axis=0)[:, np.newaxis] / batch_size)
            else:
                new_momentum_b = np.zeros_like(self.Biases[i])
            self.Weights[i] += new_momentum_W
            self.Biases[i] += new_momentum_b
            self.momentum_W[i] = new_momentum_W
            self.momentum_b[i] = new_momentum_b

    def train(self, X, y, X_valid, y_valid, batch_size=16, epochs=100, lr=0.02, l2_regularization=0, momentum_strength=0, valid_cyc=1, full_bias=True):
        train_accuracy_scores = []
        valid_accuracy_scores = []
        loss_scores = []
        self.l2_regularization = l2_regularization
        self.momentum_strength = momentum_strength
        i = 0
        for i in range(epochs):
            mapping = np.arange(X.shape[0])
            np.random.shuffle(mapping)

            for j in tqdm(range(0, X.shape[0], batch_size)):
                if j + batch_size > X.shape[0]:
                    X_batch = X[mapping[-batch_size:], :]
                    y_batch = y[mapping[-batch_size:]]
                else:
                    X_batch = X[mapping[j:j+batch_size], :]
                    y_batch = y[mapping[j:j+batch_size]]
                self.forward_propogation(X_batch)
                self.backward_propagation(y_batch)
                self.update_weights(lr, batch_size, full_bias=full_bias)

            losses, _ = self.loss_compute(y_batch, self.activations[-1])
            loss = np.sum(losses) / batch_size
            loss_scores.append(loss)
            print(f"Epoch {i + 1}: train loss = {loss}, learning_rate = {lr}")

            if self.classifacation is True and i % valid_cyc == 0:
                train_act = np.array(self.predict(X))
                valid_act = np.array(self.predict(X_valid))
                train_pred = np.argmax(train_act, axis=1)
                train_true = np.argmax(y, axis=1)
                valid_pred = np.argmax(valid_act, axis=1)
                train_accuracy_scores.append(self.accuracy_score(train_true, train_pred))
                valid_accuracy_scores.append(self.accuracy_score(y_valid, valid_pred))
                print(f"Epoch {i + 1}: train loss = {loss}, train acc = {train_accuracy_scores[-1]}, valid acc = {valid_accuracy_scores[-1]}, learning_rate = {lr}")
            elif self.classifacation is False and i % valid_cyc == 0:
                train_act = np.array(self.predict(X))
                valid_act = np.array(self.predict(X_valid))
                train_pred = np.argmax(train_act, axis=1)
                train_true = np.argmax(y, axis=1)
                valid_pred = np.argmax(valid_act, axis=1)
                train_accuracy_scores.append(self.accuracy_score(train_true, train_pred))
                valid_accuracy_scores.append(self.accuracy_score(y_valid, valid_pred))
                print(f"Epoch {i + 1}: train loss = {loss}, train acc = {train_accuracy_scores[-1]}, valid acc = {valid_accuracy_scores[-1]}, learning_rate = {lr}")

        if self.classifacation is True:
            return train_accuracy_scores, valid_accuracy_scores, loss_scores
        else:
            return train_accuracy_scores, valid_accuracy_scores, loss_scores

    def predict(self, X):
        return self.forward_propogation(X, iftest=1)

    def accuracy_score(self, y_true, y_pred):
        equal = (y_true == y_pred)
        acc = np.sum(equal) / equal.size
        return acc

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    def load(self, path):
        with open(path, 'rb') as f:
            loaded_obj = pickle.load(f)
        self.Weights = loaded_obj.Weights
        self.Biases = loaded_obj.Biases

    def ft_backward(self, y):
        _, loss_grad = self.loss_compute(y, self.activations[-1])
        # 第一步的梯度计算
        gradient_b = loss_grad
        gradient_W = gradient_b[:, :, np.newaxis] * self.activations[-2][:, np.newaxis, :] + self.l2_regularization * self.Weights[-1]

        return gradient_W, gradient_b

    def ft_update_weights(self, gradient_W, gradient_b, lr, batch_size):
        new_momentum_W = self.momentum_strength * self.ft_momentum_W - lr * (np.sum(gradient_W, axis=0)/batch_size)
        new_momentum_b = self.momentum_strength * self.ft_momentum_b - lr * (np.sum(gradient_b, axis=0)[:, np.newaxis]/batch_size)
        self.Weights[-1] += new_momentum_W
        self.Biases[-1] += new_momentum_b
        self.ft_momentum_W = new_momentum_W
        self.ft_momentum_b = new_momentum_b

    def fine_tune(self, X, y, X_valid, y_valid, batch_size=16, epochs=100, lr=0.02, l2_regularization=0, momentum_strength=0.2, valid_cyc=1):
        if self.classifacation is True:
            train_accuracy_scores = []
            valid_accuracy_scores = []
        loss_scores = []
        self.l2_regularization = l2_regularization
        self.momentum_strength = momentum_strength
        self.ft_momentum_W = np.zeros_like(self.Weights[-1])
        self.ft_momentum_b = np.zeros_like(self.Biases[-1])
        i = 0
        for i in range(epochs):
            mapping = np.arange(X.shape[0])
            np.random.shuffle(mapping)

            for j in range(0, X.shape[0], batch_size):
                if j + batch_size > X.shape[0]:
                    X_batch = X[mapping[-batch_size:], :]
                    y_batch = y[mapping[-batch_size:]]
                else:
                    X_batch = X[mapping[j:j+batch_size], :]
                    y_batch = y[mapping[j:j+batch_size]]
                self.forward_propogation(X_batch)
                gradient_W, gradient_b = self.ft_backward(y_batch)
                self.ft_update_weights(gradient_W, gradient_b, lr, batch_size)

            losses, _ = self.loss_compute(y_batch, self.activations[-1])
            loss = np.sum(losses) / batch_size
            loss_scores.append(loss)
            print(f"Epoch {i}: train loss = {loss}, learning_rate = {lr}")

            if self.classifacation is True and i % valid_cyc == 0:
                train_act = np.array(self.predict(X))
                valid_act = np.array(self.predict(X_valid))
                train_pred = np.argmax(train_act, axis=1)
                train_true = np.argmax(y, axis=1)
                valid_pred = np.argmax(valid_act, axis=1)
                train_accuracy_scores.append(self.accuracy_score(train_true, train_pred))
                valid_accuracy_scores.append(self.accuracy_score(y_valid, valid_pred))
                print(f"Epoch {i}: train loss = {loss}, train acc = {train_accuracy_scores[-1]}, valid acc = {valid_accuracy_scores[-1]}, learning_rate = {lr}")
            elif self.classifacation is False and i % valid_cyc == 0:
                train_act = np.array(self.predict(X))
                valid_act = np.array(self.predict(X_valid))
                valid_loss = self.loss_compute(y_valid, valid_act)[0] / X_valid.shape[0]
                print(f"Epoch {i}: train loss = {loss}, valid loss = {valid_loss}, learning_rate = {lr}")

        if self.classifacation is True:
            return train_accuracy_scores, valid_accuracy_scores, loss_scores
        else:
            return loss_scores


class Conv2d():
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, bias=True):
        # input and output
        self.input = None
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.X_col = None
        self.W_col = None

        # params
        self.params = {}
        self.params['W'] = None
        self.params['b'] = None
        std = np.sqrt(2 / (self.in_channels + self.out_channels))
        self.params['W'] = np.random.normal(0, std, size=(self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1]))
        if bias:
            self.params['b'] = np.zeros((self.out_channels, ))
        self.grads = {}

    def forward(self, input):
        if isinstance(input, np.ndarray) is False:
            input = input.numpy()
        self.input = input
        batchsize, _, in_H, in_W = input.shape

        out_H = (in_H - self.kernel_size[0] + 2 * self.padding) // self.stride + 1
        out_W = (in_W - self.kernel_size[1] + 2 * self.padding) // self.stride + 1

        # 这里使用了快速卷积，利用im2col将卷积运算转换为了矩阵乘法，大大加快了运算速度
        input_col = im2col(input, self.kernel_size[0], self.kernel_size[1], self.stride, self.padding)
        W_col = self.params['W'].reshape((self.out_channels, -1))
        output = np.dot(W_col, input_col)
        output = np.array(np.hsplit(output, batchsize)).reshape((batchsize, self.out_channels, out_H, out_W))
        self.W_col = W_col
        self.input_col = input_col

        if self.params['b'] is not None:
            output += self.params['b'][:, np.newaxis, np.newaxis]

        return output

    def backward(self, output_grad, l2_regularization=0):
        batch_size = output_grad.shape[0]
        output_grad_col = output_grad.reshape((output_grad.shape[0] * output_grad.shape[1], -1))
        output_grad_col = np.array(np.vsplit(output_grad_col, batch_size))
        output_grad_col = np.concatenate(output_grad_col, axis=-1)

        # 同样用矩阵乘法的视角计算梯度，加快计算速度
        self.grads['W'] = np.dot(output_grad_col, self.input_col.T).reshape(self.params['W'].shape) + l2_regularization * self.params['W']
        self.grads['b'] = np.mean(output_grad, axis=(0, 2, 3)) + l2_regularization * self.params['b']
        input_grad = np.dot(self.W_col.T, output_grad_col)
        input_grad = col2im(input_grad, self.input.shape, self.kernel_size[0], self.kernel_size[1], self.stride, self.padding)
        return input_grad


class Conv_BP():
    def __init__(self, in_channels, out_channels, kernel_size, layers, stride=1, padding=1, bias=True, dropout=0, classifacation=False,
                 activate='leakyrelu'):
        self.conv = Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias)
        self.bp_net = back_propagation(layers, dropout, classifacation, activate)
        self.momentum_W = np.zeros_like(self.conv.params['W'])
        self.momentum_b = np.zeros_like(self.conv.params['b'])

    def forward(self, X):
        X = self.conv.forward(X)
        self.conv_out_shape = X.shape
        X = self.bp_net.forward_propogation(X.reshape((X.shape[0], -1)))
        return X

    def backward(self, y):
        self.bp_net.backward_propagation(y)
        grad_conv_out = np.dot(self.bp_net.gradients_b[0], self.bp_net.Weights[0])
        grad_conv_out = grad_conv_out.reshape(self.conv_out_shape)
        self.conv.backward(grad_conv_out, self.l2_regularization)

    def update_weights(self, lr, batch_size, full_bias):
        self.bp_net.update_weights(lr, batch_size, full_bias)
        new_momentum_W = self.bp_net.momentum_strength * self.momentum_W - lr * (np.sum(self.conv.grads['W'], axis=0) / batch_size)
        new_momentum_b = self.bp_net.momentum_strength * self.momentum_b - lr * (np.sum(self.conv.grads['b'], axis=0) / batch_size)
        self.conv.params['W'] += new_momentum_W
        self.conv.params['b'] += new_momentum_b
        self.momentum_W = new_momentum_W
        self.momentum_b = new_momentum_b

    def predict(self, X):
        return self.bp_net.predict(self.conv.forward(X).reshape((X.shape[0], -1)))

    def train(self, X, y, X_valid, y_valid, batch_size=16, epochs=100, lr=0.02, l2_regularization=0.01, momentum_strength=0.1, valid_cyc=1, full_bias=True):
        if self.bp_net.classifacation is True:
            train_accuracy_scores = []
            valid_accuracy_scores = []
        loss_scores = []
        self.bp_net.l2_regularization = l2_regularization
        self.l2_regularization = l2_regularization
        self.bp_net.momentum_strength = momentum_strength
        for i in range(epochs):
            mapping = np.arange(X.shape[0])
            np.random.shuffle(mapping)

            for j in tqdm(range(0, X.shape[0], batch_size)):
                if j + batch_size > X.shape[0]:
                    X_batch = X[mapping[-batch_size:]]
                    y_batch = y[mapping[-batch_size:]]
                else:
                    X_batch = X[mapping[j:j+batch_size]]
                    y_batch = y[mapping[j:j+batch_size]]
                self.forward(X_batch)
                self.backward(y_batch)
                self.update_weights(lr, batch_size, full_bias)

            losses, _ = self.bp_net.loss_compute(y_batch, self.bp_net.activations[-1])
            loss = np.sum(losses) / batch_size
            loss_scores.append(loss)
            print(f"Epoch {i + 1}: train loss = {loss}, learning_rate = {lr}")

            if self.bp_net.classifacation is True and i % valid_cyc == 0:
                train_act = np.array(self.predict(X))
                print(train_act.shape)
                valid_act = np.array(self.predict(X_valid))
                train_pred = np.argmax(train_act, axis=1)
                train_true = np.argmax(y, axis=1)
                valid_pred = np.argmax(valid_act, axis=1)
                train_accuracy_scores.append(self.bp_net.accuracy_score(train_true, train_pred))
                valid_accuracy_scores.append(self.bp_net.accuracy_score(y_valid, valid_pred))
                print(f"Epoch {i + 1}: train loss = {loss}, train acc = {train_accuracy_scores[-1]}, valid acc = {valid_accuracy_scores[-1]}, learning_rate = {lr}")
            elif self.bp_net.classifacation is False and i % valid_cyc == 0:
                train_act = np.array(self.predict(X))
                valid_act = np.array(self.predict(X_valid))
                valid_loss = self.bp_net.loss_compute(y_valid, valid_act)[0] / X_valid.shape[0]
                print(f"Epoch {i + 1}: train loss = {loss}, valid loss = {valid_loss}, learning_rate = {lr}")

        if self.bp_net.classifacation is True:
            return train_accuracy_scores, valid_accuracy_scores, loss_scores
        else:
            return loss_scores

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    def load(self, path):
        with open(path, 'rb') as f:
            loaded_obj = pickle.load(f)
        self.conv = loaded_obj.conv
        self.bp_net = loaded_obj.bp_net
