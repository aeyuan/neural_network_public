# Alex Yuan, June 2018

import numpy as np

class ff_newtork:
    def __init__(self,
                 loss_fxn=None,
                 loss_fxn_prime=None,
                 learn_rate = 0.01,
                 optimizer="Nesterov",
                 mu=0.95,
                 grad_clip=5):
        """
        loss_fxn (type=function): A function whose first parameter is y, whose second parameter
        is yhat, and which returns the element-wise loss of y and yhat. Default
        is: (0.5) * (yhat - y) ** 2

        loss_fxn_prime (type=function): A function whose first parameter is y, whose second
        parameter is yhat, and which returns the element-wise gradient of the
        loss with respect to yhat. Default is: (yhat - y)

        learn_rate (type=float): Learning rate

        optimizer (type=string): Specifies the optimization algorithm. Options are
        "SGD" for gradient descent or "Nesterov" for Nesterov momentum

        mu (type=float): mu parameter for Nesterov momentum optimizer

        grad_clip (type=float): Maximum absolute value of gradient. Let
        grad_clip=0 or grad_clip=False to turn off gradient clipping.
        """
        self.layers = []
        self.loss_fxn = loss_fxn
        self.loss_fxn_prime = loss_fxn_prime
        self.learn_rate = learn_rate
        self.optimizer = optimizer
        self.mu = mu
        self.grad_clip = grad_clip
        if self.loss_fxn == None:
            self.loss_fxn = self.__default_loss_fxn
        if self.loss_fxn_prime == None:
            self.loss_fxn_prime = self.__default_loss_fxn_prime
        if not self.optimizer in ["Nesterov", "SGD"]:
            msg = "optimizer must be \'Nesterov\' or \'SGD\'"
            raise ValueError(msg)
        if self.optimizer == "Nesterov":
            self.v_W = []
            self.v_b = []

    def __default_loss_fxn(self, y, yhat):
        return (0.5) * (yhat - y) ** 2

    def __default_loss_fxn_prime(self, y, yhat):
        return (yhat - y)

    def __initialize_DP_dict(self, num_layers):
        gh_n = {}
        gW_n = {}
        gb_n = {}
        for i in range(num_layers + 1):
            gh_n[i] = 0
            gW_n[i] = 0
            gb_n[i] = 0
        return (gh_n, gW_n, gb_n)

    def add_layer(self, new_layer):
        self.layers.append(new_layer)
        if self.optimizer == "Nesterov":
            self.v_W.append(np.zeros_like(new_layer.Weights))
            self.v_b.append(np.zeros_like(new_layer.Bias))

    def predict(self, A):
        """
        Makes a prediction based on the current network parameters

        ARGUMENTS:

        A (type=numpy.array): Features of a batch as an mxn numpy array, where
        m is the number of features and n is the number of samples in a batch

        RETURNS (type=numpy.ndarray):

        Predictions
        """
        self.data = [A]
        for i in range(len(self.layers)):
            input = self.data[i]
            output = self.layers[i].evaluate_layer(input)
            self.data.append(output)
        return self.data[-1]

    def train(self, A, y):
        """
        Trains the network on a single batch of data

        ARGUMENTS:

        A (type=numpy.array): Features of a batch as an mxn numpy array, where
        m is the number of features and n is the number of samples in a batch

        y (type=numpy.array): Response of a batch as an mx1 numpy array

        RETURNS (type=numpy.float64):

        mean loss
        """
        num_layers = len(self.layers)
        yhat = self.predict(A)
        gyhat = self.loss_fxn_prime(y, yhat)
        (gh_n_Total, gW_n_Total, gb_n_Total) = self.__initialize_DP_dict(num_layers)
        # loop over batches for gradient computation. TODO: change this from
        # a loop to optimize for speed later
        for i in range(len(gyhat[0])):
            # here i denotes the sample index
            (gh_n, gW_n, gb_n) = self.__initialize_DP_dict(num_layers)
            # base case
            gh_n[num_layers] = np.expand_dims(gyhat[:,i],-1)
            self.__backprop(gh_n, gW_n, gb_n, num_layers, i)
            (gW_n_Total, gb_n_Total) = self.__update_totals((gW_n_Total, gb_n_Total),(gW_n, gb_n), num_layers)
            #print(gW_n_Total)
        # update weights based on gradients
        for layer_idx in range(num_layers):
            # divide gradient by batch size
            gW_n_Total[layer_idx] = gW_n_Total[layer_idx] / A.shape[1]
            gb_n_Total[layer_idx] = gb_n_Total[layer_idx] / A.shape[1]
            if self.grad_clip:
                gW_n_Total[layer_idx] = np.maximum(gW_n_Total[layer_idx], -self.grad_clip)
                gW_n_Total[layer_idx] = np.minimum(gW_n_Total[layer_idx], self.grad_clip)
                gb_n_Total[layer_idx] = np.maximum(gb_n_Total[layer_idx], -self.grad_clip)
                gb_n_Total[layer_idx] = np.minimum(gb_n_Total[layer_idx], self.grad_clip)
            if self.optimizer == "SGD":
                self.layers[layer_idx].Weights -= self.learn_rate * gW_n_Total[layer_idx]
                self.layers[layer_idx].Bias -= self.learn_rate * gb_n_Total[layer_idx]
            if self.optimizer == "Nesterov":
                """
                from http://cs231n.github.io/neural-networks-3/
                v_prev = v # back this up
                v = mu * v - learning_rate * dx # velocity update stays the same
                x += -mu * v_prev + (1 + mu) * v # position update changes form
                """
                # Nesterov
                v_W_prev = self.v_W[layer_idx]
                self.v_W[layer_idx] = self.mu * self.v_W[layer_idx] - self.learn_rate * gW_n_Total[layer_idx]
                self.layers[layer_idx].Weights = (
                    self.layers[layer_idx].Weights +
                    -self.mu * v_W_prev +
                    (1 + self.mu) * self.v_W[layer_idx])
                v_b_prev = self.v_b[layer_idx]
                self.v_b[layer_idx] = self.mu * self.v_b[layer_idx] - self.learn_rate * gb_n_Total[layer_idx]
                self.layers[layer_idx].Bias = (
                    self.layers[layer_idx].Bias +
                    -self.mu * v_b_prev +
                    (1 + self.mu) * self.v_b[layer_idx])
        return np.mean(self.loss_fxn(y, yhat))


    def __backprop(self, gh_n, gW_n, gb_n, num_layers, sample_index):
        for i in range(num_layers-1, -1, -1):
            # here i denotes the layer index
            gW_n[i] = self.__compute_gW_n(gh_n[i+1], i, sample_index)
            gb_n[i] = self.__compute_gb_n(gh_n[i+1], i, sample_index)
            gh_n[i] = self.__compute_gh_n(gh_n[i+1], i, sample_index)

    def __compute_gW_n(self, gh_next, layer_num, si):
        input = np.expand_dims(self.data[layer_num][:,si],-1)
        result = None
        for row_num in range(self.layers[layer_num].Bias.shape[0]):
            row = (
                gh_next[row_num,:] *
                np.expand_dims(self.layers[layer_num].evaluate_fprime(input)[row_num,:],0) *
                input.T)
            if type(result) == type(None):
                result = row
            else:
                result = np.concatenate((result,row))
        return np.array(result)

    def __compute_gb_n(self, gh_next, layer_num, si):
        input = np.expand_dims(self.data[layer_num][:,si],-1)
        return gh_next * self.layers[layer_num].evaluate_fprime(input)

    def __compute_gh_n(self, gh_next, layer_num, si):
        input = np.expand_dims(self.data[layer_num][:,si],-1)
        B = gh_next * self.layers[layer_num].evaluate_fprime(input)
        W = self.layers[layer_num].Weights
        return np.matmul(W.T, B)

    def __update_totals(self, Totals, Currents, num_layers):
        (gW_n_Total, gb_n_Total) = Totals
        (gW_n, gb_n) = Currents
        for i in range(num_layers):
            gW_n_Total[i] += gW_n[i]
            gb_n_Total[i] += gb_n[i]
        return (gW_n_Total, gb_n_Total)
