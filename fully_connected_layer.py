# Alex Yuan, June 2018

import numpy as np

class fully_connected_layer:
    def __init__(self,
                 num_inputs,
                 num_outputs,
                 activation_fxn=None,
                 afxn_prime=None,
                 weights_init=None,
                 bias_init=None,
                 reg_param=0):
        """
        num_inputs (type=int): Dimension of intput to the layer

        num_outputs (type=int): Dimension of layer output

        activation_fxn (type=function): activation function. See
        activation_functions.py for examples.

        afxn_prime(type=function): derivative of activation function. See
        activation_functions.py for examples.

        weights_init (type=numpy.array): An mxn array of initial weight values,
        where m=num_outputs and n=num_inputs

        bias_init (type=numpy.array): An mx1 array of initial bias values,
        where m=num_outputs

        reg_param: Sorry, this feature is not yet supported. Just leave it
        as default for now.
        """
        # save arguments
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.activation_fxn = activation_fxn
        self.afxn_prime = afxn_prime
        self.weights_init = weights_init
        self.bias_init = bias_init
        self.reg_param = reg_param
        # handle default or non-default inputs
        if type(self.activation_fxn) == type(None):
            self.activation_fxn = self.__default_activation_fxn
        if type(self.afxn_prime) == type(None):
            self.afxn_prime = self.__default_afxn_prime
        if type(weights_init) == type(None):
            self.Weights = self.__default_weights_init(self.num_outputs, self.num_inputs)
        else:
            try:
                assert (self.num_outputs, self.num_inputs) == np.shape(self.weights_init)
            except:
                msg = "weights_init must have shape %s" % str((self.num_outputs, self.num_inputs))
                raise TypeError(msg)
            self.Weights = self.weights_init
        if type(self.bias_init) == type(None):
            self.Bias = self.__default_bias_init(self.num_outputs)
        else:
            try:
                assert (num_outputs, 1) == np.shape(self.bias_init)
            except:
                msg = "bias_init must have shape %s" % str((num_outputs, 1))
                raise TypeError(msg)
            self.Bias = self.bias_init

    def __default_activation_fxn(self, A):
        # relu
        return np.maximum(A,0)

    def __default_afxn_prime(self,A):
        # relu prime
        return (A > 0) * 1.0

    def __default_weights_init(self, num_outputs, num_inputs):
        # zeros
        return np.random.normal(loc=0, scale=0.4,
                                size=(num_outputs,num_inputs))

    def __default_bias_init(self, num_outputs):
        # zeros
        return np.random.normal(loc=0, scale=0.4,
                                size=(num_outputs,1))

    def evaluate_layer(self, A):
        return self.activation_fxn(np.matmul(self.Weights, A) + self.Bias)

    def evaluate_fprime(self, A):
        try:
            return self.afxn_prime(np.matmul(self.Weights, A) + self.Bias)
        except ValueError:
            print("weights:")
            print(self.Weights)
            print("A:")
            print(A)
            raise ValueError("murp")

    # TODO: implement regularization with self.reg_param
