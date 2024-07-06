from atto import Value
import random

class Module:

    def set_gradients_zero(self):
        for param in self.parameters():
            param.gradient = 0

    def parameters(self):
        return []

class Neuron(Module):

    def __init__(self, n_input, linear=False):
        self.w = [Value(random.uniform(-1,1)) for i in range(n_input)]
        self.b = Value(0)
        self.linear = linear

    def __call__(self, x):
        preactivation = sum([wi * xi for (wi, xi) in zip(self.w, x)], self.b)
        activation = preactivation if self.linear else preactivation.relu()

        return activation
    
    def parameters(self):
        return self.w + [self.b]
    
class Layer(Module):

    def __init__(self, n_input, n_output, **kwargs):
        self.neurons = [Neuron(n_input, **kwargs) for i in range(n_output)]

    def __call__(self, x):
        output = [neuron(x) for neuron in self.neurons]
        return output[0] if len(output) == 1 else output
    
    def parameters(self):
        return [parameter for neuron in self.neurons for parameter in neuron.parameters()]
    
class MultiLayerPerceptron(Module):
    
    def __init__(self, n_input, n_outputs):
        sizes = [n_input] + n_outputs
        self.layers = [Layer(sizes[i], sizes[i+1], linear=i==len(sizes)-1) for i in range(len(n_outputs))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [parameters for layer in self.layers for parameters in layer.parameters()]

