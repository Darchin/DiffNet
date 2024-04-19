from tensor import Tensor 
from random import random 

class Neuron:
    def __init__(self, input_size, activation_function):
        self.weights = [Tensor(random()) for i in range(input_size)]
        self.bias = Tensor(random())
        self.activation_function = activation_function

    def forward(self, x):
        weighted_sum = sum([w_i * x_i for w_i, x_i in zip(self.weights, x)])
        return self.activation_function(weighted_sum + self.bias)

    def __call__(self, x):
        return self.forward(x)

    def parameters(self):
        return [weight.grad for weight in self.weights] + [self.bias.grad]

class Layer:
    def __init__(self, input_size, output_size, activation_function):
        self.neurons = [Neuron(input_size, activation_function) for _ in range(output_size)]

    def forward(self, x):
        out = [neuron(x) for neuron in self.neurons]
        return out[0] if len(out)==1 else out

    def __call__(self, x):
        return self.forward(x)

class MLP:
    def __init__(self, input_size, layer_sizes, activation_functions):
        layers_total = [input_size] + layer_sizes
        self.layers = [Layer(layers_total[i], layers_total[i+1], j) for i, j in zip(range(len(layer_sizes)), activation_functions)]

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def __call__(self, x):
        return self.forward(x)

    def parameters(self):
        return [[neuron.parameters() for neuron in layer.neurons] for layer in self.layers]