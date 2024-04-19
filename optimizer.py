from mlp import MLP 
from tensor import Tensor 

class Optimizer:

    def __init__(self, model: MLP, learning_rate: float):
        self.model = model
        self.learning_rate = learning_rate

    def step(self):
        for layer in self.model.layers:
            for neuron in layer.neurons:
                for weight_idx, weight in enumerate(neuron.weights):
                    neuron.weights[weight_idx].value -= self.learning_rate * weight.grad
                neuron.bias.value -= self.learning_rate * neuron.bias.grad

    def zero_grad(self, loss_func: Tensor):
        tensor_deque = loss_func.topoSort()
        while len(tensor_deque) != 0:
            tensor_deque.pop().grad = 0
