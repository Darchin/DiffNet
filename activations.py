import math 
from tensor import Tensor

class F:

    @staticmethod
    def tanh(var: Tensor):
        out = Tensor(value=math.tanh(var.value), children=set([var]))
        def backward():
            var.grad = (1-math.tanh(var.value)**2) * out.grad
        out._backward = backward
        return out

    @staticmethod
    def sigmoid(var: Tensor):
        out = Tensor(value=1/(1+math.exp(-var.value)), children=set([var]))
        def backward():
            var.grad = (1/(1+math.exp(-var.value)) * (1 - 1/(1+math.exp(-var.value)))) * out.grad
        out._backward = backward
        return out

    @staticmethod
    def relu(var: Tensor):
        out = Tensor(value=max([0, var.value]), children=set([var]))
        def backward():
            var.grad = 0 if var.value <= 0 else out.grad
        out._backward = backward
        return out


