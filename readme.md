# DiffNet 
DiffNet is an educational neural network that is trained using reverse mode automatic differentiation. It provides similar API to pytorch. Check out `demo.ipynb` to see training results on a simple linearly separable 2-dimensional dataset. 

- **tensor.py** is where we implemented reverse mode automatic differentiation.
- **mlp.py** consists of our representation for neurons, layers, and multi-layer perceptron.
- **optimizer.py** optimizer implements the simplest optimizer which just uses the learning rate to subtract derivatives from their corresponding weights.  
- **loss.py** defines the mean-square-error loss function 
- **activations.py** defines sigmoid, relu, and tanh activation functions 