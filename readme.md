# simple-mlp-trainer
This project is a minimal framework for training MLPs that implements gradient descent via reverse mode auto differentiation. Check out `demo.ipynb` to see training results on a simple linearly separable 2-dimensional dataset. 

- **tensor.py** is where we implemented reverse mode automatic differentiation.
- **mlp.py** consists of our representation for neurons, layers, and multi-layer perceptron.
- **optimizer.py** implements the gradient descent.
- **loss.py** defines the mean-square-error loss function 
- **activations.py** defines sigmoid, relu, and tanh activation functions 
