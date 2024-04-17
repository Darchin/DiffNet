from mlp import *
def main():
    train_data = [[0,0],[0,1],[1,0],[1,1]]
    labels = [0, 1, 1, 0]

    input_size = len(train_data[0]) # Number of features in the input layer
    layer_sizes = [2, 3, 1] # Number of neurons in each hidden and output layer
    model = MLP(input_size, layer_sizes)

    n_epochs = 100
    opt = Optimizer(model, 0.01)
    for _ in range(n_epochs):
    # Forward pass: Compute predictions for the entire dataset
        loss = Tensor(0)
        for x, y in zip(train_data, labels):
            yhat = model(x)
            loss += Tensor.CostFunctions.mse(y, yhat)
        opt.zero_grad()
        opt.step(loss)
        print(loss.value)
    # Zero the gradients to prevent accumulation from previous iterations
    # optim.zero_grad()
    # Backward pass: Compute the gradient of the loss function with respect to model parameters
    # loss.backward()
    # Update the model parameters using the optimizer
    # optim.step()

if __name__ == '__main__':
    main()