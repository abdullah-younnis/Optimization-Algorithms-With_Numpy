# Optimization_algorithms_using_numpy
In this repository I will implement SGD using numpy

### Stochastic Gradient Descent Components

Components:
    - X: numpy array of shape (n_samples, n_features), the feature matrix
    - y: numpy array of shape (n_samples,), the target vector
    - learning_rate: float, the learning rate of the algorithm
    - n_epochs: int, the number of epochs (iterations over the entire dataset)
    - batch_size: int, the number of samples to use in each mini-batch
 Returns:
            - theta: numpy array of shape (n_features,), the learned model parameters
            - losses: list of length (n_epochs), the training loss at each epoch



### Stochastic Gradient Descent Operations

* Gradient computation: At each iteration, the gradient of the loss function with respect to the model parameters is computed. This involves computing the partial derivatives of the loss function with respect to each of the model parameters.

* Parameter update: Once the gradient has been computed, the model parameters are updated by taking a step in the direction of the negative gradient, scaled by the learning rate. This involves multiplying the gradient vector by the learning rate and subtracting the result from the current parameter values.

* Mini-batch sampling: In order to compute the gradient efficiently, SGD typically uses mini-batch sampling. This involves randomly selecting a subset of the training data at each iteration and using that subset to compute the gradient. This reduces the computational burden of computing the gradient on the entire dataset and can help SGD converge faster.

* Random initialization: SGD typically starts with random initial values for the model parameters. This ensures that the algorithm explores a variety of parameter settings and can escape from local minima.

* Stopping criteria: SGD typically stops when a stopping criterion is met, such as when the algorithm has completed a fixed number of iterations or when the improvement in the loss function falls below a certain threshold.
