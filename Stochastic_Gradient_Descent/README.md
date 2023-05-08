# Stochastic Gradient Descent
In this repository I will implement SGD using numpy

### Stochastic Gradient Descent Components

Components:


    - X: numpy array of shape (n_samples, n_features), the feature matrix
    - y: numpy array of shape (n_samples,), the target vector
    - learning_rate: float, the learning rate of the algorithm
    - n_epochs: int, the number of epochs (iterations over the entire dataset)
    - batch_size: int, the number of samples to use in each mini-batch act as Stopping criteria
    
 Returns:
 
 
            - theta: numpy array of shape (n_features,), the learned model parameters
            - losses: list of length (n_epochs), the training loss at each epoch



### Stochastic Gradient Descent Operations

* **Gradient computation**: At each iteration, the gradient of the loss function with respect to the model parameters is computed. This involves computing the partial derivatives of the loss function with respect to each of the model parameters.

* **Parameter update**: Once the gradient has been computed, the model parameters are updated by taking a step in the direction of the negative gradient, scaled by the learning rate. This involves multiplying the gradient vector by the learning rate and subtracting the result from the current parameter values.

* **Mini-batch sampling**: In order to compute the gradient efficiently, SGD typically uses mini-batch sampling. This involves randomly selecting a subset of the training data at each iteration and using that subset to compute the gradient. This reduces the computational burden of computing the gradient on the entire dataset and can help SGD converge faster.

* **Random initialization**: SGD typically starts with random initial values for the model parameters. This ensures that the algorithm explores a variety of parameter settings and can escape from local minima.

* **Stopping criteria**: SGD typically stops when a stopping criterion is met, such as when the algorithm has completed a fixed number of iterations or when the improvement in the loss function falls below a certain threshold.


### Strengths of Stochastic Gradient Descent include:

* **Efficiency**: SGD is computationally efficient because it updates the parameters of the model using only a subset of the training data at each iteration. This makes it well-suited for large datasets and complex models.

* **Convergence speed**: SGD can converge faster than batch gradient descent because it updates the parameters more frequently, which can help the algorithm escape local minima and find the optimal solution more quickly.

* **Robustness**: SGD is less sensitive to noisy data than batch gradient descent because it updates the parameters more frequently and can avoid getting stuck in local minima.

### Weaknesses of Stochastic Gradient Descent include:

* **Variance in updates**: Because SGD updates the parameters using only a subset of the training data, the updates can be more noisy and have higher variance than batch gradient descent. This can make the learning process more unstable and require more tuning of the hyperparameters.

* **Learning rate tuning**: SGD requires careful tuning of the learning rate to ensure that the algorithm converges to the optimal solution. If the learning rate is too high, the algorithm can oscillate and fail to converge. If the learning rate is too low, the algorithm may converge too slowly.

* **Initialization sensitivity**: SGD can be sensitive to the initial values of the parameters, which can affect the convergence speed and final performance of the algorithm.


### Recommended to use SGD in the following situations:

* **Large datasets**: SGD is computationally efficient and can handle large datasets because it updates the parameters using only a subset of the training data at each iteration.

* **High-dimensional data**: SGD can handle high-dimensional data because it updates the parameters using only a subset of the features at each iteration.

* **Non-convex optimization problems**: SGD can be effective for non-convex optimization problems because it can escape local minima and find the optimal solution more quickly than batch gradient descent.

* **Online learning**: SGD can be used for online learning because it can update the parameters as new data becomes available.

* **Convex optimization problems with noisy data**: SGD is less sensitive to noisy data than batch gradient descent because it updates the parameters more frequently and can avoid getting stuck in local minima.

- Overall, SGD is a powerful optimization algorithm that can be effective for a wide range of machine learning problems. It is particularly well-suited for large datasets, high-dimensional data, non-convex optimization problems, and online learning. However, the hyperparameters of SGD must be carefully tuned to ensure that the algorithm converges to the optimal solution.

### Recommended to not use SGD in the following situations:

* **Small datasets**: SGD can be less effective for small datasets because the updates can be more noisy and have higher variance. In this case, batch gradient descent or even analytical solutions may be more appropriate.

* **Low-dimensional data**: SGD can be less effective for low-dimensional data because the updates can become too noisy. In this case, batch gradient descent or even analytical solutions may be more appropriate.

* **Sensitive to initialization**: If the optimization problem is sensitive to the initialization of the parameters, or if the model has a large number of local minima, SGD may not be the best choice. In this case, more robust optimization algorithms such as Adam or Adagrad may be more appropriate.

* **Need for exact solutions**: If the optimization problem requires an exact solution, SGD may not be the best choice because it only provides an approximate solution. In this case, optimization algorithms such as L-BFGS or conjugate gradient descent may be more appropriate.

* **Non-differentiable objective functions**: If the objective function is non-differentiable, SGD may not be the best choice. In this case, optimization algorithms such as simulated annealing or genetic algorithms may be more appropriate.
