# Gradient descent with momentum
In this repository I will implement GDM using numpy

### Gradient descent with momentum Components

Parameters:

    X: numpy array of shape (n_samples, n_features), the feature matrix
    y: numpy array of shape (n_samples,), the target vector
    beta: float, Beta helps to converge more quickly and avoid getting stuck in flat regions of the cost function.
    n_epochs: int, the number of epochs (iterations over the entire dataset)
    batch_size: int, the number of samples to use in each mini-batch
    
Returns:

    theta: numpy array of shape (n_features,), the learned model parameters
    losses: list of length (n_epochs), the training loss at each epoch



### Gradient descent with momentum Operations

* **Gradient computation**: At each iteration, the gradient of the loss function with respect to the model parameters is computed. This involves computing the partial derivatives of the loss function with respect to each of the model parameters.
Gradient is important because it captures all the partial derivatives of a scalar-valued multivariable function
In the case of scalar-valued multivariable functions, meaning those with a multidimensional input but a one-dimensional output, the answer is the gradient. The gradient of a function f f, denoted as nabla f ∇f, is the collection of all its partial derivatives into a vector.

* **Velocity initialization**: Initialize the velocity vector, which is a running average of the gradients, to zero.

* **Momentum update**: Update the velocity vector using a momentum term, beta, and the gradient. This involves computing the weighted average of the current velocity vector and the gradient vector.

* **Parameter update**: Once the gradient has been computed, the model parameters are updated by taking a step in the direction of the negative gradient, scaled by the learning rate. This involves multiplying the gradient vector by the learning rate and subtracting the result from the current parameter values.

* **Mini-batch sampling**: In order to compute the gradient efficiently, SGD typically uses mini-batch sampling. This involves randomly selecting a subset of the training data at each iteration and using that subset to compute the gradient. This reduces the computational burden of computing the gradient on the entire dataset and can help SGD converge faster.

* **Random initialization**: SGD typically starts with random initial values for the model parameters. This ensures that the algorithm explores a variety of parameter settings and can escape from local minima.

* **Stopping criteria**: SGD typically stops when a stopping criterion is met, such as when the algorithm has completed a fixed number of iterations or when the improvement in the loss function falls below a certain threshold.


### Strengths of Gradient Descent with Momentum include:

Gradient Descent with Momentum (GDM) has several strengths that make it a popular optimization algorithm for training machine learning models. Here are some of its strengths:

* **Faster convergence**: GDM accelerates convergence by using a velocity vector, which helps to smooth out the update direction and reduce oscillations. This can help the algorithm to converge more quickly and avoid getting stuck in flat regions of the cost function.

* **Improved generalization**: By reducing oscillations and smoothing out the update direction, GDM can help the model generalize better to new data. This is particularly useful for deep neural networks, where overfitting can be a major problem.

* **Robustness to noisy gradients**: GDM is more robust to noisy gradients compared to standard Gradient Descent (GD), as the momentum term can help to filter out noise and reduce variance in the gradient estimates.

* **No hyperparameter tuning**: GDM has fewer hyperparameters to tune compared to other optimization algorithms, such as Adagrad or Adam. This can simplify the training process and reduce the risk of overfitting to the validation set.

* **Easy to implement**: GDM is relatively easy to implement and can be used with a wide range of machine learning models.

### Weaknesses of Gradient Descent with Momentum include:

* **Requires careful tuning of hyperparameters**: Although GDM has fewer hyperparameters than other optimization algorithms, it still requires careful tuning of the learning rate, momentum, and other hyperparameters to achieve optimal performance. Poorly chosen hyperparameters can lead to slow convergence or even divergence.

* **Can overshoot the minimum**: The momentum term in GDM can cause the algorithm to overshoot the minimum of the cost function, especially if the learning rate is too high. This can lead to oscillations and slow convergence.

* **May not work well for shallow neural networks**: GDM is designed to work well for deep neural networks with highly non-convex cost functions. However, for shallow neural networks or other models with simpler cost functions, GDM may not provide significant benefits over standard Gradient Descent.

* **May converge to suboptimal solutions**: Like all optimization algorithms, GDM may converge to suboptimal solutions if the cost function has multiple local minima or saddle points. In some cases, GDM may even converge to a non-global minimum, especially if the learning rate is too high.

* **Higher computational complexity**: GDM has a higher computational complexity than standard Gradient Descent due to the additional velocity vector update operation. This can make GDM slower than GD in some cases, especially for small datasets or shallow neural networks.


### Recommended to use GDM in the following situations:

* **Deep neural networks**: GDM is highly effective for training deep neural networks with highly non-convex cost functions. The momentum term can help the algorithm to converge more quickly and avoid getting stuck in flat regions of the cost function, leading to faster convergence and better generalization performance.

* **Noisy gradient estimates**: GDM is more robust to noisy gradient estimates compared to standard Gradient Descent (GD), as the momentum term can help to filter out noise and reduce variance in the gradient estimates. This can make GDM more effective for training models on datasets with noisy or sparse features.

* **Large datasets**: GDM is particularly useful for training models on large datasets, as the mini-batch sampling can reduce the computational burden of computing the gradient on the entire dataset. This can make GDM faster and more efficient than standard GD on large datasets.

### Recommended not to use GDM in the following situations:

* **Non-differentiable cost functions**: GDM assumes that the cost function is differentiable with respect to the model parameters. If the cost function is non-differentiable, GDM may not be applicable. In such cases, other optimizationalgorithms, such as genetic algorithms or simulated annealing, may be more appropriate.

* **Low-dimensional data**: When dealing with low-dimensional data, the mini-batch sampling used in GDM may not provide significant computational benefits, and the additional velocity vector update operation can make GDM slower than GD. In such cases, GD or other simpler optimization algorithms may be more appropriate.

* **Limited memory**: GDM requires additional memory to store the velocity vector, which can be a concern for models with limited memory. In such cases, simpler optimization algorithms, such as GD or stochastic gradient descent (SGD), may be more appropriate.


### Conclusion

Gradient Descent with Momentum (GDM) is a powerful optimization algorithm that can be used to train a wide range of machine learning models, especially deep neural networks with highly non-convex cost functions. 

The momentum term in GDM helps to accelerate convergence, reduce oscillations, and improve generalization performance. 

GDM is particularly effective for dealing with noisy gradient estimates, training on large datasets, and requires fewer hyperparameters to tune compared to other optimization algorithms. However, GDM may not be the best choice for shallow neural networks or low-dimensional data, and may require careful hyperparameter tuning. Overall, GDM represents a valuable addition to the machine learning practitioner's toolkit and can be used to improve the performance of many different types of machine learning models.

### Resources

https://medium.com/optimization-algorithms-for-deep-neural-networks/gradient-descent-with-momentum-dce805cd8de8

https://www.youtube.com/watch?v=iudXf5n_3ro