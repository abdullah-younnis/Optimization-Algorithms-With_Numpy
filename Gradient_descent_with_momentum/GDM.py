import numpy as np


def Gradient_descent_with_momentum(
    X, y, eta=0.01, beta=0.9, n_epochs=100, batch_size=32
):
    """
    Implementation of Gradient descent with momentum algorithm using numpy.
    Below are my explaination for compomnents of gradient_descent_with_momentum
    Parameters:
    - X: numpy array of shape (n_samples, n_features), the feature matrix
    - y: numpy array of shape (n_samples,), the target vector
    - beta: float, Beta helps to converge more quickly and avoid getting stuck in flat regions of the cost function.
    - n_epochs: int, the number of epochs (iterations over the entire dataset)
    - batch_size: int, the number of samples to use in each mini-batch

    Returns:
    - theta: numpy array of shape (n_features,), the learned model parameters,
      also known as features or predictors,
      theta represents the coefficients (also known as weights or slopes)
    - losses: list of length (n_epochs), the training loss at each epoch
    """

    # intialize the model parameters
    n_samples, n_features = np.shape(X)
    theta = np.zeros(n_features)
    velocity_vector = np.zeros(n_features)
    # intialize list variable to store losses
    losses = []
    # loop over the epochs
    for epoch in range(n_epochs):
        # shuffle the data This can help to reduce any
        # biases or patterns in the data that might exist due
        # to the order in which the examples were collected or processed.
        permutation = np.random.permutation(n_samples)
        X_shuffled = X[permutation]
        y_shuffled = y[permutation]
        # intialize number n_bathces
        n_batches = n_samples // batch_size
        # Loop over mini-batches
        batch_losses = []
        for i in range(n_batches):
            X_batch = X_shuffled[i : i + batch_size]
            y_batch = y_shuffled[i : i + batch_size]
            # computing our gradient
            y_pred = np.dot(X_batch, theta)
            error = y_pred - y_batch
            x_transposed = np.transpose(X_batch)
            gradient = np.dot(x_transposed, error) / batch_size
            # Update the velocity vector
            velocity_vector = beta * velocity_vector + (1 - beta) * gradient
            # update model parameter
            theta = theta - eta * velocity_vector
            # Compute the mean squared error for the mini-batch
            y_pred = np.dot(X_batch, theta)
            batch_loss = np.mean((y_pred - y) ** 2)
            batch_losses.append(batch_loss)
    # Compute the training loss for the epoch
    epoch_loss = np.mean(batch_losses)
    losses.append(epoch_loss)

    return theta, losses
