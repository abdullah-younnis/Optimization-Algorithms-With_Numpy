import numpy as np

def stochastic_gradient_descent(X, y, eta=0.01, n_epochs=100, batch_size=32):
    """
    Implementation of Stochastic Gradient Descent algorithm using numpy.
    Below are my explaination for compomnents of stochastic_gradient_descent
    Parameterss:
    - X: numpy array of shape (n_samples, n_features), the feature matrix
    - y: numpy array of shape (n_samples,), the target vector
    - eta: float, the learning rate of the algorithm
    - n_epochs: int, the number of epochs (iterations over the entire dataset)
    - batch_size: int, the number of samples to use in each mini-batch and act as Stopping criteria
    
    Returns:
    - theta: numpy array of shape (n_features,), the learned model parameters,
      also known as features or predictors,
      theta represents the coefficients (also known as weights or slopes)
    - losses: list of length (n_epochs), the training loss at each epoch
    """
    # Initialize model parameters and giving random initialization
    n_samples, n_features = np.shape(X)
    theta = np.zeros(n_features)
    
    # Initialize list to store training losses
    losses = []
    
    # Loop over epochs
    for epoch in range(n_epochs):
        # Shuffle the data
        permutation = np.random.permutation(n_samples)
        X_shuffled = X[permutation]
        y_shuffled = y[permutation]
        
        # Loop over mini-batches and mini-batches here gicing the SGD more accuracy
        for i in range(0, n_samples, batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]
            
            # Compute the gradient
            y_pred = np.dot(X_batch,theta)
            error = (y_pred - y_batch)
            x_transposed = np.transpose(X_batch)
            gradient = np.dot(x_transposed, error) / batch_size
            # Update the model parameters
            theta -= eta * gradient
        
        # Compute the training loss at this epoch
        y_pred = np.dot(X,theta)
        loss = np.mean((y_pred - y)**2)
        losses.append(loss)
        
    return theta, losses
