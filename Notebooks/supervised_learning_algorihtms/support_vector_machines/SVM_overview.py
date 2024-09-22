import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

# Initialize random state and generate a dataset
random_state = np.random.RandomState(0)
n_samples = 20
X = random_state.rand(n_samples, 2)  # Random 2D points
y = np.ones(n_samples)  # Label all points as 1 initially
y[X[:, 0] + 0.1 * random_state.randn(n_samples) < 0.5] = -1.0  # Update labels based on condition

plt.figure(figsize=(10, 10))  # Create a figure to hold subplots

# Loop over different values of the regularization parameter C
for i, C in enumerate([1e-3, 1.0, 1e3, 1e6]):  # Replace np.inf with a large number 1e6
    # Create a linear SVM model with the given value of C
    estimator = SVC(kernel="linear", C=C, random_state=random_state)
    estimator.fit(X, y)  # Fit the model on the dataset

    # Create a range of values for plotting the decision boundary
    xx = np.linspace(0, 1, 100)

    # Calculate slope 'a' for the decision boundary
    # Check to avoid division by zero in case the second coefficient is zero
    if estimator.coef_[0][1] != 0:
        a = -estimator.coef_[0][0] / estimator.coef_[0][1]
        yy = a * xx - estimator.intercept_[0] / estimator.coef_[0][1]
    else:
        # Handle case when the second coefficient is zero
        yy = np.zeros_like(xx)  # Create a flat line (undefined slope)

    # Create subplot for each C value
    plt.subplot(2, 2, 1 + i)

    # Plot the support vectors
    plt.scatter(estimator.support_vectors_[:, 0], estimator.support_vectors_[:, 1],
                c="green", s=100, label="Support Vectors")

    # Plot the data points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')

    # Plot the decision boundary line
    plt.plot(xx, yy, 'k-', label="Decision Boundary")

    # Set plot title and formatting
    plt.title("$C = %g$" % C)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xticks([])
    plt.yticks([])
    plt.legend(loc="upper right")

# Show the final plot with all subplots
plt.tight_layout()  # Adjust layout to prevent overlapping subplots
plt.show()
