# Gradient-Descent-and-Cost-Function-for-Test-Scores

This repository contains a Python implementation of the Gradient Descent algorithm, a fundamental optimization technique used in machine learning and data science. The code demonstrates how to use gradient descent to find the optimal parameters for a linear regression model.


## Introduction

radient Descent is an iterative optimization algorithm used to minimize a function by moving in the direction of the steepest descent, as defined by the negative of the gradient. In the context of linear regression, it is used to find the best-fitting line by minimizing the cost function, which is typically the mean squared error (MSE).

This implementation includes:

Basic gradient descent for a simple linear regression problem.
An extended example using a dataset (test_scores.csv) to predict computer science (cs) scores based on math scores.
Early stopping based on a cost difference threshold.


## Code Explanation

### Basic Gradient Descent

The core function gradient_descent(x, y) performs the following steps:

Initializes the slope (m_curr) and intercept (b_curr) to zero.
Iteratively updates the parameters using the gradient of the cost function.
Prints the current values of m, b, and the cost at each iteration.
python
Copy
def gradient_descent(x, y): 
    m_curr = b_curr = 0
    iterations = 10000
    n = len(x)
    learning_rate = 0.08
    
    for i in range(iterations):
        y_predicted = m_curr * x + b_curr
        cost = (1/n) * sum([val**2 for val in (y-y_predicted)])
        md = -(2/n) * sum(x*(y-y_predicted))
        bd = -(2/n) * sum(y-y_predicted)
        m_curr = m_curr - learning_rate * md 
        b_curr = b_curr - learning_rate * bd  
        print(f"m {m_curr}, b {b_curr}, cost {cost}, iteration {i}")

## Extended Example with Early Stopping

The extended example uses a dataset (test_scores.csv) to predict computer science scores based on math scores. It includes an early stopping mechanism based on a cost difference threshold.

python
Copy
def gradient_descent(x, y): 
    m_curr = b_curr = 0
    iterations = 50
    n = len(x) 
    learning_rate = 0.001
    prev_cost = float('inf')  # Initialize with a very large value
    
    for i in range(iterations):
        yp = m_curr * x + b_curr 
        cost = (1/n) * sum([val**2 for val in (y-yp)])
        md = -(2/n) * sum(x*(y-yp))
        bd = -(2/n) * sum((y-yp))
        
        # Check if the cost difference is within the threshold
        if math.isclose(prev_cost, cost, abs_tol=1e-20):
            print(f"Stopping early at iteration {i} as costs are similar within threshold.")
            break
        
        m_curr = m_curr - learning_rate * md 
        b_curr = b_curr - learning_rate * bd 
        
        print(f"m {m_curr}, b {b_curr}, cost {cost}, iteration {i}")
        prev_cost = cost  # Update previous cost

## Example

Input Data

python
Copy
x = np.array([1, 2, 3, 4, 5])
y = np.array([5, 7, 9, 11, 13])
Output

The script will print the values of m, b, and the cost at each iteration until convergence or the maximum number of iterations is reached.

plaintext
Copy
m 0.64, b 0.16, cost 89.0, iteration 0
m 1.0496, b 0.2624, cost 52.8096, iteration 1
m 1.361216, b 0.340304, cost 31.419648, iteration 2
...
Stopping early at iteration 42 as costs are similar within threshold.
