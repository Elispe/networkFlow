import cvxpy as cp
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

P = np.array([30, 40, 30, 70, 70]);
delta_min = -10;
delta_max = 10;

N_vehicles = 100;

theta = np.array([12, 8, 4, 25, 20]);
alpha = 0.2;
beta = .01;
omega = np.array([1, 3, 2, 5, 5]);

Y = cp.Variable(5, "Y")
X = cp.Variable(5, "X")

obj = -(theta - alpha * Y) @ (P) - theta @ (Y) + alpha * cp.sum_squares(Y) + beta * (omega @ X)
constraints = []
constraints += [cp.sum(X) <= N_vehicles]
constraints += [X >= 0]
constraints += [X >= theta - alpha * Y]
constraints += [Y >= -10]
constraints += [Y <= 10]
problem = cp.Problem(cp.Minimize(obj), constraints)
problem.solve()
print(problem.status)

print("theta\n", theta)
print("effective demand\n", theta - alpha*Y.value)
print("X\n", X.value)
print("Y\n", Y.value)
print("optimal value", problem.value)
