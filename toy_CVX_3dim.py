# 3-dim toy example
import cvxpy as cp
import numpy as np
import warnings
import matplotlib.pyplot as plt
warnings.simplefilter(action='ignore', category=FutureWarning)

# Write False below for the deterministic case
is_stochastic = True
N_vehicles = 100;
numNodes = 3
ei = np.ones(numNodes)
nu = [N_vehicles for i in range(numNodes)]

P = np.array([[30, 40, 30], [20, 40, 10], [30, 70, 60]]);
delta_min = -10;
delta_max = 10;

theta = np.array([[12, 8, 4],[3, 22, 4], [25, 5, 20]]);
alpha = 1;
beta = 0.1;
omega = np.array([[1, 3, 2], [2, 4, 4], [3, 5, 7]]);

Y = cp.Variable((numNodes, numNodes), "Y")
X = cp.Variable((numNodes, numNodes), "X")

obj = cp.trace(-(theta - alpha * Y).T @ (P) - theta.T @ (Y) + beta * (omega.T @ X)) + alpha * cp.sum_squares(Y)
constraints = []
constraints += [X @ ei <= nu]
constraints += [X >= 0]
constraints += [X >= theta - alpha * Y]
constraints += [Y >= -10]
constraints += [Y <= 10]
problem = cp.Problem(cp.Minimize(obj), constraints)
problem.solve()

print("theta\n", theta)
print("effective demand\n", theta - alpha*Y.value)
print("X_cvx\n", X.value)
print("Y_cvx\n", Y.value)
Xcvx = X.value
Ycvx = Y.value

n_it = 1000
eta = 0.02 # stepsize
gamma1 = 0.1
gamma2 = 0.01
#Initialize variables
X0 = theta
Y0 = np.zeros((numNodes,numNodes))
L0 = np.zeros((numNodes,numNodes))
errx = []
erry = []

#Compare with deterministic primal-dual method
def primal_dual_determ(X0, Y0, L0):

    Y = cp.Variable((numNodes, numNodes), "Y")
    X = cp.Variable((numNodes, numNodes), "X")

    gradX = beta * omega - L0 + gamma1 * X0
    gradY = -(theta - 2*alpha * Y0 - alpha*P) - alpha*L0 + gamma1 * Y0

    obje = cp.sum_squares(X-X0+eta*gradX) + cp.sum_squares(Y-Y0+eta*gradY)
    constr = []
    constr += [X @ ei <= nu]
    constr += [X >= 0]
    constr += [Y >= -10]
    constr += [Y <= 10]
    probl = cp.Problem(cp.Minimize(obje), constr)
    probl.solve()

    gradL = theta - alpha*Y.value - X.value - gamma2*L0
    L0 = L0 + eta*gradL
    L0[L0 < 0] = 0

    errorx = np.linalg.norm(X.value-Xcvx)
    errx.append(errorx)
    errory = np.linalg.norm(Y.value-Ycvx)
    erry.append(errory)

    return X.value, Y.value, L0

#Compare with stochastic primal-dual method
def primal_dual_stoch_known_grad(X0, Y0, L0):
    z = theta-alpha*Y0
    gradX = beta * omega - L0 + gamma1 * X0
    gradY = -z + gamma1 * Y0

    L0 = L0+10*eta*(z-X0-gamma2*L0)
    L0[L0 < 0] = 0

    Y = cp.Variable((numNodes, numNodes), "Y")
    X = cp.Variable((numNodes, numNodes), "X")

    obje = cp.sum_squares(X-X0+10*eta*gradX) + cp.sum_squares(Y-Y0+10*eta*gradY)
    constr = []
    constr += [X @ ei <= nu]
    constr += [X >= 0]
    constr += [Y >= -10]
    constr += [Y <= 10]
    probl = cp.Problem(cp.Minimize(obje), constr)
    probl.solve()

    return X.value, Y.value, L0

def primal_dual_stoch(X1, Y1, L1, X0, Y0):
    z = theta-alpha*Y1 + np.random.randn(numNodes,numNodes)
    gradX = beta * omega - L1 + gamma1 * X1
    gradY = -z + gamma1 * Y1

    L1 = L1+10*eta*(z-X1-gamma2*L1)
    L1[L1 < 0] = 0

    errorx = np.linalg.norm(X1-X0)
    errx.append(errorx)
    errory = np.linalg.norm(Y1-Y0)
    erry.append(errory)

    Y = cp.Variable((numNodes, numNodes), "Y")
    X = cp.Variable((numNodes, numNodes), "X")

    obje = cp.sum_squares(X-X1+10*eta*gradX) + cp.sum_squares(Y-Y1+10*eta*gradY)
    constr = []
    constr += [X @ ei <= nu]
    constr += [X >= 0]
    constr += [Y >= -10]
    constr += [Y <= 10]
    probl = cp.Problem(cp.Minimize(obje), constr)
    probl.solve()

    return X.value, Y.value, L1

for i in range(n_it):
    # Online algo
    if is_stochastic:
        X0, Y0, L0 = primal_dual_stoch_known_grad(X0, Y0, L0)
    else:
        X0, Y0, L0 = primal_dual_determ(X0, Y0, L0)

print("\nResults", end=" ")
if is_stochastic:
    print("stochastic case, known grad")
else:
    print("deterministic case")
print("X\n", X0)
print("Y\n", Y0)

if is_stochastic:
    X1 = theta
    Y1 = np.zeros((numNodes,numNodes))
    L1 = np.zeros((numNodes,numNodes))
    for i in range(n_it):
        # Online algo
        X1, Y1, L1 = primal_dual_stoch(X1, Y1, L1, X0, Y0)

    print("\nstochastic case, unknown grad")
    print("X", X1)
    print("Y", Y1)

fig, axs = plt.subplots(2)
axs[0].plot([x / numNodes for x in errx])
axs[1].plot([y / numNodes for y in erry])
axs[0].set_ylabel("err_x")
axs[1].set_ylabel("err_y")

plt.xlabel("minutes")
plt.show()



