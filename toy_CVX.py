# Reproduce Matlab code "Price_allocation_last.m"
# (of course the random vector is different, so results are slightly different for the stochastic case "primal_dual_stoch")
# 1D case
import cvxpy as cp
import numpy as np
import warnings
import matplotlib.pyplot as plt
warnings.simplefilter(action='ignore', category=FutureWarning)

numNodes = 5
P = np.array([30, 40, 30, 70, 70]);
delta_min = -10;
delta_max = 10;
N_vehicles = 100;
theta = np.array([12, 8, 4, 25, 20]);
alpha = 1;
beta = .1;
omega = np.array([1, 3, 2, 5, 5]);

Y = cp.Variable(numNodes, "Y")
X = cp.Variable(numNodes, "X")

obj = -(theta - alpha * Y) @ (P) - theta @ (Y) + alpha * cp.sum_squares(Y) + beta * (omega @ X)
constraints = []
constraints += [cp.sum(X) <= N_vehicles]
constraints += [X >= 0]
constraints += [X >= theta - alpha * Y]
constraints += [Y >= -10]
constraints += [Y <= 10]
problem = cp.Problem(cp.Minimize(obj), constraints)
problem.solve()

print("theta\n", theta)
print("\nCVX results")
print("effective demand\n", theta - alpha*Y.value)
print("X\n", X.value)
print("Y\n", Y.value)
Xcvx = X.value
Ycvx = Y.value

#Compare deterministic/stochastic primal-dual method
n_it = 2000
eta = 0.02 # stepsize
gamma1 = 0.1
gamma2 = 0.01

#Initialize variables
X0 = theta
Y0 = np.zeros(numNodes)
L0 = np.zeros(numNodes)
errx = []
erry = []
def primal_dual_determ(X0, Y0, L0):
    gradX = beta * omega - L0 + gamma1 * X0
    gradY = -(theta - 2*alpha * Y0 - alpha*P) - L0 + gamma1 * Y0 # In Matlab
    #gradY = -(theta - 2*alpha * Y0 - alpha*P) - alpha*L0 + gamma1 * Y0 # I think this is the correct expression

    Y = cp.Variable(5, "Y")
    X = cp.Variable(5, "X")

    obje = cp.sum_squares(X-X0+eta*gradX) + cp.sum_squares(Y-Y0+eta*gradY)
    constr = []
    constr += [cp.sum(X) <= N_vehicles]
    constr += [X >= 0]
    constr += [Y >= -10]
    constr += [Y <= 10]
    probl = cp.Problem(cp.Minimize(obje), constr)
    probl.solve()

    gradL = theta - alpha*Y.value - X.value - gamma2*L0
    L0 = L0 + eta*gradL
    L0[L0 < 0] = 0

    # Uncomment to get error of  deterministic case wrt cvx
    # Must comment error part from function primal_dual_stoch
    # errorx = np.linalg.norm(X.value-Xcvx)
    # errx.append(errorx)
    # errory = np.linalg.norm(Y.value-Ycvx)
    # erry.append(errory)

    return X.value, Y.value, L0


def primal_dual_stoch_known_grad(X0, Y0, L0):
    z = theta-alpha*Y0
    gradX = beta * omega - L0 + gamma1 * X0
    #gradY = -z + gamma1 * Y0 # this is the correct expression I think
    gradY = -z -L0 + gamma1 * Y0 # this is in Matlab

    L0 = L0+eta*(z-X0-gamma2*L0)
    L0[L0 < 0] = 0

    Y = cp.Variable(5, "Y")
    X = cp.Variable(5, "X")

    obje = cp.sum_squares(X-X0+eta*gradX) + cp.sum_squares(Y-Y0+eta*gradY)
    constr = []
    constr += [cp.sum(X) <= N_vehicles]
    constr += [X >= 0]
    constr += [Y >= -10]
    constr += [Y <= 10]
    probl = cp.Problem(cp.Minimize(obje), constr)
    probl.solve()

    return X.value, Y.value, L0

def primal_dual_stoch(X1, Y1, L1, X0, Y0):
    eta = 0.05 #stepsize
    z = theta-alpha*Y1 + np.random.randn(numNodes)
    gradX = beta * omega - L1 + gamma1 * X1
    gradY = -z + gamma1 * Y1

    L1 = L1+eta*(z-X1-gamma2*L1)
    L1[L1 < 0] = 0

    errorx = np.linalg.norm(X1-X0)
    errx.append(errorx)
    errory = np.linalg.norm(Y1-Y0)
    erry.append(errory)

    Y = cp.Variable(5, "Y")
    X = cp.Variable(5, "X")

    obje = cp.sum_squares(X-X1+eta*gradX) + cp.sum_squares(Y-Y1+eta*gradY)
    constr = []
    constr += [cp.sum(X) <= N_vehicles]
    constr += [X >= 0]
    constr += [Y >= -10]
    constr += [Y <= 10]
    probl = cp.Problem(cp.Minimize(obje), constr)
    probl.solve()

    return X.value, Y.value, L1

#Consider deterministic case
X2 = theta
Y2= np.zeros(numNodes)
L2 = np.zeros(numNodes)
for i in range(n_it):
    X2, Y2, L2 = primal_dual_determ(X2, Y2, L2)

print("\ndeterm results")
print("effective demand\n", theta - alpha*Y2)
print("X", X2)
print("Y", Y2)
print("L", L2)

#Consider stochastic case, with/without gradient knowledge
for i in range(n_it):
    X0, Y0, L0 = primal_dual_stoch_known_grad(X0, Y0, L0)

print("\nstoch, known grad results")
print("effective demand\n", theta - alpha*Y0)
print("X", X0)
print("Y", Y0)
print("L", L0)

X1 = theta
Y1= np.zeros(numNodes)
L1 = np.zeros(numNodes)
for i in range(n_it):
    X1, Y1, L1 = primal_dual_stoch(X1, Y1, L1, X0, Y0)

print("\nstoch, unknown grad results")
print("effective demand\n", theta - alpha*Y1)
print("X", X1)
print("Y", Y1)
print("L", L1)

# Plot error e.g. now stochastic case with gradient knowledge vs without gradient knowledge
fig, axs = plt.subplots(2)
axs[0].plot([x / numNodes for x in errx])
axs[1].plot([y / numNodes for y in erry])
axs[0].set_ylabel("err_x")
axs[1].set_ylabel("err_y")

plt.xlabel("minutes")
plt.show()




