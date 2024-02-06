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
print("effective demand\n", theta - alpha*Y.value)
print("\nCVX results")
print("X\n", X.value)
print("Y\n", Y.value)
Xcvx = X.value
Ycvx = Y.value

#Compare with deterministic/stochastic primal-dual method
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
    gradY = -(theta - 2*alpha * Y0 - alpha*P) - alpha*L0 + gamma1 * Y0

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

    errorx = np.linalg.norm(X.value-Xcvx)
    errx.append(errorx)
    errory = np.linalg.norm(Y.value-Ycvx)
    erry.append(errory)

    return X.value, Y.value, L0


def primal_dual_stoch_known_grad(X0, Y0, L0):
    z = theta-alpha*Y0
    gradX = beta * omega - L0 + gamma1 * X0
    gradY = -z + gamma1 * Y0

    L0 = L0+10*eta*(z-X0-gamma2*L0)
    L0[L0 < 0] = 0

    Y = cp.Variable(5, "Y")
    X = cp.Variable(5, "X")

    obje = cp.sum_squares(X-X0+10*eta*gradX) + cp.sum_squares(Y-Y0+10*eta*gradY)
    constr = []
    constr += [cp.sum(X) <= N_vehicles]
    constr += [X >= 0]
    constr += [Y >= -10]
    constr += [Y <= 10]
    probl = cp.Problem(cp.Minimize(obje), constr)
    probl.solve()

    return X.value, Y.value, L0

def primal_dual_stoch(X1, Y1, L1, X0, Y0):
    z = theta-alpha*Y1 + np.random.randn(numNodes)
    gradX = beta * omega - L1 + gamma1 * X1
    gradY = -z + gamma1 * Y1

    L1 = L1+10*eta*(z-X1-gamma2*L1)
    L1[L1 < 0] = 0

    errorx = np.linalg.norm(X1-X0)
    errx.append(errorx)
    errory = np.linalg.norm(Y1-Y0)
    erry.append(errory)

    Y = cp.Variable(5, "Y")
    X = cp.Variable(5, "X")

    obje = cp.sum_squares(X-X1+10*eta*gradX) + cp.sum_squares(Y-Y1+10*eta*gradY)
    constr = []
    constr += [cp.sum(X) <= N_vehicles]
    constr += [X >= 0]
    constr += [Y >= -10]
    constr += [Y <= 10]
    probl = cp.Problem(cp.Minimize(obje), constr)
    probl.solve()

    return X.value, Y.value, L1

for i in range(n_it):
    # Online algo
    X0, Y0, L0 = primal_dual_stoch_known_grad(X0, Y0, L0)

print("\nprimal-dual results")
print("X", X0)
print("Y", Y0)

X1 = theta
Y1= np.zeros(numNodes)
L1 = np.zeros(numNodes)
for i in range(n_it):
    # Online algo
    X1, Y1, L1 = primal_dual_stoch(X1, Y1, L1, X0, Y0)

print("\nprimal-dual results")
print("X", X1)
print("Y", Y1)

fig, axs = plt.subplots(2)
axs[0].plot([x / numNodes for x in errx])
axs[1].plot([y / numNodes for y in erry])
axs[0].set_ylabel("err_x")
axs[1].set_ylabel("err_y")

plt.xlabel("minutes")
plt.show()




