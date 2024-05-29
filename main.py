#Online implementation
from Data import Data
import numpy as np
import cvxpy as cp
from generateDF import numNodes, tau, final_df
import time
import matplotlib.pyplot as plt

rng = np.random.default_rng(seed=1)

# Introduce parameters
beta = 0.1  # operational cost ($/min)
ei = np.ones(numNodes)
alpha = 0.5  # elasticity
gamma1 = 0.1 # regularization param
gamma2 = 0.01
eta = 0.02  # stepsize

#Initialize L0, X0 and Y0
L0 = np.diag([0 for i in range(numNodes)])
print("L0\n", L0)
theta = np.array([[1.2, 0.8, 0.4], [0.3, 2.2, 0.4], [2.5, 0.5, 2.0]]);
#X0 = np.array([[rng.integers(low=10, high=30) for n in range(numNodes)]for n in range(numNodes)])
X0 = theta
print("X0\n", X0)
Y0 = np.diag([0 for i in range(numNodes)])
print("Y0\n", Y0)

max_riding_time = 0
for h in range(len(tau)):
    max_riding_time = max(np.max(tau[h]), max_riding_time)
max_riding_time = max_riding_time//60

#Initialize time counter
counters = {}
for i in range(max_riding_time):
    counters[i] = rng.integers(low=20, high=100, size=numNodes)

for i in range(numNodes):
    for j in range(numNodes):
        travel_time = tau[h][i][j]//60
        counters[travel_time - 1][j] += X0[i][j]

n_veh = 0
# Fleet size
for i in range(len(counters)):
    n_veh += np.sum(counters[i])
print("fleet size:", n_veh)

data = Data(final_df, tau, numNodes)

# deploy price policy Y0 at every iteration, observe R0
def SGD(h, min, X0, Y0, L0):
    # riding time
    print("tau\n", tau[h] // 60)
    # nominal price
    P = tau[h] // 60 * beta * 20
    print("nominal price\n", P)
    # idling vehicles
    nu = counters[0]
    print("idling", nu)
    theta = data.avg_R(h, min, period_min=1)
    print("theta\n", theta)

    print("**SGD results**")

    # Online algo
    R0 = theta - alpha * Y0
    for i in range(numNodes):
        for j in range(numNodes):
            if R0[i][j]<=0:
                R0[i][j] = 0
                print("Warning: negative effective demand")
    print(" effective demand\n", R0)

    z = theta-alpha*Y0
    gradX = beta * tau[h] // 60 - L0 + gamma1 * X0
    gradY = -z + gamma1 * Y0

    L0 = L0+10*eta*(z-X0-gamma2*L0)
    L0[L0 < 0] = 0

    Y = cp.Variable((numNodes, numNodes), "Y")
    X = cp.Variable((numNodes, numNodes), "X")

    obj = cp.sum_squares(X - X0 + eta * gradX) + cp.sum_squares(Y - Y0 + eta * gradY)
    constraints = []
    constraints += [X @ ei == nu]
    constraints += [X >= 0]
    constraints += [Y >= -10]
    constraints += [Y <= 10]
    problem = cp.Problem(cp.Minimize(obj), constraints)
    problem.solve()

    X0 = X.value
    Y0 = Y.value

    print("X\n", X0)
    print("Y\n", Y0)
    print("L\n", L0)

    # Generate integer actions
    X0_int = []
    a = np.arange(0, numNodes, 1, dtype=int)
    for row in X0:
        print("row", row)
        size = np.sum(row)
        print("size", size)
        if size < 0.1:
            X0_int.append(np.zeros(numNodes, dtype=int))
        else:
            countEl = np.zeros(numNodes, dtype=int)
            p = row / np.linalg.norm(row)
            print("prob", p*p)
            print("sum prob", np.linalg.norm(p))
            res = rng.choice(a, int(np.round(size)), True, p*p)
            for el in res:
                countEl[el] += 1
            X0_int.append(countEl)
    print("EV dispatch\n", X0_int)

    # Update counters
    for t in range(max_riding_time - 1):
        for i in range(numNodes):
            counters[t][i] = counters[t + 1][i]

    for i in range(numNodes):
        counters[max_riding_time - 1][i] = 0

    for i in range(numNodes):
        for j in range(numNodes):
            travel_time = tau[h][i][j]//60
            counters[travel_time - 1][j] += X0_int[i][j]

    return X0, Y0, L0

start_time = time.time()

sim_window = 5*60 #minutes
h_in = 18
min_in = 0
for minu in range(sim_window):
    print("***Time", h_in, ":", min_in)
    X0, Y0, L0 = SGD(h_in, min_in, X0, Y0, L0)
    if min_in != 59:
        min_in += 1
    else:
        h_in += 1
        min_in = 0

    #Fleet size test
    n_veh_count = 0
    for i in range(len(counters)):
        n_veh_count += np.sum(counters[i])
    if n_veh_count != n_veh:
        raise Exception("number vehicles not conserved")

print("--- %s seconds ---" % (time.time() - start_time))







