#Online implementation
from Data import Data
import numpy as np
import cvxpy as cp
from generateDF import numNodes, tau, final_df
import time
import random

rng = np.random.default_rng()

# starting price TO DO: educated guess?
Y0 = np.array([np.zeros(numNodes) for n in range(numNodes)])
for i in range(numNodes):
    for j in range(numNodes):
        Y0[i][j] = random.uniform(0, 1)

# Introduce parameters
beta = 0.04  # operational cost ($/min)
ei = np.ones(numNodes)
gamma = 0.1
eta = 0.1 # step size
alpha = 0.1 # elasticity
L0 = np.diag([0 for i in range(numNodes)])
X0 = np.array([[random.randint(0,2) for n in range(numNodes)]for n in range(numNodes)])
print("X0", X0)

max_riding_time=0
for h in range(len(tau)):
    max_riding_time = max(np.max(tau[h]), max_riding_time)
max_riding_time = max_riding_time//60

# Initialize time counter
counters = {}
for i in range(max_riding_time):
    counters[i] = np.array([random.randint(0,5) for i in range(numNodes)])

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

# deploy price policy Y0, observe R0
def SGD(h, min, X0, Y0, L0):
    # idling vehicles
    nu = counters[0]
    print("idling", nu)
    theta = data.draw_R(h, min)
    print("theta", theta)
    R0 = theta - alpha * Y0
    for i in range(numNodes):
        for j in range(numNodes):
            if R0[i][j]<=0:
                R0[i][j]=0
    print("R0", R0)
    print("tau", tau[h]//60)

    gammaX0 = [[xx*gamma for xx in r] for r in X0]
    gradX = beta * tau[h]//60 + L0 + gammaX0
    gradY = -R0 + gamma * Y0
    gradL = X0 - R0 - gamma * L0

    Y = cp.Variable((numNodes, numNodes), "Y")
    X = cp.Variable((numNodes, numNodes), "X")
    L = cp.Variable((numNodes, numNodes), "L")  # dual variable

    objX = cp.sum_squares(X - X0 + eta * gradX)
    constraintsX = []
    constraintsX += [X @ ei == nu]
    constraintsX += [X >= 0]
    problemX = cp.Problem(cp.Minimize(objX), constraintsX)
    problemX.solve()

    objY = cp.sum_squares(Y - Y0 + eta * gradY)
    constraintsY = []
    constraintsY += [Y >= 0]
    constraintsY += [Y <= 1]
    problemY = cp.Problem(cp.Minimize(objY), constraintsY)
    problemY.solve()

    objL = cp.sum_squares(L - L0 + eta * gradL)
    constraintsL = []
    constraintsL += [L >= 0]
    constraintsL += [L <= 1]
    problemL = cp.Problem(cp.Minimize(objL), constraintsL)
    problemL.solve()

    X0 = X.value
    Y0 = Y.value
    L0 = L.value
    print("X0", X0)
    print("Y0", Y0)

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
            p = row / size
            print("prob", p)
            res = rng.choice(a, int(np.round(size)), True, np.abs(p))
            for el in res:
                countEl[el] += 1
            X0_int.append(countEl)

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

    print("X0_int", X0_int)
    return X0_int, Y0, L0

start_time = time.time()

sim_window = 5*60 #minutes
h_in = 10
min_in = 1
for minu in range(sim_window):
    X0, Y0, L0 = SGD(h_in, min_in, X0, Y0, L0)
    if min_in != 59:
        min_in += 1
    else:
        h_in += 1
        min_in = 0

    # Fleet size test
    n_veh = 0
    for i in range(len(counters)):
        n_veh += np.sum(counters[i])
    print("fleet size:", n_veh)

print("--- %s seconds ---" % (time.time() - start_time))









