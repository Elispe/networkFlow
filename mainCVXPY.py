#Use solver to get solution
from Data import Data
import numpy as np
import cvxpy as cp
from generateDF import numNodes, tau, final_df
import time
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

rng = np.random.default_rng(seed=1)

# Introduce parameters
beta = 0.01  # operational cost ($/min)
ei = np.ones(numNodes)
alpha = 0.2 # elasticity

max_riding_time=0
for h in range(len(tau)):
    max_riding_time = max(np.max(tau[h]), max_riding_time)
max_riding_time = max_riding_time//60

# Initialize time counter
counters = {}
for i in range(max_riding_time):
    counters[i] = rng.integers(low=20, high=100, size=numNodes)

n_veh = 0
# Fleet size
for i in range(len(counters)):
    n_veh += np.sum(counters[i])
print("fleet size:", n_veh)

data = Data(final_df, tau, numNodes)

def solver(h, min):
    # riding time
    print("tau\n", tau[h] // 60)

    # nominal price
    P = tau[h]//60*beta*200
    print("nominal price\n", P)

    # idling vehicles
    nu = counters[0]
    print("idling", nu)
    theta = data.draw_R(h, min)*10
    print("theta\n", theta)

    # for i in range(numNodes):
    #     for j in range(numNodes):
    #         if R0[i][j]<=0:
    #             R0[i][j]=0
    # print("R0\n", R0)

    Y = cp.Variable((numNodes, numNodes), "Y")
    X = cp.Variable((numNodes, numNodes), "X")

    obj = -cp.trace((theta - alpha * Y).T @ P + theta.T @ Y) + alpha * cp.sum_squares(Y) + beta * cp.trace(cp.transpose(tau[h]//60)@X)
    constraints = []
    constraints += [X @ ei == nu]
    constraints += [X >= 0]
    constraints += [X >= theta - alpha * Y]
    constraints += [Y >= -10]
    constraints += [Y <= 10]
    problem = cp.Problem(cp.Minimize(obj), constraints)
    problem.solve()
    print("problem status:", problem.status)

    R = theta - alpha * Y.value
    # Test: effective demand must be non-neg
    for i in range(numNodes):
        for j in range(numNodes):
            if R[i][j]<=0:
                raise Exception("negative effective demand")

    print("effective demand\n", R)
    print("X\n", X.value)
    print("Y\n", Y.value)

    # Generate integer actions
    X0_int = []
    a = np.arange(0, numNodes, 1, dtype=int)
    for row in X.value:
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

start_time = time.time()

sim_window = 60 #minutes
h_in = 18
min_in = 0
for minu in range(sim_window):
    print("***Time", h_in, ":", min_in)
    solver(h_in, min_in)
    if min_in != 59:
        min_in += 1
    else:
        h_in += 1
        min_in = 0

    # Fleet size test
    n_veh_count = 0
    for i in range(len(counters)):
        n_veh_count += np.sum(counters[i])
    if n_veh_count != n_veh:
        raise Exception("number vehicles not conserved")

print("--- %s seconds ---" % (time.time() - start_time))









