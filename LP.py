# Solve LP with CVXPY without considering the impact of prices on the demand
# number of vehicles is conserved
# no queue: missed rides are lost
# works when rides within same area are NOT considered (modify generateDF.py)
import cvxpy as cp
import numpy as np
from Data import Data
from generateDF import numNodes, tau, final_df
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
rng = np.random.default_rng(seed=1)

max_num_periods = 5 #to be checked (make it large enough)
counters = {}
for i in range(max_num_periods):
    counters[i] = np.array(np.zeros(numNodes))

#Need average demand R_bar and ride time tau at a given time t
n_veh = 20*numNodes
nu = [20 for i in range(numNodes)] #num vehicles at each node at the beginning (take 1200 EVs in tot for 18 areas)
ei = np.ones(numNodes)
period_min = 5 # period = 5 min
beta = 1.25 # $/period
data = Data(final_df, tau, numNodes)
penalty = 2.2 # for idling vehicles
pen = np.diag([penalty for i in range(numNodes)])

def opt_probl(h, min):
    R_bar = data.avg_R(h, min, period_min)
    tau_period = np.round(tau[h]/(period_min*60))

    # Optimization problem
    Y = cp.Variable((numNodes,numNodes), "Y")
    X = cp.Variable((numNodes,numNodes), "X")

    obj = cp.trace(R_bar.T@Y) - beta * cp.trace(tau_period.T@X) - cp.trace(X@pen)
    constraints = []
    constraints += [X >= R_bar]
    constraints += [X@ei == nu]
    constraints += [X >= 0]
    constraints += [Y <= 50]
    problem = cp.Problem(cp.Maximize(obj), constraints)
    problem.solve()

    print("arrival[#] \n", R_bar)
    print("tau[# periods]\n", tau_period)
    print("\nCVX results")
    print("X\n", X.value)
    #print("Y\n", Y.value)
    #print(problem.status)

    return X.value

sim_window = 20 # periods
h_in = 17
min_in = 40
for t in range(sim_window):
    print("***Time", h_in, ":", min_in, "***Avg: ", period_min, " min")
    Xopt = opt_probl(h_in, min_in)

    if min_in != 55:
        min_in += period_min
    else:
        h_in += 1
        min_in = 0

    # Generate integer-valued actions
    X_int = []
    a = np.arange(0, numNodes, 1, dtype=int)
    for row in Xopt:
        #print("row", row)
        size = np.sum(row)
        #print("size", size)
        if size < 0.1:
            X_int.append(np.zeros(numNodes, dtype=int))
        else:
            countEl = np.zeros(numNodes, dtype=int)
            p = row / np.linalg.norm(row)
            #print("prob", p*p)
            #print("sum prob", np.linalg.norm(p))
            res = rng.choice(a, int(np.round(size)), True, p*p)
            for el in res:
                countEl[el] += 1
            X_int.append(countEl)
    print("EV dispatch\n", X_int)
    nu = np.diag(X_int) + counters[0]
    print("EVs available at next iteration", nu)

    # Update counters
    for t in range(max_num_periods - 1):
        for i in range(numNodes):
            counters[t][i] = counters[t + 1][i]

    for i in range(numNodes):
        for j in range(numNodes):
            if i != j:
                travel_time = np.round(tau[h_in] / (period_min * 60))[i][j]
                counters[travel_time - 1][j] += X_int[i][j]

    # Test: vehicle number conservation
    if np.sum(nu) + np.sum([i for i in counters.values()]) == n_veh:
        print("Number of vehicles conserved")


