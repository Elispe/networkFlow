# Non-electric case, matrix formulation
import tripData as tData
import networkx as nx
import numpy as np
import cvxpy as cp

# import matplotlib.pyplot as plt

# Fleet size
n_veh = 4 * 250

# DiGraph - unweighted
g = nx.DiGraph()
# elist = [(1, 2), (1, 3), (2, 1), (2, 3), (2, 4), (2, 5), (3, 1), (3, 2), (3, 4), (3, 5), (4, 2), (4, 3), (4, 5),
#          (4, 6),
#          (4, 7), (5, 2), (5, 3), (5, 4),
#          (5, 6), (5, 7), (6, 4), (6, 5), (6, 7), (6, 8), (6, 9), (7, 4), (7, 6), (7, 8), (7, 9), (8, 6), (8, 7),
#          (8, 9),
#          (9, 6), (9, 7), (9, 8)]
elist = [(1, 2), (1, 3), (2, 1), (2, 4), (3, 1), (3, 4), (4, 2), (4, 3)]
g.add_edges_from(elist)
numNodes = g.number_of_nodes()
# nx.draw(g, with_labels=True )
# import matplotlib.pyplot as plt
# plt.show()

x_bar = np.array([n_veh / numNodes for i in range(numNodes)])

# operational cost
beta = 2.5  # $

# Build path length matrix
print("Precomputed travel duration:")
S = np.array([np.zeros(numNodes, dtype=int) for n in range(numNodes)])
for orig in range(1, numNodes + 1):
    for destin in range(1, numNodes + 1):
        S[orig - 1][destin - 1] = len(nx.shortest_path(g, source=orig, target=destin)) - 1

for row in S:
    print(row)

counters = {}
for i in range(np.max(S)):
    counters[i] = np.array(np.zeros(numNodes))

total_missed_rides = 0

price_factor = 3

def optimize_xy(pu_loc, do_loc, x_idle):

    # Build demand matrix
    R = np.array([np.zeros(numNodes) for _ in range(numNodes)])
    R_scaled = np.array([np.zeros(numNodes) for _ in range(numNodes)])
    for i in range(len(pu_loc)):
        if not pu_loc[i] == do_loc[i]:
            R[pu_loc[i] - 1][do_loc[i] - 1] += 1

    print("Demand matrix:")
    for ro in R:
        print(ro)

    for m in range(numNodes):
        for n in range(numNodes):
            if R[m][n] != 0:
                R_scaled[m][n] = R[m][n] / (S[m][n] * beta * price_factor)

    print("R_scaled matrix:")
    for ro in R_scaled:
        print(ro)

    nu = x_idle + counters[0]
    print("Available vehicles at each node:", nu)

    # Solve using cvxpy
    ei = np.ones(numNodes)
    Y = cp.Variable(R.shape, "Y")
    X = cp.Variable(S.shape, "X")
    objective = cp.Maximize(cp.trace(R.T@Y) - cp.trace(R_scaled.T@cp.square(Y)) - beta * cp.trace(S.T@X))
    #objective = cp.Maximize(cp.trace(R.T @ Y) - beta * cp.trace(S.T @ X))
    constraints = []
    constraints += [R-cp.multiply(R_scaled, Y) <= X]
    #constraints += [R <= X]
    constraints += [X @ ei == nu]
    constraints += [X >= 0]
    constraints += [Y <= S * beta * price_factor]
    problem = cp.Problem(objective, constraints)
    problem.solve()

    print("status:", problem.status)
    print("Prices in $:\n", np.round(Y.value, 2))
    print("X:\n", np.round(X.value, 2))

    print("Effective Demand matrix:")
    for ro in R-np.multiply(R_scaled, Y.value):
        print(ro)

    x_idle = np.array([X.value[i][i] for i in range(numNodes)])

    # Update counters
    for t in range(np.max(S) - 1):
        for i in range(numNodes):
            counters[t][i] = counters[t + 1][i]

    for i in range(numNodes):
        counters[np.max(S) - 1][i] = 0

    for i in range(numNodes):
        for j in range(numNodes):
            if i != j:
                travel_time = S[i][j]
                counters[travel_time - 1][j] += X.value[i][j]

    print("Cost function:", np.round(objective.value, 2))

    # Missed requests
    miss_rides = 0
    for i in range(numNodes):
        for j in range(numNodes):
            if R[i][j] > X.value[i][j]:
                miss_rides += R[i][j] - X.value[i][j]
    print("Missed rides: " + str(miss_rides) + " out of " + str(np.sum(R)))
    return x_idle, miss_rides


for k in range(tData.num_it):
    PULoc = tData.records[k][0]
    DOLoc = tData.records[k][1]

    idling_ev = sum(x_bar)
    riding_ev = sum([sum(counters[i]) for i in range(len(counters))])
    print("total number vehicles:", np.round(idling_ev+riding_ev))
    print("vehicles idling:", np.round(idling_ev, 2))
    print("vehicles riding:", np.round(riding_ev, 2))
    for i in range(np.max(S)):
        print("counter" + str(i), np.round(counters[i], 2))

    # if idling_ev + riding_ev != n_veh:
    #     raise Exception('Number of vehicles not conserved!')

    print("*** Minute: " + str(k * tData.time_period) + " ***")

    x_bar, missed_rides = optimize_xy(PULoc, DOLoc, x_bar)
    total_missed_rides += missed_rides
print("---")
print("Total missed rides:", total_missed_rides)
print("QoS:", np.round(100 - (total_missed_rides / tData.numRequestsRed * 100), 2))

