# Non-electric case, matrix formulation
import tripData as tData
import networkx as nx
import numpy as np
import cvxpy as cp

# import matplotlib.pyplot as plt

# Fleet size
n_veh = 9 * 20

# DiGraph - unweighted
g = nx.DiGraph()
elist = [(1, 2), (1, 3), (2, 1), (2, 3), (2, 4), (2, 5), (3, 1), (3, 2), (3, 4), (3, 5), (4, 2), (4, 3), (4, 5),
         (4, 6),
         (4, 7), (5, 2), (5, 3), (5, 4),
         (5, 6), (5, 7), (6, 4), (6, 5), (6, 7), (6, 8), (6, 9), (7, 4), (7, 6), (7, 8), (7, 9), (8, 6), (8, 7),
         (8, 9),
         (9, 6), (9, 7), (9, 8)]
# elist = [(1, 2), (1, 3), (2, 1), (2, 4), (3, 1), (3, 4), (4, 2), (4, 3)]
g.add_edges_from(elist)
numNodes = g.number_of_nodes()
# nx.draw(g)
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
    counters[i] = np.array([0] * numNodes)

total_missed_rides = 0


def optimize_xy(pu_loc, do_loc, x_idle):

    # Build demand matrix
    R = np.array([np.zeros(numNodes, dtype=int) for _ in range(numNodes)])
    for i in range(len(pu_loc)):
        if not pu_loc[i] == do_loc[i]:
            R[pu_loc[i] - 1][do_loc[i] - 1] += 1

    print("Demand matrix:")
    for ro in R:
        print(ro)

    # Solve using cvxpy
    ei = np.ones(numNodes)
    nu = x_idle + counters[0]
    print("Available vehicles at each node:", nu)
    Y = cp.Variable(R.shape, "Y")
    X = cp.Variable(S.shape, "X")
    objective = cp.Maximize(cp.trace(R.T@Y) - beta * cp.trace(S.T@X))
    constraints = []
    constraints += [R == X]
    constraints += [X @ ei == nu]
    constraints += [X >= 0]
    constraints += [Y >= 0]
    problem = cp.Problem(objective, constraints)
    problem.solve()

    print("status:", problem.status)
    print("Prices in $:", Y.value)
    print("X:", X.value)

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

    return x_idle


for k in range(tData.num_it):
    PULoc = tData.records[k][0]
    DOLoc = tData.records[k][1]

    print("*** Minute: " + str(k * tData.time_period) + " ***")
    print("total number vehicles:", np.round(sum(x_bar) + sum([sum(counters[i]) for i in range(len(counters))])))
    print("vehicles idling:", np.round(sum(x_bar), 2))
    print("vehicles riding:", np.round(sum([sum(counters[i]) for i in range(len(counters))]), 2))
    for i in range(np.max(S)):
        print("counter" + str(i), np.round(counters[i], 2))

    x_bar = optimize_xy(PULoc, DOLoc, x_bar)

