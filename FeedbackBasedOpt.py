# Non-electric case, matrix formulation, feedback-based optimization
import tripData as tData
import networkx as nx
import numpy as np
import cvxpy as cp
import time
import matplotlib.pyplot as plt
start_time = time.time()

# Fleet size
n_veh = 9 * 100

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
# nx.draw(g, with_labels=True )
# import matplotlib.pyplot as plt
# plt.show()

x_bar = np.array([n_veh / numNodes for i in range(numNodes)])

# operational cost
beta = 1  # $

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
n_iter = 300
err = []

def optimize_xy(pu_loc, do_loc, x_idle):

    # Build original demand matrix
    theta = np.array([np.zeros(numNodes) for _ in range(numNodes)])
    for i in range(len(pu_loc)):
        if not pu_loc[i] == do_loc[i]:
            theta[pu_loc[i] - 1][do_loc[i] - 1] += 1

    print("Demand matrix:")
    for ro in theta:
        print(ro)

    nu = x_idle + counters[0]
    print("Available vehicles at each node:", nu)

    # Solve using cvxpy for N iterations
    ei = np.ones(numNodes)
    gamma = 0.1
    eta = 0.1
    alpha = 0.1
    X0 = np.diag([n_veh/numNodes for i in range(numNodes)])
    L0 = np.diag([0 for i in range(numNodes)])
    Y0 = S
    R0 = theta

    for i in range(n_iter):
        count = 0

        print(" ***Iteration number", i)
        gradX = beta * S + L0 + gamma * X0
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
        problemY = cp.Problem(cp.Minimize(objY), constraintsY)
        problemY.solve()

        objL = cp.sum_squares(L - L0 + eta * gradL)
        constraintsL = []
        constraintsL += [L >= 0]
        constraintsL += [L <= 2]
        problemL = cp.Problem(cp.Minimize(objL), constraintsL)
        problemL.solve()

        # R = cp.Variable((numNodes, numNodes), "R")
        # objR = - alpha * cp.trace(R.T @ Y0) + 1 / 2 * cp.sum_squares(R - R0)
        # constraintsR = []
        # constraintsR += [R <= R0]
        # constraintsR += [R >= 0]
        # problemR = cp.Problem(cp.Minimize(objR), constraintsR)
        # problemR.solve()
        #
        # print("statusR:", problemR.status)
        # print("R:\n", np.round(R.value, 2))

        # print("statusX:", problemX.status)
        # print("statusY:", problemY.status)
        # print("statusL:", problemL.status)
        #print("Prices in $:\n", np.round(Y.value, 2))
        print("X:\n", np.round(X.value, 1))
        #print("Lambda:\n", np.round(L.value, 2))
        print("R0:\n", np.round(R0, 2))

        error = np.linalg.norm(X.value-X0)**2 + np.linalg.norm(Y.value-Y0)**2 + np.linalg.norm(L.value-L0)**2
        print("Error", error)
        err.append(error)

        X0=X.value
        Y0=Y.value
        L0=L.value
        R0=theta-alpha*Y0

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

    print("Cost function:", np.round(objX.value, 2))

    # Missed requests
    miss_rides = 0
    for i in range(numNodes):
        for j in range(numNodes):
            if theta[i][j] > X.value[i][j]:
                miss_rides += theta[i][j] - X.value[i][j]
    print("Missed rides: " + str(miss_rides) + " out of " + str(np.sum(theta)))

    # Test
    # Check that the demand is satisfied, ie the constraint X >= R is achieved.
    for ii in range(numNodes):
        for jj in range(numNodes):
            if X0[ii][jj] < R0[ii][jj]:
                count += 1
    print("constraint violated", count)

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
print("--- %s seconds ---" % (time.time() - start_time))
print("Total missed rides:", total_missed_rides)
print("QoS:", np.round(100 - (total_missed_rides / tData.numRequestsRed * 100), 2))

# Data for plotting
plt.yscale("log")
plt.plot(err)
plt.xlabel("n_iter")
plt.ylabel("error")

plt.show()
