# Network flow
import random

import tripData as tData
import networkx as nx
import numpy as np
import cvxpy as cp

#Fleet size
n_veh = 100

# DiGraph - unweighted
g = nx.DiGraph()
elist = [(1, 2), (1, 3), (2, 1), (2, 3), (2, 4), (2, 5), (3, 1), (3, 2), (3, 4), (3, 5), (4, 2), (4, 3), (4, 5),
         (4, 6),
         (4, 7), (5, 2), (5, 3), (5, 4),
         (5, 6), (5, 7), (6, 4), (6, 5), (6, 7), (6, 8), (6, 9), (7, 4), (7, 6), (7, 8), (7, 9), (8, 6), (8, 7),
         (8, 9),
         (9, 6), (9, 7), (9, 8)]
g.add_edges_from(elist)
numNodes = g.number_of_nodes()

#operational cost
beta = 2.5 #$

for k in range(tData.num_it):
    PULoc = tData.records[k][0]
    DOLoc = tData.records[k][1]
    numRideReq = len(PULoc)

    print("*** Minute: " + str(k * tData.time_period) + " ***")
    demand_matrix = [np.zeros(numNodes, dtype=int) for n in range(numNodes)]
    tau = [np.zeros(numNodes, dtype=int) for n in range(numNodes)]

    # Build demand matrix
    for i in range(numRideReq):
        demand_matrix[PULoc[i] - 1][DOLoc[i] - 1] += 1
        # Calculate shortest path
        tau[PULoc[i] - 1][DOLoc[i] - 1] = len(nx.shortest_path(g, source=PULoc[i], target=DOLoc[i])) - 1

    print("Demand matrix:")
    for row in demand_matrix:
        print(row)

    flat_demand_matrix = []
    for sublist in demand_matrix:
        for item in sublist:
            flat_demand_matrix.append(item)

    print("Path-length matrix:")
    for row in tau:
        print(row)

    flat_tau = []
    for sublist in tau:
        for item in sublist:
            flat_tau.append(item)

    #Solve using cvxpy
    scaled_flat_demand_matrix = []
    z_price = cp.Variable(numNodes ** 2)
    x = cp.Variable(numNodes ** 2)
    for i, item in enumerate(flat_demand_matrix):
        if(flat_tau[i] != 0):
            item = item / (flat_tau[i] * 2 * beta)
        scaled_flat_demand_matrix.append(item)
    objective = cp.Maximize(z_price @ flat_demand_matrix - cp.square(z_price) @ scaled_flat_demand_matrix - beta * (flat_tau @ x))
    constraints = []
    constraints += [x >= 0]
    constraints += [cp.sum(x) == n_veh]
    for i in range(numNodes ** 2):
        constraints += [z_price[i] <= flat_tau[i] * 2 * beta + 0.5]
        constraints += [flat_demand_matrix[i]*(1-z_price[i]/(flat_tau[i] * 2 * beta + 0.5)) <= x[i]]
    for i in range(numNodes):
        constraints += [cp.sum([x[i * numNodes + n] for n in range(numNodes)]) == cp.sum([x[i + numNodes * n] for n in range(numNodes)])]
    problem = cp.Problem(objective, constraints)
    problem.solve()

    print("Prices:" )
    for row in z_price.value.reshape(9, 9):
        print(np.round(row, 2))
    print("x:")
    for row in x.value.reshape(9, 9):
        print(np.round(row, 2))
        #print(np.sum(np.round(row, 2)))
    # print("x transpose:")
    # for row in x.value.reshape(9, 9).transpose():
    #     print(np.round(row, 2))
    #     print(np.sum(np.round(row, 2)))
    print("Cost function:\n", objective.value)