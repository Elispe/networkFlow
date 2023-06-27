# Network flow
import tripData as tData
import networkx as nx
import numpy as np
import cvxpy as cp

# import matplotlib.pyplot as plt

# Fleet size
n_veh = 100

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
# nx.draw(g)
# plt.show()

x_bar = [n_veh / numNodes for i in range(numNodes)]

# operational cost
beta = 2.5  # $
gamma = 1

# Build path length matrix
print("Precomputed travel duration:")
tau = [np.zeros(numNodes, dtype=int) for n in range(numNodes)]
for orig in range(1, numNodes + 1):
    for destin in range(1, numNodes + 1):
        tau[orig - 1][destin - 1] = len(nx.shortest_path(g, source=orig, target=destin)) - 1

for row in tau:
    print(row)

flat_tau = []
for sublist in tau:
    for item in sublist:
        flat_tau.append(item)

counter1 = [0 for i in range(numNodes)]
counter2 = [0 for i in range(numNodes)]

total_missed_rides = 0


def optimize_x(pu_loc, do_loc, x_idle):
    demand_matrix = [np.zeros(numNodes, dtype=int) for _ in range(numNodes)]

    # Build demand matrix
    for i in range(len(pu_loc)):
        if not pu_loc[i] == do_loc[i]:
            demand_matrix[pu_loc[i] - 1][do_loc[i] - 1] += 1

    print("Demand matrix:")
    for ro in demand_matrix:
        print(ro)

    flat_demand_matrix = []
    for sub in demand_matrix:
        for ite in sub:
            flat_demand_matrix.append(ite)

    # Solve using cvxpy
    scaled_flat_demand_matrix = []
    z_price = cp.Variable(numNodes ** 2)
    x = cp.Variable(numNodes ** 2)
    for i, it in enumerate(flat_demand_matrix):
        if flat_tau[i] != 0:
            it = it / (flat_tau[i] * 3 * beta)
        scaled_flat_demand_matrix.append(it)
    objective = cp.Maximize(
        z_price @ flat_demand_matrix - cp.square(z_price) @ scaled_flat_demand_matrix - beta * (flat_tau @ x)
        # - gamma * cp.sum([cp.square(flat_demand_matrix[i]-x[i]) for i in range(numNodes)]))
        - gamma * cp.sum([x[i] for i in range(0, numNodes ** 2, numNodes + 1)]))
    constraints = []
    constraints += [x >= 0]
    constraints += [cp.sum(x) == sum(x_idle) + sum(counter1)]
    for i in range(numNodes ** 2):
        constraints += [z_price[i] <= flat_tau[i] * 3 * beta]
        constraints += [flat_demand_matrix[i] * (1 - z_price[i] / (flat_tau[i] * 3 * beta)) <= x[i]]
    for i in range(numNodes):
        # constraints += [cp.sum([x[i * numNodes + n] for n in range(numNodes)]) == cp.sum([x[i + numNodes * n]
        # for n in range(numNodes)])]
        constraints += [cp.sum([x[i * numNodes + n] for n in range(numNodes)]) == x_idle[i] + counter1[i]]
    problem = cp.Problem(objective, constraints)
    problem.solve()

    print("Prices in $:")
    for ro in z_price.value.reshape(numNodes, numNodes):
        print(np.round(ro, 2))

    x_action = []
    print("x:")
    for ro in x.value.reshape(numNodes, numNodes):
        print(np.round(ro, 2))
        x_action.append(ro)

    x_idle = [x_action[i][i] for i in range(numNodes)]

    # Update counters
    for i in range(numNodes):
        counter1[i] = counter2[i]
        counter2[i] = 0
    for i in range(numNodes):
        for j in range(numNodes):
            if x_action[i][j] != 0 and i != j:
                if tau[i][j] == 1:
                    counter1[j] += x_action[i][j]
                else:
                    counter2[j] += x_action[i][j]

    # Missed requests
    miss_rides = 0
    for i in range(numNodes):
        for j in range(numNodes):
            if demand_matrix[i][j] > x_action[i][j]:
                miss_rides += demand_matrix[i][j] - x_action[i][j]
    print("Missed rides: " + str(int(miss_rides)) + " out of " + str(sum(flat_demand_matrix)))

    print("Cost function:", np.round(objective.value, 2))

    return x_idle, miss_rides


for k in range(tData.num_it):
    PULoc = tData.records[k][0]
    DOLoc = tData.records[k][1]

    print("*** Minute: " + str(k * tData.time_period) + " ***")
    print("currently idling", np.round(x_bar, 2))
    print("counter1", np.round(counter1, 2))
    print("counter2", np.round(counter2, 2))
    print("total number vehicles:", int(sum(x_bar) + sum(counter1) + sum(counter2)))
    x_bar, missed_rides = optimize_x(PULoc, DOLoc, x_bar)
    total_missed_rides += missed_rides

print("Total missed rides:", int(total_missed_rides))
