#Online implementation
#Used to make plots on overleaf
from Data import Data
import numpy as np
import cvxpy as cp
from generateDF import numNodes, tau, final_df
import time
import matplotlib.pyplot as plt

rng = np.random.default_rng(seed=1)

# Introduce parameters
period_min = 5 # 1 period is 5 min
beta = 1.25  # operational cost ($/period)
ei = np.ones(numNodes)
alpha = 0.2  # elasticity
gamma1 = 0.1 # regularization param
gamma2 = 0.01
eta = 0.02  # stepsize

# Simulation window
sim_window = 288 # periods
h_in = 0
min_in = 00

# Plot
profit = []
og_demand = []
eff_demand = []
satisf_demand = []

#Initialize L0, X0 and Y0
print("Initialize X0, Y0 and L0 at h", h_in)
X0 = np.array([[rng.integers(low=10, high=20) for n in range(numNodes)]for n in range(numNodes)])
print("X0\n", X0)
Y0 = np.diag([0 for i in range(numNodes)])
print("Y0\n", Y0)
L0 = np.diag([0 for i in range(numNodes)])
print("L0\n", L0)

max_riding_time = 0
for h in range(len(tau)):
    max_riding_time = max(np.max(tau[h]), max_riding_time)
max_riding_time = int(np.round(max_riding_time/(60*period_min)))

#Initialize time counter
counters = {}
for i in range(max_riding_time):
    counters[i] = rng.integers(low=10, high=20, size=numNodes)

for i in range(numNodes):
    for j in range(numNodes):
        travel_time = np.round(tau[h_in][i][j]/(60*period_min))
        if travel_time == 0:
            counters[travel_time][j] += X0[i][j]
        else:
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
    print("tau\n", np.round(tau[h] / (60*period_min)))
    # idling vehicles
    nu = counters[0]
    print("idling", nu)
    theta = data.avg_R(h, min, period_min=period_min)
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

    # Plots
    prof = np.trace(np.matmul(np.transpose(R0), Y0))
    profit.append(prof)
    og_demand.append(np.sum(theta))
    eff_demand.append(np.sum(R0))

    z = theta-alpha*Y0
    gradX = beta * np.round(tau[h] / (60*period_min)) - L0 + gamma1 * X0
    gradY = -z + gamma1 * Y0

    L0 = L0+eta*(z-X0-gamma2*L0)
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

    satis = np.diag([0.0 for k in range(numNodes)])
    for i in range(numNodes):
        for j in range(numNodes):
            if R0[i][j] <= X0_int[i][j]:
                satis[i][j] = R0[i][j]
            else:
                satis[i][j] = X0_int[i][j]
    satisf_demand.append(np.sum(satis))

    # Update counters
    for t in range(max_riding_time - 1):
        for i in range(numNodes):
            counters[t][i] = counters[t + 1][i]

    for i in range(numNodes):
        counters[max_riding_time - 1][i] = 0

    for i in range(numNodes):
        for j in range(numNodes):
            travel_time = np.round(tau[h][i][j]/(60*period_min))
            if travel_time == 0:
                counters[travel_time][j] += X0_int[i][j]
            else:
                counters[travel_time - 1][j] += X0_int[i][j]

    return X0, Y0, L0

start_time = time.time()

for t in range(sim_window):
    print("***Time", h_in, ":", min_in, "***Avg: ", period_min, " min")
    X0, Y0, L0 = SGD(h_in, min_in, X0, Y0, L0)

    if min_in != 55:
        min_in += period_min
    else:
        h_in += 1
        min_in = 0

    #Fleet size test
    n_veh_count = 0
    for i in range(len(counters)):
        n_veh_count += np.sum(counters[i])
    if n_veh_count != n_veh:
        raise Exception("number vehicles not conserved")
    else:
        print("fleet size", n_veh)

print("--- %s seconds ---" % (time.time() - start_time))

# Plots in the stochastic case with gradient knowledge.
# Profit here intended as effective demand times differential prices (ignore nominal prices)
fig, ax = plt.subplots()
ax.plot(profit)
ax.set(xlabel='periods (5 min)', ylabel='profit ($)')
fig.savefig("profit.png")

fig1, ax1 = plt.subplots()
ax1.plot(og_demand, label="original")
ax1.plot(eff_demand, label="effective")
ax1.plot(satisf_demand, label="satisfied")
ax1.set(xlabel='periods (5 min)', ylabel='demand (#)')
ax1.legend(loc="upper left")
fig1.savefig("demand.png")

plt.show()


