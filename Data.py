import numpy as np
#rng = np.random.default_rng()

class Data:
    def __init__(self, df, tau, numNodes):
        self.df = df
        self.tau = tau
        self.numNodes = numNodes
    def draw_R(self, h_in, min_in):

        # Keep only ride requests received within 1 min from specified time
        mask_hour_min = ((self.df["tpep_pickup_datetime"].dt.hour == h_in) &
                         (self.df["tpep_pickup_datetime"].dt.minute == min_in))

        final_min_df = self.df.loc[mask_hour_min]
        final_min_df.set_index("tpep_pickup_datetime", inplace=True)

        # Stationary data
        R00_unscaled = []
        theta = np.array([np.zeros(self.numNodes) for _ in range(self.numNodes)])
        begin = final_min_df.index[0].day
        for index, row in final_min_df.iterrows():
            if index.day == begin:
                theta[row["PUArea"] - 1][row["DOArea"] - 1] += 1
            else:
                R00_unscaled.append(theta)
                begin = index.day
                theta = np.array([np.zeros(self.numNodes) for _ in range(self.numNodes)])
                theta[row["PUArea"] - 1][row["DOArea"] - 1] += 1
        R00_unscaled.append(theta)

        # TO DO: Add null matrices corresponding to times with no requests
        # TO DO: Standardize data: vectorize, use scaler, go back to matrix form
        # TO DO: make sure you can reproduce results fixing a seed for rng

        #return R00_unscaled[rng.integers(0, len(R00_unscaled))]
        R00_exp = np.sum(R00_unscaled, axis=0)/len(R00_unscaled)

        return R00_exp




