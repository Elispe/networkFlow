"""
Extracts ride request data from the Taxi and Limousine Commission, Manhattan, NY
Returns an average over stationary data (demand matrix) received within a specific time window (e.g. 10.00-10:05AM) across a time interval (e.g. a month), specified in generateDF.py
"""
import numpy as np
#rng = np.random.default_rng()

class Data:
    def __init__(self, df, tau, numNodes):
        self.df = df
        self.tau = tau
        self.numNodes = numNodes
    def avg_R(self, h_in, min_in, period_min):

        # Keep only ride requests received within 1 min from specified time
        mask_hour_min = ((self.df["tpep_pickup_datetime"].dt.hour == h_in) &
                         (self.df["tpep_pickup_datetime"].dt.minute >= min_in) &
                         (self.df["tpep_pickup_datetime"].dt.minute < min_in + period_min))

        final_min_df = self.df.loc[mask_hour_min]
        final_min_df = final_min_df.sort_values(by='tpep_pickup_datetime')
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

        # TO DO: Standardize data: vectorize, use scaler, go back to matrix form

        # Return a random element
        # R00_unscaled[rng.integers(0, len(R00_unscaled))]
        # TO DO: make sure you can reproduce results fixing a seed for rng

        # Return average over a certain amount of days, depends on dataset used. OBS! Adjust manually variable num_days depending on the month(s) considered!!
        # Enter total number of days over which the avg is made
        num_days = 31
        R00_exp = np.sum(R00_unscaled, axis=0)/num_days

        return R00_exp




