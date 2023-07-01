"""
Extract ride request data from the Taxi and Limousine Commission, Manhattan, NY
Choose the size of a subsample for the time window of interest
"""
import pandas as pd
import numpy as np
import datetime

np.random.seed(8)

# Read in yellow taxi data (March 2022) and convert into a Pandas Dataframe
trips_df = pd.read_parquet('yellow_tripdata_2022-03.parquet', engine='pyarrow')

# Remove not needed features
trips_df["trip_duration"] = trips_df["tpep_dropoff_datetime"] - trips_df["tpep_pickup_datetime"]
trips_df = trips_df.drop(
    ["VendorID", "passenger_count", "tpep_dropoff_datetime", "RatecodeID", "store_and_fwd_flag", "payment_type",
     "extra",
     "mta_tax", "tip_amount", "tolls_amount", "total_amount", "improvement_surcharge", "congestion_surcharge",
     "airport_fee"], axis=1)

# Choose which days you want the data for
yyyy = "2022"
mm = "03"
dd_in = "1"
dd_fin = "2"
mask = (trips_df['tpep_pickup_datetime'] >= "-".join([yyyy, mm, dd_in])) & \
       (trips_df['tpep_pickup_datetime'] < "-".join([yyyy, mm, dd_fin]))
selected_trips = trips_df.loc[mask]
ordered_trips = selected_trips.sort_values(by="tpep_pickup_datetime")

# Read in locationID data
map_df = pd.read_csv('taxiZones.csv')

# Remove not needed features
map_df = map_df.drop("service_zone", axis=1)
manhattan_loc_df = map_df.loc[map_df['Borough'] == 'Manhattan']
manhattan_loc_df = manhattan_loc_df.drop("Borough", axis=1)

# Group zones into 18 areas
darea = {'LocationID': [12, 13, 261, 87, 88, 209, 125, 211, 144, 231, 45, 148, 232, 158, 249, 113, 114, 79, 4, 68, 90,
                        234, 107, 224, 246, 100, 186, 164, 170, 137, 50, 48, 230, 163, 161, 162, 229, 233, 143, 142,
                        239, 141, 140, 237, 24, 151, 238, 236, 263, 262, 166, 41, 74, 75, 42, 152, 116],
         'Area': [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 6, 6, 6, 7, 7, 8, 8, 8, 9, 9, 9, 10, 10, 10,
                  10, 11, 11, 11, 11, 12, 12, 12, 13, 13, 13, 14, 14, 14, 15, 15, 15, 16, 16, 17, 17, 18, 18, 18]}
area_df = pd.DataFrame(data=darea)
area_df = pd.merge(manhattan_loc_df, area_df, how='left', left_on=['LocationID'], right_on=['LocationID'])
area_df = area_df.loc[:, ['LocationID', 'Area']]
final_df = pd.merge(ordered_trips, area_df, how='left', left_on=['PULocationID'], right_on=['LocationID'])
final_df = final_df.drop('LocationID', axis=1)
final_df = pd.merge(final_df, area_df, how='left', left_on=['DOLocationID'], right_on=['LocationID'])
final_df = final_df.drop('LocationID', axis=1)
final_df = final_df.rename(columns={"Area_x": "PUArea", "Area_y": "DOArea"})
final_df = final_df.drop(["PULocationID", "DOLocationID"], axis=1)
maskNaN = (final_df['PUArea'] > 0) & (final_df['DOArea'] > 0)
final_df = final_df.loc[maskNaN]
final_df['PUArea'] = final_df['PUArea'].astype('Int64')
final_df['DOArea'] = final_df['DOArea'].astype('Int64')

# Keep only trips within a selected number of areas in Manhattan
mask_areas = (final_df['PUArea'] <= 9) & (final_df['DOArea'] <= 9)
final_areas_df = final_df.loc[mask_areas]
# Show the features in dataset along datatype
# final_areas_df.info()

# Keep only ride requests received within the time window specified below. Choose the time period.
h_in = 6
time_period = 5  # minutes
sim_duration = 60 * 18  # minutes
ini = datetime.datetime(int(yyyy), int(mm), int(dd_in), h_in, 0, 0)
delta_min = datetime.timedelta(seconds=time_period * 60)
num_it = int(sim_duration / time_period)

records = {}
numRequestsRed = 0

# Subsample the ride requests. Choose scaling factor below
scalingFactor = 1
for i in range(num_it):
    mask_areas_min = (final_areas_df['tpep_pickup_datetime'] >= ini) & (
            final_areas_df['tpep_pickup_datetime'] < ini + delta_min)
    final_areas_min = final_areas_df.loc[mask_areas_min]
    numRequests = len(final_areas_min)
    PU_arr = final_areas_min['PUArea'].tolist()
    DO_arr = final_areas_min['DOArea'].tolist()

    print(str(numRequests) + " requests between " + str(ini) + " and " + str(ini + delta_min))
    print("Keep: " + str(round(numRequests / scalingFactor)))

    numRequestsRed += round(numRequests / scalingFactor)

    # Scale down the number of requests/minute by scalingFactor
    index = []
    while len(index) < round(numRequests / scalingFactor):
        rand_num = np.random.randint(0, numRequests)
        if rand_num not in index:
            index.append(rand_num)
    index.sort()

    PUReduced = []
    DOReduced = []
    for j in index:
        PUReduced.append(PU_arr[j])
        DOReduced.append(DO_arr[j])

    records[i] = [PUReduced, DOReduced]
    ini = ini + delta_min

print("Total number riding requests:", numRequestsRed)
