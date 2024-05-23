# Generate Tau matrix, and price matrix, averaged within the considered time interval
# Generate arrival matrix, count ride requests within the considered time interval

import pandas as pd
import datetime
import numpy as np

# Generate dataframe with columns: tpep_pickup_datetime, trip_duration, total_amount, PUArea, DOArea
trips_df = pd.read_parquet('yellow_tripdata_2019-05.parquet', engine='pyarrow')
# Remove not needed features
trips_df["trip_duration"] = trips_df["tpep_dropoff_datetime"] - trips_df["tpep_pickup_datetime"]
trips_df = trips_df.drop(
    ["VendorID", "passenger_count", "tpep_dropoff_datetime", "RatecodeID", "store_and_fwd_flag", "payment_type",
     "extra",
     "mta_tax", "tip_amount", "tolls_amount", "improvement_surcharge", "congestion_surcharge",
     "airport_fee", "trip_distance", "fare_amount"], axis=1)

# Read in locationID data
map_df = pd.read_csv('taxiZones.csv')
# Remove not needed features
map_df = map_df.drop("service_zone", axis=1)
manhattan_loc_df = map_df.loc[map_df['Borough'] == 'Manhattan']
manhattan_loc_df = manhattan_loc_df.drop("Borough", axis=1)
# Group zones into 18 areas
darea = {
    'LocationID': [12, 13, 261, 87, 88, 209, 125, 211, 144, 231, 45, 148, 232, 158, 249, 113, 114, 79, 4, 68,
                   90,
                   234, 107, 224, 246, 100, 186, 164, 170, 137, 50, 48, 230, 163, 161, 162, 229, 233, 143, 142,
                   239, 141, 140, 237, 24, 151, 238, 236, 263, 262, 166, 41, 74, 75, 42, 152, 116],
    'Area': [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 6, 6, 6, 7, 7, 8, 8, 8, 9, 9, 9, 10, 10,
             10,
             10, 11, 11, 11, 11, 12, 12, 12, 13, 13, 13, 14, 14, 14, 15, 15, 15, 16, 16, 17, 17, 18, 18, 18]}
area_df = pd.DataFrame(data=darea)
area_df = pd.merge(manhattan_loc_df, area_df, how='left', left_on=['LocationID'], right_on=['LocationID'])
area_df = area_df.loc[:, ['LocationID', 'Area']]
final_df = pd.merge(trips_df, area_df, how='left', left_on=['PULocationID'], right_on=['LocationID'])
final_df = final_df.drop('LocationID', axis=1)
final_df = pd.merge(final_df, area_df, how='left', left_on=['DOLocationID'], right_on=['LocationID'])
final_df = final_df.drop('LocationID', axis=1)
final_df = final_df.rename(columns={"Area_x": "PUArea", "Area_y": "DOArea"})
final_df = final_df.drop(["PULocationID", "DOLocationID"], axis=1)
maskNaN = (final_df['PUArea'] > 0) & (final_df['DOArea'] > 0)
final_df = final_df.loc[maskNaN]
max_trip_duration = final_df["trip_duration"] < datetime.timedelta(hours=3) #set limit on max trip duration
final_df = final_df.loc[max_trip_duration]
final_df['PUArea'] = final_df['PUArea'].astype('Int64')
final_df['DOArea'] = final_df['DOArea'].astype('Int64')

# Keep only trips within a selected number of areas in Manhattan
numNodes = 3
mask_areas = ((final_df['PUArea'] <= numNodes) & (final_df['DOArea'] <= numNodes))
            # & (final_df['PUArea'] != final_df['DOArea'])
final_df = final_df.loc[mask_areas]

# Keep only specific data within a time window
ini = datetime.datetime(2019, 5, 4, 18, 0, 0)
fin = datetime.datetime(2019, 5, 4, 20, 0, 0)
mask_areas = ((final_df['tpep_pickup_datetime'] >= ini) & (final_df['tpep_pickup_datetime'] <= fin))
final_df = final_df.loc[mask_areas]

# Generate Tau matrix, riding time from i to j, average within the considered time interval
# Generate price matrix, average within the considered time interval
# Generate arrival matrix, count ride requests within the considered time interval

tau = np.array([np.zeros(numNodes, dtype=int) for n in range(numNodes)])
price = np.array([np.zeros(numNodes, dtype=int) for n in range(numNodes)])
arrival = np.array([np.zeros(numNodes, dtype=int) for n in range(numNodes)])

for orig in range(1, numNodes + 1):
    for destin in range(1, numNodes + 1):
        mask_areas = (final_df['PUArea'] == orig) & (final_df['DOArea'] == destin)
        final_areas_df = final_df.loc[mask_areas]
        arr = (final_areas_df['trip_duration'] / np.timedelta64(1, 's')).to_numpy()
        tau[orig - 1][destin - 1] = np.average(arr)
        arr_price = final_areas_df['total_amount'].to_numpy()
        price[orig - 1][destin - 1] = np.average(arr_price)
        arrival[orig - 1][destin - 1] = len(arr)
print("tau[s]\n", tau)
print("price[$]\n", price)
print("arrival[#] \n", arrival)
lmax = 50



