# Pricing and routing in electric mobility on demand: a stochastic optimization with decision-dependent data approach

Python code that implements a network flow model. Objective: maximize the profits of an AMoD operator. 

# Required Software

Python: https://www.python.org/downloads/

cvxpy: https://www.cvxpy.org/install/

# Data

Data for the ride requests can be downloaded from the Manhattan Taxi and Limousine Commission website https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page, selecting 2022 >> March >> Yellow Taxi Trip Records (PARQUET).

The .parquet file must be saved in the same folder as the main file, under the name <em>yellow_tripdata_2022-03.parquet</em>.

# Execution

Run the file <em>mainNoElectric.py</em> to consider the non-electric case.

The same folder should also contain:
<ul>
  <li>tripData.py: to extract a subsample of ride request data from the file <em>yellow_tripdata_2022-03.parquet</em></li>
  <li>taxiZones.cvs: taxi zone lookup table</li>
</ul>
