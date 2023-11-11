# Merge taxi data month by month.
# Generate .parquet file.

import pyarrow.parquet as pq

files = ["yellow_tripdata_2022-01.parquet", "yellow_tripdata_2022-02.parquet", "yellow_tripdata_2022-03.parquet"]

with pq.ParquetWriter("yellow_tripdata_2022.parquet", schema=pq.ParquetFile(files[0]).schema_arrow) as writer:
    for file in files:
        writer.write_table(pq.read_table(file))
