from google.cloud import bigquery
import pandas as pd
from bq_helper import BigQueryHelper



def main():
    noaa_goes = BigQueryHelper(active_project= "bigquery-public-data", dataset_name = "noaa_goes16");
    noaa_goes.list_tables();

    noaa_goes.table_schema('abi_l1b_radiance')
    print(noaa_goes.head("abi_l1b_radiance", num_rows=10))

    query = """
    SELECT dataset_name, platform_id, scene_id FROM `bigquery-public-data.noaa_goes16.abi_l1b_radiance` WHERE geospatial_westbound_longitude<120 and geospatial_eastbound_longitude>75 and geospatial_northbound_latitude<50 and geospatial_southbound_latitude>30
    """

    print("Query size in GB is %f " % noaa_goes.estimate_query_size(query))
    
main()
