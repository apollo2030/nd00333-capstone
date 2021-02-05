# The script MUST contain a function named azureml_main
# which is the entry point for this module.

# imports up here can be used to
import pandas as pd
import datetime

def distribute_per_week(start, end, column_name, country):
    index = pd.date_range(start, end, freq="W").strftime('%Y-%U')
    df = index.to_frame(name='week')
    df['country'] = country
    df = df.set_index(['country','week'])
    df[column_name] = True
    return df 

def distribute_per_weeknumbers(row):
    """Gets a dataframe row and converts it into a time series dataframe binned in weeks"""
    date_end = row['date_end'] if row['date_end'] != 'NA' else datetime.date.today()
    df = distribute_per_week(row['date_start'], date_end, row['Response_measure'], row['Country'])
    return df

# The entry point function MUST have two input arguments.
# If the input port is not connected, the corresponding
# dataframe argument will be None.
#   Param<dataframe1>: a pandas.DataFrame
#   Param<dataframe2>: a pandas.DataFrame
def azureml_main(dataframe1 = None, dataframe2 = None):

    # Execution logic goes here

    measures_per_week = dataframe1.apply(distribute_per_weeknumbers, axis=1)

    # merge duplicate columns
    # transpose the initial dataframe '.T'
    # group by row which have now duplicate index values (which were column names)
    # apply the 'any' aggregation function 
    # transpose back the result
    merged_measures_per_week = pd \
    .concat(measures_per_week.array, axis=1) \
    .fillna(value=False) \
    .T.groupby(level=0).any().T 

    deaths_per_week = dataframe2 \
    .drop(columns=['country_code','source']) \
    .rename(columns={'year_week':'week'}) \
    .set_index(['indicator','continent','country', 'week']) \
    .loc[('deaths','Europe')]

    result = deaths_per_week \
    .join(merged_measures_per_week) \
    .apply(lambda x: x.abs().fillna(0) if x.dtype.kind in 'iufc' else x.fillna(False)) \
    .sort_index()

    result.reset_index(level=['country', 'week'])
    result.drop(columns=['weekly_count','population','cumulative_count'])

    # If a zip file is connected to the third input port,
    # it is unzipped under "./Script Bundle". This directory is added
    # to sys.path. Therefore, if your zip file contains a Python file
    # mymodule.py you can import it using:
    # import mymodule
    return result