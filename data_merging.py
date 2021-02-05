import pandas as pd
import datetime
from azureml.core import Workspace, Dataset

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

ws = Workspace.from_config()
dsf = Dataset.get_by_name(ws, name='covid-19-response-weekly').to_pandas_dataframe()

measures_per_week = dsf.apply(distribute_per_weeknumbers, axis=1)

# merge duplicate columns
# transpose the initial dataframe '.T'
# group by row which have now duplicate index values (which were column names)
# apply the 'any' aggregation function 
# transpose back the result
merged_measures_per_week = pd \
.concat(measures_per_week.array, axis=1) \
.fillna(value=False) \
.T.groupby(level=0).any().T 


dsf = Dataset.get_by_name(ws, name='covid-19-cases-deaths-weekly').to_pandas_dataframe()

deaths_per_week = dsf \
.drop(columns=['country_code','source']) \
.rename(columns={'year_week':'week'}) \
.set_index(['indicator','continent','country', 'week']) \
.loc[('deaths','Europe')]

deaths_per_week

res2 = deaths_per_week \
.join(merged_measures_per_week) \
.apply(lambda x: x.abs().fillna(0) if x.dtype.kind in 'iufc' else x.fillna(False)) \
.sort_index()
res2

for i in res2.index.get_level_values('country').unique():
    res2.loc[i].plot(title=i, y=['weekly_count'],kind='area')

