
# Covid-19 cases forecasting

The project intends to build a model that can predict the number of infection cases of Covid-19 based on the selection of measures that are applied in that country. It uses data from the european disease center so the model is applicable for countries in Europe.

## Dataset

### Overview
*TODO*: Explain about the data you are using and where you got it from.

### Task
Forecast the number of cases per week per country. The data needed for that is in 2 datasets. One contains the measure taken and the corresponding start and end date. The second contains the number of cases per country per week. I will combine the 2 datasets and create the 3rd composite dataset. The measures will end up being one hot encoded per week.

### Access
To access the data I registered the 2 initial datasets and made a training pipeline that would join the 2 into the 3rd dataset. The 3rd dataset is saved in the default datastore of the workspace. The 3rd dataset then is manually registered in the machine learning studio. From here i use the standard way to access de registered dataset:
```Python
dataset = Dataset.get_by_name(ws, name='covid-19-measures-cases-weekly')
df = dataset.to_pandas_dataframe()
```

## Automated ML
To setup the automated ml run i used the forecasting task with the corresponding configuration.
The pipeline gave out the best model. One of the difficulties is the fact that in jupyter notebooks in vs code the intellisense is flanky - it works well for python files but not so for notebooks. Documentation is also not the best possible - for instance the forecasting_parameters parameter for the  AutoMlConfig is not documented. 

# ForecastingParameters Configuration

We are going to create a forecasting model.  
Azure ml sdk has already a class that groups all the parameters needed for a forecasting task: `ForecastingParameters`.  

In our case the dataset timestamp column is called `week` so we set `time_column_name='week'`.  
We would like to have an accurate forecast for at least 10 weeks so we set the `forecast_horizon` to 10.   
We have the same time series repeated per country - to capture that we set the `time_series_id_column_names` to `['country']`.   
The time series frequency is per week and we can express that with the parameter `freq='W'`.  
Target_lags is set to auto because we don't know which features are dependent and which not.  
`target_rolling_window_size=10` means that we take into account 10 past records in order to perform the forecasting.  

## AutoML Configuration
We are going to create a forecasting model so we set `task='forecasting'`   
In order to get the model cross validated better we set the `n_cross_validations=30`  
We need to specify the column that we want to forecast so we set `label_column_name='rate_14_day'`  
The forecast task need to optimize for normalized_root_mean_squared_error because it is a better metric for values that do not differ in order of magnitudes from each other thus we set `primary_metric` to 'normalized_root_mean_squared_error'.  
Running the experiment for too long is not our goal so we set a timeout to 20 minutes: `experiment_timeout_hours=0.3`  
We will block the following models:
 - 'ExtremeRandomTrees' because it performs poor when there is a high number of noisy features (in high dimensional data-sets).
 - 'AutoArima' - because our dataset is a multivariate time series - it depends not only on one variable but on 64.
 - 'Prophet' because it works best with time series that have strong seasonal effects but in our case we don't know that and don't want to assume that either.  
 
To do that we set `blocked_models` to `['ExtremeRandomTrees', 'AutoArima', 'Prophet']`.   
We have already limited the run duration of our experiment to 20 minute so we will disable early stopping like this: `enable_early_stopping=False`

*TODO*: Give an overview of the `automl` settings and configuration you used for this experiment

### Results

*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?

*TODO* Remember to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search


### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

