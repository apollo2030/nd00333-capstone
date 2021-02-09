from skits.preprocessing import HorizonTransformer

from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import FeatureUnion

from skits.feature_extraction import AutoregressiveTransformer
from skits.pipeline import ForecasterPipeline
from skits.preprocessing import ReversibleImputer
from sklearn.metrics import mean_squared_error

from azureml.core import Workspace, Dataset
from azureml.core.run import Run

import argparse
import numpy as np
import pickle

ws = Workspace.from_config()
ds = Dataset.get_by_name(ws, name='covid-19-measures-cases-weekly')
df = ds.to_pandas_dataframe()
run = Run.get_context()

# number of weeks to predict in the future
param_horizon = 10
# number of previous weeks to use as features
param_datapoints_in_past_as_features = 5
param_n_jobs = 16
param_n_estimators = 10
param_test_size = 10

y = df[df['country']=='Netherlands']['rate_14_day'].values
X = y.reshape(-1, 1).copy()

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--param_horizon', type=int, default=2, help="The horizon in the future to predict in weeks")
    parser.add_argument('--param_datapoints_in_past_as_features', type=int, default=23, help="Number of past observations to turn into features")
    parser.add_argument('--param_test_size', type=int, default=10, help="Number of test samples")

    parser.add_argument('--param_n_jobs', type=int, default=16, help="Number of parallel jobs to run")
    parser.add_argument('--param_n_estimators', type=int, default=10, help="Number of estimators in Xgboost estimator")
    parser.add_argument('--save', type=bool, default=False, help="Save the model")

    args = parser.parse_args()

    lin_pipeline = ForecasterPipeline([
        # Convert the `y` target into a horizon
        ('pre_horizon', HorizonTransformer(horizon=args.param_horizon)),
        ('pre_reversible_imputer', ReversibleImputer(y_only=True)),
        ('features', FeatureUnion([
            ('ar_features', AutoregressiveTransformer(num_lags=args.param_datapoints_in_past_as_features)),
        ])),
        ('post_feature_imputer', ReversibleImputer()),
        ('regressor', MultiOutputRegressor(LinearRegression(fit_intercept=False), n_jobs=args.param_n_jobs))
    ])

    train_size = len(X) - args.param_test_size

    lin_pipeline = lin_pipeline.fit(X[:train_size], y[:train_size])
    lin_prediction = lin_pipeline.predict(X, start_idx=train_size)

    y_actual = lin_pipeline.transform_y(X)
    
    
    mrse = mean_squared_error(y_actual[-args.param_test_size:], lin_prediction)

    run.log("MRSE", np.float(mrse))
    if  args.save==True:
        pickle.dump(lin_pipeline, open( "outputs/model.pkl", "wb" ) )

if __name__ == '__main__':
    main()