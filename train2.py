import numpy as np
from skits.preprocessing import HorizonTransformer

from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import FeatureUnion

from skits.feature_extraction import AutoregressiveTransformer
from skits.pipeline import ForecasterPipeline
from skits.preprocessing import ReversibleImputer
from sklearn.metrics import mean_absolute_error
from azureml.core import Workspace, Dataset

ws = Workspace.from_config()
ds = Dataset.get_by_name(ws, name='covid-19-measures-cases-weekly')
df = ds.to_pandas_dataframe()

y = df[df['country']=='Netherlands']['rate_14_day'].values

# period_minutes = 5
# samples_per_hour = int(60 / period_minutes)
# samples_per_day = int(24 * samples_per_hour)
# samples_per_week = int(7 * samples_per_day)

# number of weeks to predict in the future
param_horizon = 2
# number of previous weeks to use as features
param_datapoints_in_past_as_features = 10
param_n_jobs = 16
param_n_estimators = 5

lin_pipeline = ForecasterPipeline([
    # Convert the `y` target into a horizon
    ('pre_horizon', HorizonTransformer(horizon=param_horizon)),
    ('pre_reversible_imputer', ReversibleImputer(y_only=True)),
    ('features', FeatureUnion([
        ('ar_features', AutoregressiveTransformer(num_lags=param_datapoints_in_past_as_features)),
    ])),
    ('post_feature_imputer', ReversibleImputer()),
    ('regressor', MultiOutputRegressor(LinearRegression(fit_intercept=False), n_jobs=param_n_jobs))
])

X = y.reshape(-1, 1).copy()

test_size = 10
train_size = len(X) - test_size
lin_pipeline = lin_pipeline.fit(X[:train_size], y[:train_size])

lin_prediction = lin_pipeline.predict(X, start_idx=train_size)

y_actual = lin_pipeline.transform_y(X)
lin_mae = mean_absolute_error(y_actual[-test_size:], lin_prediction, multioutput='raw_values')

forecast = lin_pipeline.forecast(X, start_idx=train_size + 10, trans_window=param_horizon)


from sklearn.preprocessing import FunctionTransformer, Binarizer
from sklearn.compose import ColumnTransformer

ids = [i+3 for i in range(64)]

tr = ColumnTransformer([
    ('one_hotter', Binarizer(), ids),
    ('ar_features', AutoregressiveTransformer(num_lags=param_datapoints_in_past_as_features),[2])
    ])
# transformer = FunctionTransformer(bool_to_int, validate=True)
# X = np.array([[True, False], [False, True]])
# transformer.transform(X)

from xgboost import XGBRegressor

xgb_pipeline = ForecasterPipeline([
    # Convert the `y` target into a horizon
    ('pre_horizon', HorizonTransformer(horizon=param_horizon)),
    ('pre_reversible_imputer', ReversibleImputer(y_only=True)),
    # ('features', FeatureUnion([
    #     # Generate a week's worth of autoregressive features
    #     ('ar_features', AutoregressiveTransformer(num_lags=param_datapoints_in_past_as_features)),
    # ])),
    ('features', FeatureUnion([('all_features', tr)])),
    ('post_feature_imputer', ReversibleImputer()),
    ('regressor', MultiOutputRegressor(XGBRegressor(n_jobs=param_n_jobs, n_estimators=param_n_estimators)))
])

xgb_prediction = xgb_pipeline.fit(X[:train_size], y[:train_size]).predict(X, start_idx=train_size)

y_actual = xgb_pipeline.transform_y(X)
xgb_mae = mean_absolute_error(y_actual[-test_size:], xgb_prediction, multioutput='raw_values')

