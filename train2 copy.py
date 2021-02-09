import numpy as np
from skits.preprocessing import HorizonTransformer

from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import FeatureUnion

from skits.feature_extraction import AutoregressiveTransformer
from skits.pipeline import ForecasterPipeline
from skits.preprocessing import ReversibleImputer
from sklearn.metrics import mean_absolute_error
from azureml.core import Workspace, Dataset
from xgboost import XGBRegressor
from cleaner import Cleaner

ws = Workspace.from_config()
ds = Dataset.get_by_name(ws, name='covid-19-measures-cases-weekly')
df = ds.to_pandas_dataframe()

# number of weeks to predict in the future
param_horizon = 10
# number of previous weeks to use as features
param_datapoints_in_past_as_features = 5
param_n_jobs = 16
param_n_estimators = 10

from sklearn.preprocessing import Binarizer
from sklearn.compose import ColumnTransformer

ids = [i for i in range(64)]

tr = ColumnTransformer([
    ('ar_features', AutoregressiveTransformer(num_lags=param_datapoints_in_past_as_features),[0]),
    ('one_hotter', Binarizer(), ids),
    ])

xgb_pipeline = ForecasterPipeline([
    # Convert the `y` target into a horizon
    ('pre_horizon', HorizonTransformer(horizon=param_horizon)),
    ('pre_reversible_imputer', ReversibleImputer(y_only=True)),
    ('features', FeatureUnion([('all_features', tr)])),
    ('post_feature_imputer', Cleaner()),
    ('regressor', MultiOutputRegressor(XGBRegressor(n_jobs=param_n_jobs, n_estimators=param_n_estimators)))
])
test_size = 10
train_size = len(X) - test_size

y = df[df['country']=='Netherlands']['rate_14_day'].values
X = df[df['country']=='Netherlands']
X.drop(['country','week'], axis=1, inplace=True)

xgb_prediction = xgb_pipeline.fit(X[:train_size], y[:train_size]).predict(X, start_idx=train_size)

y_actual = xgb_pipeline.transform_y(y)
xgb_mae = mean_absolute_error(y_actual[-test_size:], xgb_prediction, multioutput='raw_values')


from azureml.core import Environment
from azureml.core.conda_dependencies import CondaDependencies

myenv = Environment("myenv")