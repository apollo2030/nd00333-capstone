# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
import json
import logging
import os
import pickle
import numpy as np
import pandas as pd
import joblib

import azureml.automl.core
from azureml.automl.core.shared import logging_utilities, log_server
from azureml.telemetry import INSTRUMENTATION_KEY

from inference_schema.schema_decorators import input_schema, output_schema
from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType
from inference_schema.parameter_types.pandas_parameter_type import PandasParameterType


input_sample = pd.DataFrame({"country": pd.Series(["Netherlands"], dtype="object"), "week": pd.Series(["2021-2-7"], dtype="datetime64[ns]"), "AdaptationOfWorkplace": pd.Series([False], dtype="bool"), "AdaptationOfWorkplacePartial": pd.Series([False], dtype="bool"), "BanOnAllEvents": pd.Series([False], dtype="bool"), "BanOnAllEventsPartial": pd.Series([False], dtype="bool"), "ClosDaycare": pd.Series([False], dtype="bool"), "ClosDaycarePartial": pd.Series([False], dtype="bool"), "ClosHigh": pd.Series([False], dtype="bool"), "ClosHighPartial": pd.Series([False], dtype="bool"), "ClosPrim": pd.Series([False], dtype="bool"), "ClosPrimPartial": pd.Series([False], dtype="bool"), "ClosPubAny": pd.Series([False], dtype="bool"), "ClosPubAnyPartial": pd.Series([False], dtype="bool"), "ClosSec": pd.Series([False], dtype="bool"), "ClosSecPartial": pd.Series([False], dtype="bool"), "ClosureOfPublicTransport": pd.Series([False], dtype="bool"), "ClosureOfPublicTransportPartial": pd.Series([False], dtype="bool"), "EntertainmentVenues": pd.Series([False], dtype="bool"), "EntertainmentVenuesPartial": pd.Series([False], dtype="bool"), "GymsSportsCentres": pd.Series([False], dtype="bool"), "GymsSportsCentresPartial": pd.Series([False], dtype="bool"), "HotelsOtherAccommodation": pd.Series([False], dtype="bool"), "HotelsOtherAccommodationPartial": pd.Series([False], dtype="bool"), "IndoorOver100": pd.Series([False], dtype="bool"), "IndoorOver1000": pd.Series([False], dtype="bool"), "IndoorOver50": pd.Series([False], dtype="bool"), "IndoorOver500": pd.Series([False], dtype="bool"), "MasksMandatoryAllSpaces": pd.Series([False], dtype="bool"), "MasksMandatoryAllSpacesPartial": pd.Series([False], dtype="bool"), "MasksMandatoryClosedSpaces": pd.Series([False], dtype="bool"), "MasksMandatoryClosedSpacesPartial": pd.Series([False], dtype="bool"), "MasksVoluntaryAllSpaces": pd.Series([False], dtype="bool"), "MasksVoluntaryAllSpacesPartial": pd.Series([False], dtype="bool"), "MasksVoluntaryClosedSpaces": pd.Series([False], dtype="bool"), "MasksVoluntaryClosedSpacesPartial": pd.Series([False], dtype="bool"), "MassGather50": pd.Series([False], dtype="bool"), "MassGather50Partial": pd.Series([False], dtype="bool"), "MassGatherAll": pd.Series([False], dtype="bool"), "MassGatherAllPartial": pd.Series([False], dtype="bool"), "NonEssentialShops": pd.Series([False], dtype="bool"), "NonEssentialShopsPartial": pd.Series([False], dtype="bool"), "OutdoorOver100": pd.Series([False], dtype="bool"), "OutdoorOver1000": pd.Series([False], dtype="bool"), "OutdoorOver50": pd.Series([False], dtype="bool"), "OutdoorOver500": pd.Series([False], dtype="bool"), "PlaceOfWorship": pd.Series([False], dtype="bool"), "PlaceOfWorshipPartial": pd.Series([False], dtype="bool"), "PrivateGatheringRestrictions": pd.Series([False], dtype="bool"), "PrivateGatheringRestrictionsPartial": pd.Series([False], dtype="bool"), "RegionalStayHomeOrder": pd.Series([False], dtype="bool"), "RegionalStayHomeOrderPartial": pd.Series([False], dtype="bool"), "RestaurantsCafes": pd.Series([False], dtype="bool"), "RestaurantsCafesPartial": pd.Series([False], dtype="bool"), "SocialCircle": pd.Series([False], dtype="bool"), "SocialCirclePartial": pd.Series([False], dtype="bool"), "StayHomeGen": pd.Series([False], dtype="bool"), "StayHomeGenPartial": pd.Series([False], dtype="bool"), "StayHomeOrder": pd.Series([False], dtype="bool"), "StayHomeOrderPartial": pd.Series([False], dtype="bool"), "StayHomeRiskG": pd.Series([False], dtype="bool"), "StayHomeRiskGPartial": pd.Series([False], dtype="bool"), "Teleworking": pd.Series([False], dtype="bool"), "TeleworkingPartial": pd.Series([False], dtype="bool"), "WorkplaceClosures": pd.Series([False], dtype="bool"), "WorkplaceClosuresPartial": pd.Series([False], dtype="bool")})
try:
    log_server.enable_telemetry(INSTRUMENTATION_KEY)
    log_server.set_verbosity('INFO')
    logger = logging.getLogger('azureml.automl.core.scoring_script_forecasting')
except Exception:
    pass


def init():
    global model
    # This name is model.id of model that we want to deploy deserialize the model file back
    # into a sklearn model
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'my_model.pickle')
    path = os.path.normpath(model_path)
    path_split = path.split(os.sep)
    log_server.update_custom_dimensions({'model_name': path_split[1], 'model_version': path_split[2]})
    try:
        logger.info("Loading model from path.")
        model = joblib.load(model_path)
        logger.info("Loading successful.")
    except Exception as e:
        logging_utilities.log_traceback(e, logger)
        raise

@input_schema('data', PandasParameterType(input_sample, enforce_shape=False))
def run(data):
    try:
        y_query = None
        if 'y_query' in data.columns:
            y_query = data.pop('y_query').values
        else:
            y_query = np.full(len(data.index), np.NaN)
            
        result = model.forecast(data, y_query, ignore_data_errors=True)
    except Exception as e:
        result = str(e)
        return json.dumps({"error": result})

    forecast_as_list = result[0].tolist()
    index_as_df = result[1].index.to_frame().reset_index(drop=True)
    
    return json.dumps({"forecast": forecast_as_list,   # return the minimum over the wire: 
                       "index": json.loads(index_as_df.to_json(orient='records'))  # no forecast and its featurized values
                      })
