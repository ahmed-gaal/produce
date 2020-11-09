import math
import json
import pickle
import pandas as pd
from config import Config
from sklearn.metrics import r2_score, mean_squared_error

# Reading in the test datasets
x_test = pd.read_csv(str(Config.features_path / 'test_features.csv'))
y_test = pd.read_csv(str(Config.features_path / 'test_target.csv'))

# Loading our model we saved earlier on
model = pickle.load(open(str(Config.models_path / 'model.pickle'), 'rb'))

# Performing predictions
y_pred = model.predict(x_test)

# Calculating metrics for our model
rmse = math.sqrt(mean_squared_error(y_test, y_pred))
r_squared = r2_score(y_test, y_pred)

# Saving the metrics in a json file
with open(str(Config.metrics_file_path), 'w') as outfile:
    json.dump(dict(r_squared=r_squared, rmse=rmse), outfile)