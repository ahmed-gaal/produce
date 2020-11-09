import pickle

import pandas as pd
from config import Config
from sklearn.ensemble import RandomForestRegressor

# Creating a path to save the models
Config.models_path.mkdir(parents=True,exist_ok=True)

# Arranging the data 
x_train = pd.read_csv(str(Config.features_path / 'train_features.csv'))
y_train = pd.read_csv(str(Config.features_path / 'train_target.csv'))

# Instantiating and fitting the data with the algorithm
model = RandomForestRegressor(
    n_estimators=150, max_depth=6, random_state=Config.random_seed
)
model = model.fit(x_train, y_train.to_numpy().ravel())

# Saving the model in a pickle file
pickle.dump(model, open(str(Config.models_path / 'model.pickle'), 'wb'))
