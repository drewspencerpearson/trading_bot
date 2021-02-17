import numpy as np
import pandas as pd
from sklearn.kernel_approximation import Nystroem
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier

# NOTE: Make sure that the class is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1).values
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'].values, random_state=None)

# Average CV score on the training set was:0.8028858267395129
exported_pipeline = make_pipeline(
    Nystroem(gamma=0.7000000000000001, kernel="linear", n_components=6),
    XGBClassifier(learning_rate=1.0, max_depth=8, min_child_weight=4, n_estimators=100, nthread=1, subsample=0.45)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
