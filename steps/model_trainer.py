from typing_extensions import Annotated

from zenml import ArtifactConfig, step

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

@step
def model_trainer(data_train: pd.DataFrame, target: str) -> Annotated[
    RandomForestRegressor, ArtifactConfig(name="sklearn_classifier", is_model_artifact=True)
]:
    X_train = data_train.drop(columns=[target])
    y_train = data_train[target]
    
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    
    return model