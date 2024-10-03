from zenml import step

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

@step
def inference_predictor(data: pd.DataFrame, target: str, model: RandomForestRegressor) -> pd.Series:
    if target in data.columns:
        data = data.drop(columns=[target])
    
    predictions = model.predict(data)
    predictions = pd.Series(predictions)
    
    return predictions
