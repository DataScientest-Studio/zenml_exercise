from zenml import step

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

@step
def inference_predictor(data: pd.DataFrame, model: RandomForestRegressor) -> pd.Series:
    if 'quality' in data.columns:
        data = data.drop(columns=['quality'])
    
    predictions = model.predict(data)
    predictions = pd.Series(predictions)
    
    return predictions
