from typing_extensions import Annotated

from zenml import step

import pandas as pd
from sklearn.pipeline import Pipeline

@step
def inference_preprocessor(
    dataset_inf: pd.DataFrame,
    target: str,
    preprocess_pipeline: Pipeline,
) -> Annotated[pd.DataFrame, "inference_dataset"]:

    dataset_inf[target] = pd.Series([1] * dataset_inf.shape[0])
    dataset_inf = preprocess_pipeline.transform(dataset_inf)
    dataset_inf = dataset_inf.drop(columns=[target])
    return dataset_inf
    