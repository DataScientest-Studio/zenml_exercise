from typing import Tuple
from typing_extensions import Annotated

from zenml import step

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from utils.preprocess import DataFrameCaster

@step
def data_preprocessor(
    raw_data_train: pd.DataFrame,
    raw_data_test: pd.DataFrame,
) -> Tuple[
    Annotated[pd.DataFrame, "dataset_train"],
    Annotated[pd.DataFrame, "dataset_test"],
    Annotated[Pipeline, "preprocess_pipeline"],
]:
    preprocess_pipeline = Pipeline([
        ("scaling", StandardScaler()),
        ("cast", DataFrameCaster(raw_data_train.columns))])

    dataset_train = preprocess_pipeline.fit_transform(raw_data_train)
    dataset_test = preprocess_pipeline.transform(raw_data_test)

    return dataset_train, dataset_test, preprocess_pipeline
