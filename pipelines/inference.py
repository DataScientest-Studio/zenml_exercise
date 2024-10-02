from typing import Optional
from uuid import UUID

from zenml import pipeline
from zenml.client import Client
from zenml.config import DockerSettings

import pandas as pd

from steps import data_loader, data_splitter, data_preprocessor, inference_preprocessor, inference_predictor
from pipelines.training import training

docker_settings = DockerSettings(requirements="pipelines/requirements.txt")

@pipeline(settings={"docker": docker_settings})
def inference(
    data_inference_path: str = 'data/winequality-red-inf.csv', 
    model_id: Optional[UUID] = None,
    preprocess_pipeline_id: Optional[UUID] = None,
    ):
    if model_id is None:
        model = training()
    else:
        client = Client()
        model = client.get_artifact_version(name_id_or_prefix=model_id)

    if preprocess_pipeline_id is None:
        raw_dataset = data_loader('data/winequality-red.csv')
        raw_dataset_train, raw_dataset_test = data_splitter(raw_dataset)
        _, dataset_test, preprocess_pipeline = data_preprocessor(raw_dataset_train, raw_dataset_test)
    else:
        client = Client()
        preprocess_pipeline = client.get_artifact_version(name_id_or_prefix=preprocess_pipeline_id)

    data_inference = data_loader(data_inference_path)
    data_inference = inference_preprocessor(data_inference, preprocess_pipeline)

    
    predictions = inference_predictor(data_inference, model)
