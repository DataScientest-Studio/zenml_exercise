from typing import Optional
from uuid import UUID

from zenml import pipeline
from zenml.client import Client

from steps import model_trainer
from pipelines import feature_engineering

@pipeline
def training(
    train_dataset_id: Optional[UUID] = None,
):
    if train_dataset_id is None:
        dataset_trn, _ = feature_engineering()
    else:
        client = Client()
        dataset_trn = client.get_artifact_version(name_id_or_prefix=train_dataset_id)
    
    model = model_trainer(dataset_trn)
    return model
