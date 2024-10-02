from zenml import pipeline
from zenml.config import DockerSettings

from steps import data_loader, data_splitter, data_preprocessor

docker_settings = DockerSettings(requirements="pipelines/requirements.txt")

@pipeline(settings={"docker": docker_settings})
def feature_engineering():
    raw_dataset = data_loader('data/winequality-red.csv')
    raw_dataset_train, raw_dataset_test = data_splitter(raw_dataset)
    dataset_train, dataset_test, _= data_preprocessor(raw_dataset_train, raw_dataset_test)
    return dataset_train, dataset_test
