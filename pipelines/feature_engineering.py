from zenml import pipeline

from steps import data_loader, data_splitter, data_preprocessor

@pipeline
def feature_engineering():
    raw_dataset = data_loader('data/winequality-red.csv')
    raw_dataset_train, raw_dataset_test = data_splitter(raw_dataset)
    dataset_train, dataset_test, _= data_preprocessor(raw_dataset_train, raw_dataset_test)
    return dataset_train, dataset_test
