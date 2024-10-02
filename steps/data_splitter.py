from typing import Tuple
from typing_extensions import Annotated

from zenml import step

import pandas as pd
from sklearn.model_selection import train_test_split

@step
def data_splitter(
    dataset: pd.DataFrame, test_size: float = 0.2
) -> Tuple[
    Annotated[pd.DataFrame, "raw_dataset_trn"],
    Annotated[pd.DataFrame, "raw_dataset_tst"],
]:
    dataset_trn, dataset_tst = train_test_split(
        dataset,
        random_state=42,
    )
    dataset_trn = pd.DataFrame(dataset_trn, columns=dataset.columns)
    dataset_tst = pd.DataFrame(dataset_tst, columns=dataset.columns)
    return dataset_trn, dataset_tst
    