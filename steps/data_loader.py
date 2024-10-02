from zenml import step

import pandas as pd

@step
def data_loader(
    filepath_or_buffer: str
) -> pd.DataFrame:
    data = pd.read_csv(filepath_or_buffer)
    return data
