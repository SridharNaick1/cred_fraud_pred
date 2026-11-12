import pandas as pd
from cred_pred.config import load_config
import warnings
from pathlib import Path

def load_data(data_path: str | Path ) -> pd.DataFrame:
    df = pd.read_csv(data_path)

    return df

def validate_data(df:pd.DataFrame, config_path:str | Path , has_target: bool) -> pd.DataFrame:
    config = load_config(config_path)
    cols = config['columns']
    expected_column = [cols['id'], 
                       *cols['numeric'], 
                       *cols['flags'], 
                       *cols['ordinal'], 
                       *cols['label_encode'], 
                       *cols['category']
                       ]
    if has_target:
        expected_column.append(cols['target'])

    missing_columns = [col for col in expected_column if col not in df.columns]
    extra_columns = [col for col in df.columns if col not in expected_column]

    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")

    if extra_columns:
        warnings.warn(f'Dropping additional columns: {extra_columns}', UserWarning)

    # Keep only the expected columns, in the expected order
    df = df[expected_column]

    na_analysis = df.isna().sum()
    if (na_analysis > 0).any():
        raise ValueError(f'Data has missing values: {na_analysis} \n fix these and try again')

    # Type checking
    numeric_cols = [cols['id'], *cols['numeric']]
    if has_target:
        numeric_cols.append(cols['target'])
    bool_cols = cols['flags']
    text_cols = [*cols['ordinal'], *cols['label_encode'], *cols['category']]

    for col in numeric_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise TypeError(f"{col} should be numeric, got {df[col].dtype}")
        if col == cols['target']:
            if not df[col].isin([0, 1]).all():
                raise ValueError(f"Target value out of bounds")


    for col in bool_cols:
        if not pd.api.types.is_bool_dtype(df[col]):
            raise TypeError(f"{col} should be boolean, got {df[col].dtype}")

    for col in text_cols:
        if not pd.api.types.is_string_dtype(df[col]):
            raise TypeError(f"{col} should be text, got {df[col].dtype}")


    return df