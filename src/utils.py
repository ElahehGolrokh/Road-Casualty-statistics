import pandas as pd


def read_csv(csv_path: str) -> pd.DataFrame:
    """
    Reads csv file

    Args:
        csv_path: the relative path of the csv file

    Returns:
        df: raw pandas dataframe
    """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print('input a vaild csv file path')
    else:
        return df