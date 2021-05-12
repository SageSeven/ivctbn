import pandas as pd


def all_values(n_state, size):
    """
    Get all possible values.
    Example: n_state = 3, size = 3, then return a iterable with values
        [0, 0, 0], [1, 0, 0], [2, 0, 0], [0, 1, 0], [1, 1, 0], ..., [1, 2, 2], [2, 2, 2]
    :param n_state:
    :param size:
    :return:
    """
    result = [0] * size
    for i in range(n_state ** size):
        num = i
        for j in range(size):
            result[j] = num % n_state
            num //= n_state
        yield result


def multi_col_equal(df: pd.DataFrame, cols, values) -> pd.DataFrame:
    """
    Filter with multi columns in a dataframe.
    Example: df = DataFrame([[1,2,3,4],[2,3,4,5],[3,4,5,6]), cols = [1,2], values = [2,3]
        then result = DataFrame([[1,2,3,4]])

    :param df:
    :param cols:
    :param values:
    :return:
    """
    df1 = df[cols] == values
    df2 = df1.astype(int).T.cumprod().T.astype(bool)
    return df[df2[df2.columns[-1]]]
