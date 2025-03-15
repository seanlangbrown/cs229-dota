import pandas as pd



def print_full_df(df):
    mr = pd.get_option('display.max_rows')
    mc = pd.get_option('display.max_columns')
    dw = pd.get_option('display.width')
    mw = pd.get_option('display.max_colwidth')

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

    print(df)

    pd.set_option('display.max_rows', mr)
    pd.set_option('display.max_columns', mc)
    pd.set_option('display.width', dw)
    pd.set_option('display.max_colwidth', mw)


def print_full_columns(df):
    mr = pd.get_option('display.max_rows')
    mc = pd.get_option('display.max_columns')
    dw = pd.get_option('display.width')
    mw = pd.get_option('display.max_colwidth')

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

    print(df.columns)

    pd.set_option('display.max_rows', mr)
    pd.set_option('display.max_columns', mc)
    pd.set_option('display.width', dw)
    pd.set_option('display.max_colwidth', mw)


def snake_case(s):
    return ''.join(['_' + c.lower() if c.isupper() else c for c in s]).lower().replace(' ', '_').replace('-', '_').strip('_')
