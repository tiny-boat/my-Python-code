#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import os
# to download https://www.kaggle.com/russellyates88/suicide-rates-overview-1985-to-2016

data_path = 'F:/suicide-rates-overview-1985-to-2016'
df = (pd.read_csv(filepath_or_buffer=os.path.join(data_path, 'master.csv'))
.rename(columns={'suicides/100k pop' : 'suicides_per_100k', ' gdp_for_year ($) ' : 'gdp_year',  'gdp_per_capita ($)' : 'gdp_capita', 'country-year' : 'country_year'}) 
.assign(gdp_year=lambda _df: _df['gdp_year'].str
.replace(',','').astype(np.int64)) )

df.columns
df['generation'].unique()
df['country'].nunique()
df.describe()
df.head(5)
df.tail(5)


def mem_usage(df: pd.DataFrame) -> str:
    """This method styles the memory usage of a DataFrame to be readable as MB.
    Parameters
    ----------
    df: pd.DataFrame
        Data frame to measure.
    Returns
    -------
    str
        Complete memory usage as a string formatted for MB.
    """
    return f'{df.memory_usage(deep=True).sum() / 1024 ** 2 : 3.2f} MB'


def convert_df(df: pd.DataFrame, deep_copy: bool = True) -> pd.DataFrame:
    """Automatically converts columns that are worth stored as
    ``categorical`` dtype.
    Parameters
    ----------
    df: pd.DataFrame
        Data frame to convert.
    deep_copy: bool
        Whether or not to perform a deep copy of the original data frame.
    Returns
    -------
    pd.DataFrame
        Optimized copy of the input data frame.
    """
    return df.copy(deep=deep_copy).astype({
        col: 'category' for col in df.columns
        if df[col].nunique() / df[col].shape[0] < 0.5})

mem_usage(df)
# 10.28 MB
mem_usage(df.set_index(['country', 'year', 'sex', 'age']))
# 5.00 MB
mem_usage(convert_df(df))
# 1.40 MB
mem_usage(convert_df(df.set_index(['country', 'year', 'sex', 'age'])))
# 1.40 MB


df.query('country == "Albania" and year == 1987 and sex == "male" and age == "25-34 years"')
mi_df = df.set_index(['country', 'year', 'sex', 'age'])
mi_df.loc['Albania', 1987, 'male', '25-34 years']

mi_df.sort_index()
mi_df.index.is_monotonic

# apply assign loc query pipe groupby agg nlargest unstack

(df
 .assign(valid_cy=lambda _serie: _serie.apply(
     lambda _row: re.split(r'(?=\d{4})', _row['country_year'])[1] == str(_row['year']),
     axis=1))
 .query('valid_cy == False')
 .pipe(log_shape)
)

from sklearn.preprocessing import MinMaxScaler


def norm_df(df, columns):
    return df.assign(**{col: MinMaxScaler().fit_transform(df[[col]].values.astype(float))
                        for col in columns})

for sex in ['male', 'female']:
    print(sex)
    print(
        df
        .query(f'sex == "{sex}"')
        .groupby(['country'])
        .agg({'suicides_per_100k': 'sum', 'gdp_year': 'mean'})
        .rename(columns={'suicides_per_100k':'suicides_per_100k_sum',
                         'gdp_year': 'gdp_year_mean'})
#         Recommended in v0.25
#         .agg(suicides_per_100k=('suicides_per_100k_sum', 'sum'),
#              gdp_year=('gdp_year_mean', 'mean'))
        .pipe(norm_df, columns=['suicides_per_100k_sum', 'gdp_year_mean'])
        .corr(method='spearman')
    )
    print('\n')
