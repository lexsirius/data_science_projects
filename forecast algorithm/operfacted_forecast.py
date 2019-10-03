import pandas as pd
from pandas.tseries.offsets import MonthEnd

import numpy as np

from typing import List


def build_lag_coeffs(group: pd.Series,
                     lag: int, 
                     exclude: List[pd.Period] = None) -> pd.Series: 
    
    coeff_df = group.pct_change(lag) + 1
    coeff_df = coeff_df.to_frame().reset_index()
    coeff_df.columns = ['report_dt', 'value']
    coeff_df['day'] = coeff_df['report_dt'].dt.day
    coeff_df['diff_month'] = coeff_df.report_dt.max().to_period('M') - coeff_df['report_dt'].dt.to_period('M')
    coeff_df['report_dt_end'] = coeff_df['report_dt'].dt.to_period('M')
    coeff_df = coeff_df[~coeff_df.report_dt.dt.to_period('M').isin(exclude)]
    pivot_coeff_table = coeff_df.query('diff_month <= 12').pivot_table(values='value', index = 'day', columns='report_dt_end')
    month_mean = pivot_coeff_table.mean(axis=1)
    month_mean = month_mean[coeff_df.report_dt.max().day:]
    return month_mean

def make_operfacted_forecast(group: pd.Series, 
                             coeffs: List[pd.Series]) -> pd.Series:
    days_to_fill = (group.index.max() + MonthEnd(0)).to_period('D') \
                    - group.index.max().to_period('D') 
    index_to_fill = group.index.shift(days_to_fill)[-days_to_fill:]
    tmp_group = group.copy()
    for dt in index_to_fill:
        lst_weighted_balance = []
        for lag in range(3):
            lst_weighted_balance.append(tmp_group.iloc[-(lag+1)] * coeffs[lag][dt.day])
        tmp_group[dt] = np.mean(lst_weighted_balance)
    return tmp_group

def avg_operfacted_forecast(group: pd.Series, 
                            lag_count: int,
                            excluded_dates: List[pd.Period] = None) -> pd.Series:
    coeffs = [build_lag_coeffs(group, lag+1, excluded_dates) for lag in range(lag_count)]
    forecast = make_operfacted_forecast(group, coeffs)
    forecast = forecast[~forecast.index.isin(group.index)]
    return forecast