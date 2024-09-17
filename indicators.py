import random

import numpy as np
import pandas as pd
from scipy import stats
import math

import talib as ta
from talib import abstract

COMBO_UPPER_BOUND = 8
COMBO_LOWER_BOUND = 3

supported_functions = ['rsi', 'adx', 'cci', 'willr', 'roc', 'atr', 'apo', 'macd', 'stoch']
supported_indicators = ['rsi', 'adx', 'cci', 'willr', 'roc', 'atr', 'apo', 'macd_0', 'macd_1', 'macd_2', 'stoch_0', 'stoch_1']


def get_random_indicators(size):
    choices = np.random.choice(len(supported_indicators), size, replace=False)
    return [supported_indicators[i] for i in choices]


def get_random_size_indicators():
    size = random.randint(COMBO_LOWER_BOUND, COMBO_UPPER_BOUND)
    return get_random_indicators(size)


def get_one_random_indicator(existing_indicators):
    rest_indicators = []
    for indicator_name in supported_indicators:
        if indicator_name not in existing_indicators:
            rest_indicators.append(indicator_name)
    if len(rest_indicators) <= 1:
        print(existing_indicators, supported_indicators, rest_indicators)
    if len(rest_indicators) > 1:
        return rest_indicators[random.randint(0, len(rest_indicators)-1)]
    else:
        return rest_indicators[0]


def get_supported_cdl():
    cdl_methods = [m for m in dir(ta) if m.find('CDL') == 0]
    return cdl_methods


def add_multiple_labels(df, max_day=10):
    for i in range(max_day):
        df['close_after' + str(i + 1)] = df['close'].shift(-(i + 1))
        df['label' + str(i + 1)] = np.where(df['close_after' + str(i + 1)] > df['close'], 1, -1)


def add_days_labels(df, day_array):
    for i in day_array:
        df['close_after' + str(i + 1)] = df['close'].shift(-(i + 1))
        df['label' + str(i + 1)] = np.where(df['close_after' + str(i + 1)] > df['close'], 1, -1)


def add_price_change(df, days=5):
    df['close_after'] = df['close'].shift(-days)
    df['close_before'] = df['close'].shift(days)
    df['change_after'] = df['close_after'] - df['close']
    df['change_before'] = df['close'] - df['close_before']
    df['change_continue'] = df['change_after'] * df['change_before']


def add_candle_pattern(df, pattern_name, direction=100):
    df[pattern_name] = getattr(ta, pattern_name)(df['open'], df['high'], df['low'], df['close'])
    df[pattern_name + '_marker'] = np.where(df[pattern_name] == 100, df['high'] * 1.0001, np.NAN)
    df[pattern_name + '_marker_bear'] = np.where(df[pattern_name] == -100, df['low'] * 0.9999, np.NAN)


def add_all_candle_patterns(df):
    cdl_methods = get_supported_cdl()
    print(cdl_methods)
    for mtd in cdl_methods:
        df[mtd[3:]] = getattr(ta, mtd)(df['open'], df['high'], df['low'], df['close'])


def get_all_candle_features(df, remove_zero_days=False):
    cdl_methods = get_supported_cdl()
    print(cdl_methods)
    df_cdl = pd.DataFrame(index=df.index)
    for mtd in cdl_methods:
        df_cdl[mtd] = getattr(ta, mtd)(df['open'], df['high'], df['low'], df['close'])
    # tgt = df[target]
    df_cdl['high'] = df['high']

    if remove_zero_days:
        non_zero = df_cdl.sum(axis=1) > 0
        # tgt = tgt[non_zero]
        df_cdl = df_cdl[non_zero]

    return df_cdl


def get_swing(row):
    if row['high'] > row['high_one_day_before'] and row['high'] > row['high_two_days_before'] \
            and row['high'] > row['high_one_day_after'] and row['high'] > row['high_two_days_after']:
        return -1
    elif row['low'] < row['low_one_day_before'] and row['low'] < row['low_two_days_before'] \
            and row['low'] < row['low_one_day_after'] and row['low'] < row['low_two_days_after']:
        return 1
    else:
        return 0


def get_marker(df, column_name):
    df[column_name + '_marker'] = np.where(df[column_name] == 1, df['high'] * 1.01, np.NAN)
    return df[column_name + '_marker']


def add_indicator(df, ta_name):
    df[ta_name] = getattr(ta, ta_name)(df['close'])


def add_indicators(df):
    inputs = {
        'open': df['open'],
        'high': df['high'],
        'low': df['low'],
        'close': df['close'],
        'volume': df['volume']
    }
    for indicator_name in supported_functions:
        fun = abstract.Function(indicator_name)
        value = fun(inputs)
        if isinstance(value, list):
            for i in range(len(value)):
                df[f'{indicator_name}_{i}'] = value[i]
        else:
            df[indicator_name] = value
    df['roc1'] = ta.ROC(df['close'], timeperiod=1)


def add_indicators1(df):
    # ticks_frame = Sdf.retype(df)
    ticks_frame = df
    ticks_frame['high_one_day_before'] = ticks_frame['high'].shift(1)
    ticks_frame['high_two_days_before'] = ticks_frame['high'].shift(2)
    # ticks_frame['close_three_days_before'] = ticks_frame['close'].shift(3)
    ticks_frame['high_one_day_after'] = ticks_frame['high'].shift(-1)
    ticks_frame['high_two_days_after'] = ticks_frame['high'].shift(-2)
    # ticks_frame['close_three_days_after'] = ticks_frame['close'].shift(-3)

    ticks_frame['low_one_day_before'] = ticks_frame['low'].shift(1)
    ticks_frame['low_two_days_before'] = ticks_frame['low'].shift(2)
    ticks_frame['low_one_day_after'] = ticks_frame['low'].shift(-1)
    ticks_frame['low_two_days_after'] = ticks_frame['low'].shift(-2)
    ticks_frame['swing'] = ticks_frame.apply(
        get_swing, axis=1)
    ticks_frame['swing_high'] = np.where(ticks_frame['swing'] == -1, ticks_frame['high'], np.NAN)
    ticks_frame['swing_low'] = np.where(ticks_frame['swing'] == 1, ticks_frame['low'], np.NAN)

    return ticks_frame


def add_one_day_label(df):
    total_avg = np.abs((df['close'] - df['open']) / df['close']).mean()
    print(total_avg)
    df['close_after1'] = df['close'].shift(-1)
    df['change1'] = (df['close_after1'] - df['close']) / df['close']
    df['label_big_up'] = np.where(df['change1'] > total_avg * 2, 1, 0)
    df['label_big_down'] = np.where(df['change1'] < total_avg * -2, 1, 0)

    df['big_up_marker'] = np.where(df['label_big_up'] == 1, df['high'] * 1.0001, np.NAN)
    df['big_down_marker'] = np.where(df['label_big_down'] == 1, df['low'] * 0.9999, np.NAN)


def add_labels(df, days=5):
    total_avg = np.abs((df['close'] - df['open']) / df['close']).mean()
    print(total_avg)
    for day in range(days):
        df['close_after' + str(day)] = df['close'].shift(-day)
    df['label'] = 0
    df['max'] = 0.0
    df['min'] = 0.0
    df['avg'] = 0.0
    df['label_big_move'] = 0
    for i, row in df.iterrows():
        change = 0
        avg = 0
        for day in range(days):
            change = (row['close_after' + str(day)] - row['close']) / row['close']
            if change > row['max']:
                df.loc[i, 'max'] = change
            if change < row['min']:
                df.loc[i, 'min'] = change
            avg += change
        avg = avg / days
        df.loc[i, 'avg'] = avg

        '''for day in range(days):
            if (row['close_after' + str(day)] - row['close']) / row['close'] < -avg * 1:
                if label == 1:
                    label = 0
                else:
                    label = -1
                break'''

        df.loc[i, 'label'] = 1 if change > 0 else -1
