import time
from datetime import datetime, timedelta

from binance.um_futures import UMFutures
from binance.websocket.um_futures.websocket_client import UMFuturesWebsocketClient

import numpy as np
import pandas as pd
from pandas.plotting import register_matplotlib_converters
import json

import indicators
from talib import stream

register_matplotlib_converters()

NAN_LENGTH = 50


def get_symbols():
    return [Symbol('BTCUSDT')]


class Symbol:
    def __init__(self, name):
        self.name = name


def get_time_gap(timeframe):
    if timeframe == 1:
        return 60
    elif timeframe == 5:
        return 300
    elif timeframe == 16385:
        return 60 * 60
    elif timeframe == 16388:
        return 4 * 60 * 60
    else:
        return 24 * 60 * 60


def change_timeframe(timeframe):
    if timeframe == 1:
        return '1m'
    elif timeframe == 5:
        return '5m'
    elif timeframe == 16385:
        return '1h'
    elif timeframe == 16388:
        return '4h'
    else:
        return '1d'


def get_symmetrical_df_from_df(df):
    symmetric_df = df[::-1]
    new_df = pd.concat([df, symmetric_df])
    indicators.add_indicators(new_df)
    return new_df[NAN_LENGTH:-NAN_LENGTH]


class binance_datasource:
    def __init__(self):
        self.dataframe = None
        self.timeframe = 1
        self.client = UMFutures(key='your key',
                                secret='your secret')
        self.new_kline_handlers = []

    def start_fetching(self, symbol='BTCUSDT'):
        self.dataframe = self.get_df_for_display(symbol)
        websocket_client = UMFuturesWebsocketClient(on_message=self.message_handler)
        websocket_client.mark_price(symbol, 1)

    def shutdown(self):
        pass

    def add_new_kline_handler(self, handler):
        self.new_kline_handlers.append(handler)

    def message_handler(self, _, message):
        mt_time = datetime.utcnow()

        df = self.dataframe
        last_index = df.index[-1]
        json_message = json.loads(message)
        if 'p' not in json_message:
            return
        bid = float(json_message['p'])
        # print(mt_time, last_index, mt_time - last_index)
        if mt_time - last_index < self.get_timedelta():
            df.loc[last_index, 'close'] = bid
            if bid > df.loc[last_index, 'high']:
                df.loc[last_index, 'high'] = bid
            if bid < df.loc[last_index, 'low']:
                df.loc[last_index, 'low'] = bid

            # df.loc[last_index, 'atr'] = stream.ATR(df['high'], df['low'], df['close']) / 100
            # df.loc[last_index, 'rsi'] = stream.RSI(df['close']) / 100
            # df.loc[last_index, 'adx'] = stream.ADX(df['high'], df['low'], df['close']) / 100
            df.loc[last_index, 'roc1'] = stream.ROC(df['close'], timeperiod=1)
        else:
            row_data = [last_index + self.get_timedelta(), bid, bid, bid, bid]
            for i in range(5, len(df.columns)):
                row_data.append(np.NAN)
                # row_data.append(df.loc[last_index, df.columns.values[i]])
            new_index = last_index + self.get_timedelta()
            df.loc[new_index] = row_data
            for indicator_name in indicators.supported_functions:
                try:
                    value = getattr(stream, indicator_name.upper())(df['close'])
                except:
                    value = getattr(stream, indicator_name.upper())(df['high'], df['low'], df['close'])
                if isinstance(value, tuple):
                    for i in range(len(value)):
                        df.loc[new_index, f'{indicator_name}_{i}'] = value[i]
                else:
                    df.loc[new_index, indicator_name] = value
            df.loc[new_index, 'roc1'] = stream.ROC(df['close'], timeperiod=1)
            for new_kline_handler in self.new_kline_handlers:
                new_kline_handler(df)

    def get_period_by_timeframe(self, timeframe):
        if timeframe <= 5:
            return 60
        elif timeframe <= 60:
            return 200
        elif timeframe <= 300:
            return 900
        elif timeframe <= 900:
            return 3600
        else:
            return 7200

    def get_timedelta(self):
        if self.timeframe == 1:
            return timedelta(minutes=1)
        elif self.timeframe == 5:
            return timedelta(minutes=5)
        elif self.timeframe == 16385:
            return timedelta(hours=1)
        elif self.timeframe == 16388:
            return timedelta(hours=4)
        else:
            return timedelta(days=1)

    def get_symmetrical_df(self, symbol, timeframe=1, limit=1000, end_time=None):
        df = self.get_df(symbol, timeframe=timeframe, limit=limit+NAN_LENGTH, end_date=end_time)

        return get_symmetrical_df_from_df(df)

    def get_dataframes(self, symbol, size=1, limit=1500, timeframe=1, end_time=None):
        if end_time is None:
            end_time = int(round(datetime.now().timestamp() * 1000))

        dataframes = []
        for _ in range(size):
            df = self.get_df(symbol, timeframe=timeframe, limit=limit, end_date=end_time)
            #df.dropna(inplace=True)
            dataframes.append(df)

            end_time = end_time - (limit + 1) * 1000 * get_time_gap(timeframe)

        end_time = end_time - 50 * 1000 * get_time_gap(timeframe)
        df = self.get_df(symbol, timeframe=timeframe, limit=50, end_date=end_time)
        dataframes.append(df)

        dataframes.reverse()
        all_data = pd.concat(dataframes)
        indicators.add_indicators(all_data)

        dataframes = []
        for i in range(size):
            dataframes.append(all_data[-(size-i)*limit-1:-(size-i-1)*limit-1])

        return dataframes

    def get_df(self, symbol, timeframe=1, limit=1500, start_date=None, end_date=None):
        if end_date is None:
            end_date = int(round(datetime.now().timestamp() * 1000))
        if start_date is None:
            start_date = end_date - (limit + 1) * 1000 * get_time_gap(timeframe)
        date_object1 = datetime.fromtimestamp(start_date // 1000)
        date_object2 = datetime.fromtimestamp(end_date // 1000)
        print(date_object1, date_object2)
        print(start_date, end_date)

        data = self.client.klines(symbol, change_timeframe(timeframe), startTime=start_date, enTime=end_date,
                                  limit=limit)

        ticks_frame = pd.DataFrame(data, columns=['time', 'open', 'high', 'low', 'close', 'volume', 'close_time',
                                                  'asset_volume', 'trades', 'base_volume', 'quote_volume', 'ignore'])
        ticks_frame['time'] = ticks_frame['time'] / 1000
        ticks_frame['time'] = pd.to_datetime(ticks_frame['time'], unit='s')
        #indicators.add_indicators(ticks_frame)
        # ticks_frame.dropna(inplace=True)
        ticks_frame.index = pd.DatetimeIndex(ticks_frame['time'])
        ticks_frame = ticks_frame.astype(
            {'open': 'float', 'high': 'float', 'low': 'float', 'close': 'float', 'volume': 'float'})

        # for name in self.indicator_names:
        #    ticks_frame[name] = indicators.add_indicator(name)
        #indicators.add_indicators(ticks_frame)

        return ticks_frame

    def get_df_for_display(self, symbol, timeframe=1, limit=1500, start_date=None, end_date=None):
        ticks_frame = self.get_df(symbol, timeframe=timeframe, limit=limit, start_date=start_date, end_date=end_date)
        indicators.add_indicators(ticks_frame)

        ticks_frame['action'] = 0
        ticks_frame['buy'] = np.NAN
        ticks_frame['sell'] = np.NAN
        ticks_frame['hold_cum'] = 0
        ticks_frame['ai_cum'] = 0

        return ticks_frame
