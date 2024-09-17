from binance_ds import binance_datasource
import binance_ds
from ga import train_all
import MetaTrader5 as mt5
from matplotlib import pyplot as plt
import pandas as pd
import indicators



def main():
    data_source = binance_datasource()
    dataframes = data_source.get_dataframes('BTCUSDT', limit=1000, timeframe=mt5.TIMEFRAME_H1, size=7, end_time=1722700000000)
    # for df in dataframes:
    #    print(df[indicators.supported_indicators])
    train_frame1 = dataframes[3]
    validation_frame1 = dataframes[4]
    #validation_frame2 = dataframes[4]
    #train_frame2 = binance_ds.get_symmetrical_df_from_df(dataframes[2])
    #plt.plot(train_frame2['close'].to_numpy())
    #plt.savefig('result/flip.png')
    #validation_frame2 = dataframes[4]
    train_all([train_frame1], [[validation_frame1]], [train_frame1]+[validation_frame1]+dataframes[5:], generation=100)


if __name__ == "__main__":
    main()
