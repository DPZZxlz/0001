from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

from freqtrade.strategy import (
    BooleanParameter,
    CategoricalParameter,
    DecimalParameter,
    IStrategy,
    IntParameter,
)

class BollSquze(IStrategy):

    INTERFACE_VERSION = 3


	# ROI table:
    minimal_roi = {
		"0": (0.015),
        "30":(0.02),
        "60": (0.025),
        "120": (0.04),
	}

	# Stoploss:
    stoploss = -0.05

	# Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.03
    trailing_stop_positive_offset = 0.04
    trailing_only_offset_is_reached = True

	# Sell signal
    use_exit_signal = True
    exit_profit_only = False
    exit_profit_offset = 0.01
    ignore_roi_if_entry_signal = False
    # Optimal timeframe for the strategy.

    timeframe = "1h"
    can_short = True

    bb_squeeze_threshold_high = IntParameter(30, 110, default=90, space='buy', optimize=True,load=True)
    bb_squeeze_threshold_low = IntParameter(30, 110, default=20, space='buy', optimize=True,load=True)
    sqz_length = IntParameter(70, 120, default=100, space='buy', optimize=True,load=True)
    spread_squeeze =IntParameter(90, 160, default=120, space='buy', optimize=True,load=True)

    # 添加状态变量来标记是否已经开仓
    is_position_open = False

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Calculate Bollinger Bands
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=19, stds=2)
        dataframe["bb_lower"] = bollinger["lower"]
        dataframe["bb_middle"] = bollinger["mid"]
        dataframe["bb_upper"] = bollinger["upper"]
        dataframe["spread"] = bollinger["upper"] - bollinger["lower"]
        dataframe["avgspread"] = ta.SMA(dataframe["spread"], self.sqz_length.value)

        # 计算布林带压缩
        dataframe['bb_squeeze'] = (dataframe['spread'] / dataframe['avgspread']) * 100

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # 检查是否已经开仓，如果已经开仓则不再执行开仓逻辑
        if self.is_position_open:
            return dataframe
        
        dataframe.loc[
            ( 
                (dataframe['close'] > dataframe['bb_middle']) 
                & (dataframe['bb_squeeze'] < self.bb_squeeze_threshold_high.value)  # Squeeze is higher than threshold
                & (dataframe['bb_squeeze'] > self.bb_squeeze_threshold_low.value)  # Squeeze is higher than threshold
            ),
            'enter_long'] = 1


        dataframe.loc[
            (
                (dataframe['close'] < dataframe['bb_middle']) 
                & (dataframe['bb_squeeze'] < self.bb_squeeze_threshold_high.value)  # Squeeze is higher than 
                & (dataframe['bb_squeeze'] > self.bb_squeeze_threshold_low.value)  # Squeeze is higher than threshold
            ),
            'enter_short'] = 1

        # 标记已经开仓
        self.is_position_open = True
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            (
                (dataframe['close'] < dataframe['bb_middle']) 
                & (dataframe['bb_squeeze'] > self.bb_squeeze_threshold_high.value)
                & (dataframe['spread'] > dataframe['avgspread'])
                & (dataframe['spread'] > self.spread_squeeze.value)
            ),
            'exit_long'] = 1


        dataframe.loc[
            (
                (dataframe['close'] > dataframe['bb_middle']) 
                & (dataframe['bb_squeeze'] > self.bb_squeeze_threshold_high.value)
                & (dataframe['spread'] > dataframe['avgspread'])
                & (dataframe['spread'] > self.spread_squeeze.value)
            ),
            'exit_short'] = 1


        return dataframe
