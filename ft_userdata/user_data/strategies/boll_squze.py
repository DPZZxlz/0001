from freqtrade.strategy import IStrategy
from freqtrade.strategy import merge_informative_pair
from freqtrade.strategy import DecimalParameter
from freqtrade.exchange import timeframe_to_minutes
from pandas import DataFrame
import talib.abstract as ta

class BollSquze(IStrategy):
    INTERFACE_VERSION = 3

    lev = 10

    roi = 0.0015

	# ROI table:
    minimal_roi = {
		"0": (roi * lev)
	}

	# Stoploss:
    stoploss = -0.99

	# Trailing stop:
    trailing_stop = False
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.03
    trailing_only_offset_is_reached = True

	# Sell signal
    use_exit_signal = True
    exit_profit_only = False
    exit_profit_offset = 0.01
    ignore_roi_if_entry_signal = False
    # Optimal timeframe for the strategy.

    timeframe = "1h"
    can_short = True
    # Define parameters
    bb_squeeze_length = DecimalParameter(80, 200, default=100, space='buy', optimize=True)
    bb_squeeze_threshold = DecimalParameter(30, 110, default=60, space='buy', optimize=True)
    trailing_stop = DecimalParameter(0.01, 0.05, default=0.03, space='sell', optimize=True)

    # These values can be overridden in the config.
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Calculate Bollinger Bands
        bb_period = int(self.bb_squeeze_length.value)
        dataframe['upper'], dataframe['middle'], dataframe['lower'] = self.bbands(dataframe, bb_period)

        # Calculate Bollinger Band Squeeze
        # dataframe['bb_squeeze'] = (dataframe['upper'] - dataframe['lower']) / dataframe['middle']
        dataframe['bb_squeeze'] = (dataframe['upper'] - dataframe['lower']) / ta.SMA(dataframe['upper'] - dataframe['lower'], bb_period) * 100


        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        sqz_threshold = self.bb_squeeze_threshold.value

        # Define buy signal
        dataframe.loc[
            (
                (dataframe['close'] > dataframe['middle']) &  # Previous candle close below middle band
                (dataframe['close'] > dataframe['middle']) &  # Current price above middle band
                (dataframe['bb_squeeze'] > sqz_threshold)  # Squeeze is higher than threshold
            ),
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        sqz_threshold = self.bb_squeeze_threshold.value


        # Define sell signal
        dataframe.loc[
            (
                (dataframe['close'] < dataframe['middle']) &  # Previous candle close above middle band
                (dataframe['close'] < dataframe['middle']) &  # Current price below middle band
                (dataframe['bb_squeeze'] > sqz_threshold)  # Squeeze is higher than threshold
            ),
            'sell'] = 1

        return dataframe


