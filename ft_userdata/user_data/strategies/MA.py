import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np
import talib.abstract as ta
from freqtrade.strategy import IStrategy, informative
from freqtrade.strategy import (merge_informative_pair,
								DecimalParameter, IntParameter, BooleanParameter, CategoricalParameter, stoploss_from_open,
								stoploss_from_absolute)
from pandas import DataFrame, Series
from typing import Dict, List, Optional, Tuple, Union
from functools import reduce
from freqtrade.persistence import Trade
from datetime import datetime, timedelta, timezone
from freqtrade.exchange import timeframe_to_prev_date, timeframe_to_minutes
import talib.abstract as ta
import math
import pandas_ta as pta
import logging
from logging import FATAL

logger = logging.getLogger(__name__)

def tv_wma(df, length = 9) -> DataFrame:
	"""
	Source: Tradingview "Moving Average Weighted"
	Pinescript Author: Unknown
	Args :
		dataframe : Pandas Dataframe
		length : WMA length
		field : Field to use for the calculation
	Returns :
		dataframe : Pandas DataFrame with new columns 'tv_wma'
	"""

	norm = 0
	sum = 0

	for i in range(1, length - 1):
		weight = (length - i) * length
		norm = norm + weight
		sum = sum + df.shift(i) * weight

	tv_wma = (sum / norm) if norm > 0 else 0
	return tv_wma

def tv_hma(dataframe, length = 9, field = 'close') -> DataFrame:
	"""
	Source: Tradingview "Hull Moving Average"
	Pinescript Author: Unknown
	Args :
		dataframe : Pandas Dataframe
		length : HMA length
		field : Field to use for the calculation
	Returns :
		dataframe : Pandas DataFrame with new columns 'tv_hma'
	"""

	h = 2 * tv_wma(dataframe[field], math.floor(length / 2)) - tv_wma(dataframe[field], length)

	tv_hma = tv_wma(h, math.floor(math.sqrt(length)))
	# dataframe.drop("h", inplace=True, axis=1)

	return tv_hma

class MA(IStrategy):

	def version(self) -> str:
		return "Kasuari-v1-btc"

	INTERFACE_VERSION = 3

	lev = 10
	roi = 0.015
	# ROI table:
	minimal_roi = {
		"0": (roi * lev),
		"28": (0.042),
		"53": (0.014),

	}

	optimize_buy_hma1 = True
	buy_length_hma1 = IntParameter(1, 50, default=3, optimize=optimize_buy_hma1)
	buy_offset_hma1 = IntParameter(16, 20, default=19, optimize=optimize_buy_hma1)

	optimize_buy_hma2 = True
	buy_length_hma2 = IntParameter(1, 50, default=44, optimize=optimize_buy_hma2)
	buy_offset_hma2 = IntParameter(16, 20, default=18, optimize=optimize_buy_hma2)

	optimize_buy_hma3 = True
	buy_length_hma3 = IntParameter(1, 50, default=36, optimize=optimize_buy_hma3)
	buy_offset_hma3 = IntParameter(20, 24, default=24, optimize=optimize_buy_hma3)

	optimize_buy_hma4 = True
	buy_length_hma4 = IntParameter(1, 50, default=29, optimize=optimize_buy_hma4)
	buy_offset_hma4 = IntParameter(20, 24, default=23, optimize=optimize_buy_hma4)

	optimize_buy_dema1 = True
	buy_length_dema1 = IntParameter(1, 50, default=4, optimize=optimize_buy_dema1)
	buy_offset_dema1 = IntParameter(16, 20, default=18, optimize=optimize_buy_dema1)

	optimize_buy_dema2 = True
	buy_length_dema2 = IntParameter(1, 50, default=17, optimize=optimize_buy_dema2)
	buy_offset_dema2 = IntParameter(16, 20, default=16, optimize=optimize_buy_dema2)

	optimize_buy_dema3 = True
	buy_length_dema3 = IntParameter(1, 50, default=48, optimize=optimize_buy_dema3)
	buy_offset_dema3 = IntParameter(20, 24, default=22, optimize=optimize_buy_dema3)

	optimize_buy_dema4 = True
	buy_length_dema4 = IntParameter(1, 50, default=28, optimize=optimize_buy_dema4)
	buy_offset_dema4 = IntParameter(20, 24, default=20, optimize=optimize_buy_dema4)

	optimize_buy_tema1 = True
	buy_length_tema1 = IntParameter(1, 40, default=12, optimize=optimize_buy_tema1)
	buy_offset_tema1 = IntParameter(16, 20, default=17, optimize=optimize_buy_tema1)

	optimize_buy_tema2 = True
	buy_length_tema2 = IntParameter(1, 40, default=40, optimize=optimize_buy_tema2)
	buy_offset_tema2 = IntParameter(16, 20, default=16, optimize=optimize_buy_tema2)

	optimize_buy_tema3 = True
	buy_length_tema3 = IntParameter(1, 40, default=40, optimize=optimize_buy_tema3)
	buy_offset_tema3 = IntParameter(20, 24, default=22, optimize=optimize_buy_tema3)

	optimize_buy_tema4 = True
	buy_length_tema4 = IntParameter(1, 40, default=11, optimize=optimize_buy_tema4)
	buy_offset_tema4 = IntParameter(20, 24, default=23, optimize=optimize_buy_tema4)

	optimize_buy_rsi1 = True
	buy_rsi1 = IntParameter(1, 10, default=8, optimize=optimize_buy_rsi1)
	buy_rolling_rsi1 = IntParameter(2, 30, default=9, optimize=optimize_buy_rsi1)
	buy_diff_rsi1 = IntParameter(2, 10, default=2, optimize=optimize_buy_rsi1)

	optimize_buy_rsi2 = True
	buy_rsi2 = IntParameter(10, 19, default=11, optimize=optimize_buy_rsi2)
	buy_rolling_rsi2 = IntParameter(2, 30, default=13, optimize=optimize_buy_rsi2)
	buy_diff_rsi2 = IntParameter(2, 10, default=7, optimize=optimize_buy_rsi2)

	optimize_buy_rsi3 = True
	buy_rsi3 = IntParameter(1, 10, default=7, optimize=optimize_buy_rsi3)
	buy_rolling_rsi3 = IntParameter(2, 30, default=25, optimize=optimize_buy_rsi3)
	buy_diff_rsi3 = IntParameter(2, 10, default=7, optimize=optimize_buy_rsi3)

	optimize_buy_rsi4 = True
	buy_rsi4 = IntParameter(10, 19, default=11, optimize=optimize_buy_rsi4)
	buy_rolling_rsi4 = IntParameter(2, 30, default=19, optimize=optimize_buy_rsi4)
	buy_diff_rsi4 = IntParameter(1, 10, default=5, optimize=optimize_buy_rsi4)

	optimize_buy_mfi_1 = True
	buy_mfi_1 = IntParameter(1, 10, default=3, optimize=optimize_buy_mfi_1)
	buy_rolling_mfi_1 = IntParameter(1, 30, default=20, optimize=optimize_buy_mfi_1)
	buy_diff_mfi_1 = IntParameter(1, 10, default=5, optimize=optimize_buy_mfi_1)

	optimize_buy_mfi_2 = True
	buy_mfi_2 = IntParameter(10, 19, default=19, optimize=optimize_buy_mfi_2)
	buy_rolling_mfi_2 = IntParameter(1, 30, default=3, optimize=optimize_buy_mfi_2)
	buy_diff_mfi_2 = IntParameter(1, 10, default=6, optimize=optimize_buy_mfi_2)

	optimize_buy_mfi_3 = True
	buy_mfi_3 = IntParameter(1, 10, default=6, optimize=optimize_buy_mfi_3)
	buy_rolling_mfi_3 = IntParameter(1, 19, default=13, optimize=optimize_buy_mfi_3)
	buy_diff_mfi_3 = IntParameter(1, 10, default=9, optimize=optimize_buy_mfi_3)

	optimize_buy_mfi_4 = True
	buy_mfi_4 = IntParameter(10, 19, default=11, optimize=optimize_buy_mfi_4)
	buy_rolling_mfi_4 = IntParameter(12, 30, default=28, optimize=optimize_buy_mfi_4)
	buy_diff_mfi_4 = IntParameter(1, 10, default=1, optimize=optimize_buy_mfi_4)

	optimize_sell_ema1 = True
	sell_length_ema1 = IntParameter(1, 50, default=50, optimize=optimize_sell_ema1)
	sell_offset_ema1 = IntParameter(20, 24, default=20, optimize=optimize_sell_ema1)

	optimize_sell_ema2 = True
	sell_length_ema2 = IntParameter(1, 50, default=7, optimize=optimize_sell_ema2)
	sell_offset_ema2 = IntParameter(16, 20, default=17, optimize=optimize_sell_ema2)

	optimize_sell_ema3 = True
	sell_length_ema3 = IntParameter(1, 50, default=41, optimize=optimize_sell_ema3)
	sell_offset_ema3 = IntParameter(16, 20, default=16, optimize=optimize_sell_ema3)

	optimize_sell_ema4 = True
	sell_length_ema4 = IntParameter(1, 50, default=18, optimize=optimize_sell_ema4)
	sell_offset_ema4 = IntParameter(20, 24, default=20, optimize=optimize_sell_ema4)

	sell_clear_old_trade = IntParameter(11, 25, default=20, optimize=False)
	sell_clear_old_trade_profit = IntParameter(0, 5, default=0, optimize=False)

	# Stoploss:
	stoploss = -0.095

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

	timeframe = '15m'
	can_short = True

	timeframe_minutes = timeframe_to_minutes(timeframe)

	process_only_new_candles = True
	startup_candle_count = 999

	def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
		
		dataframe['rsi'] = ta.RSI(dataframe['close'], timeperiod=15)
		dataframe['rsi_45'] = ta.RSI(dataframe['close'], timeperiod=45)

		dataframe['mfi'] = ta.MFI(dataframe, timeperiod=15)
		dataframe['mfi_45'] = ta.MFI(dataframe, timeperiod=45)

		stoch_fast = ta.STOCHF(dataframe, 5, 3, 0, 3, 0)
		dataframe['fastk'] = stoch_fast['fastk']

		dataframe['live_data_ok'] = (dataframe['volume'].rolling(window=72, min_periods=72).min() > 0)

		if not self.optimize_buy_hma1:
			dataframe['hma_offset_buy1'] = tv_hma(dataframe, int(5 * self.buy_length_hma1.value)) * 0.05 * self.buy_offset_hma1.value

		if not self.optimize_buy_hma2:
			dataframe['hma_offset_buy2'] = tv_hma(dataframe, int(5 * self.buy_length_hma2.value)) * 0.05 * self.buy_offset_hma2.value

		if not self.optimize_buy_hma3:
			dataframe['hma_offset_buy3'] = tv_hma(dataframe, int(5 * self.buy_length_hma3.value)) * 0.05 * self.buy_offset_hma3.value

		if not self.optimize_buy_hma4:
			dataframe['hma_offset_buy4'] = tv_hma(dataframe, int(5 * self.buy_length_hma4.value)) * 0.05 * self.buy_offset_hma4.value

		if not self.optimize_buy_dema1:
			dataframe['dema_offset_buy1'] = ta.DEMA(dataframe, int(5 * self.buy_length_dema1.value)) * 0.05 * self.buy_offset_dema1.value

		if not self.optimize_buy_dema2:
			dataframe['dema_offset_buy2'] = ta.DEMA(dataframe, int(5 * self.buy_length_dema2.value)) * 0.05 * self.buy_offset_dema2.value

		if not self.optimize_buy_dema3:
			dataframe['dema_offset_buy3'] = ta.DEMA(dataframe, int(5 * self.buy_length_dema3.value)) * 0.05 * self.buy_offset_dema3.value

		if not self.optimize_buy_dema4:
			dataframe['dema_offset_buy4'] = ta.DEMA(dataframe, int(5 * self.buy_length_dema4.value)) * 0.05 * self.buy_offset_dema4.value

		if not self.optimize_buy_tema1:
			dataframe['tema_offset_buy1'] = ta.TEMA(dataframe, int(5 * self.buy_length_tema1.value)) * 0.05 * self.buy_offset_tema1.value

		if not self.optimize_buy_tema2:
			dataframe['tema_offset_buy2'] = ta.TEMA(dataframe, int(5 * self.buy_length_tema2.value)) * 0.05 * self.buy_offset_tema2.value

		if not self.optimize_buy_tema3:
			dataframe['tema_offset_buy3'] = ta.TEMA(dataframe, int(5 * self.buy_length_tema3.value)) * 0.05 * self.buy_offset_tema3.value

		if not self.optimize_buy_tema4:
			dataframe['tema_offset_buy4'] = ta.TEMA(dataframe, int(5 * self.buy_length_tema4.value)) * 0.05 * self.buy_offset_tema4.value

		if not self.optimize_sell_ema1:
			column_name = f"ema_{int(5 * self.sell_length_ema1.value)}"
			if column_name not in dataframe.columns:
				dataframe[column_name] = ta.EMA(dataframe, int(5 * self.sell_length_ema1.value))

		if not self.optimize_sell_ema2:
			column_name = f"ema_{int(5 * self.sell_length_ema2.value)}"
			if column_name not in dataframe.columns:
				dataframe[column_name] = ta.EMA(dataframe, int(5 * self.sell_length_ema2.value))
		
		if not self.optimize_sell_ema3:
			column_name = f"ema_{int(5 * self.sell_length_ema3.value)}"
			if column_name not in dataframe.columns:
				dataframe[column_name] = ta.EMA(dataframe, int(5 * self.sell_length_ema3.value))

		if not self.optimize_sell_ema4:
			column_name = f"ema_{int(5 * self.sell_length_ema4.value)}"
			if column_name not in dataframe.columns:
				dataframe[column_name] = ta.EMA(dataframe, int(5 * self.sell_length_ema4.value))

		vol_20_max = dataframe['volume'].rolling(window=20).max()
		vol_20_min = dataframe['volume'].rolling(window=20).min()
		roll_20 = ((vol_20_max - dataframe['volume']) / (vol_20_max - vol_20_min))
		dataframe['vol_base'] = roll_20.rolling(5).mean()
		dataframe['vol_20'] = roll_20

		vol_40_max = dataframe['volume'].rolling(window=40).max()
		vol_40_min = dataframe['volume'].rolling(window=40).min()
		roll_40 = ((vol_40_max - dataframe['volume']) / (vol_40_max - vol_40_min))
		dataframe['vol_40_base'] = roll_40.rolling(5).mean()
		dataframe['vol_40'] = roll_40

		return dataframe
	
	def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
		
		conditions = []
		conditions_short = []
		conditions_csi = []

		if self.optimize_buy_hma1:
			dataframe['hma_offset_buy1'] = tv_hma(dataframe, int(5 * self.buy_length_hma1.value)) * 0.05 * self.buy_offset_hma1.value

		if self.optimize_buy_hma2:
			dataframe['hma_offset_buy2'] = tv_hma(dataframe, int(5 * self.buy_length_hma2.value)) * 0.05 * self.buy_offset_hma2.value

		if self.optimize_buy_hma3:
			dataframe['hma_offset_buy3'] = tv_hma(dataframe, int(5 * self.buy_length_hma3.value)) * 0.05 * self.buy_offset_hma3.value

		if self.optimize_buy_hma4:
			dataframe['hma_offset_buy4'] = tv_hma(dataframe, int(5 * self.buy_length_hma4.value)) * 0.05 * self.buy_offset_hma4.value

		if self.optimize_buy_dema1:
			dataframe['dema_offset_buy1'] = ta.DEMA(dataframe, int(5 * self.buy_length_dema1.value)) * 0.05 * self.buy_offset_dema1.value

		if self.optimize_buy_dema2:
			dataframe['dema_offset_buy2'] = ta.DEMA(dataframe, int(5 * self.buy_length_dema2.value)) * 0.05 * self.buy_offset_dema2.value

		if self.optimize_buy_dema3:
			dataframe['dema_offset_buy3'] = ta.DEMA(dataframe, int(5 * self.buy_length_dema3.value)) * 0.05 * self.buy_offset_dema3.value

		if self.optimize_buy_dema4:
			dataframe['dema_offset_buy4'] = ta.DEMA(dataframe, int(5 * self.buy_length_dema4.value)) * 0.05 * self.buy_offset_dema4.value

		if self.optimize_buy_tema1:
			dataframe['tema_offset_buy1'] = ta.TEMA(dataframe, int(5 * self.buy_length_tema1.value)) * 0.05 * self.buy_offset_tema1.value

		if self.optimize_buy_tema2:
			dataframe['tema_offset_buy2'] = ta.TEMA(dataframe, int(5 * self.buy_length_tema2.value)) * 0.05 * self.buy_offset_tema2.value

		if self.optimize_buy_tema3:
			dataframe['tema_offset_buy3'] = ta.TEMA(dataframe, int(5 * self.buy_length_tema3.value)) * 0.05 * self.buy_offset_tema3.value

		if self.optimize_buy_tema4:
			dataframe['tema_offset_buy4'] = ta.TEMA(dataframe, int(5 * self.buy_length_tema4.value)) * 0.05 * self.buy_offset_tema4.value

		dataframe['enter_tag'] = ''
		dataframe['enter_long'] = 0

		add_check = (
			dataframe['live_data_ok']
		)

		no_hma_1 = []

		buy_offset_hma1 = (
			(dataframe['close'] < dataframe['hma_offset_buy1'])
			&
			(dataframe['rsi'] < 35)   #rsi_1
		)
		dataframe.loc[buy_offset_hma1, 'enter_tag'] += 'l_h_1 '
		if no_hma_1:
			conditions.append(buy_offset_hma1 & (reduce(lambda x, y: x | y, no_hma_1) == False))
		else:
			conditions.append(buy_offset_hma1)

		no_hma_2 = []

		buy_offset_hma2 = (
			#根据收盘价与rsi入场
			(dataframe['close'] < dataframe['hma_offset_buy2'])
			&
			(dataframe['rsi_45'] < 50) #rsi_12
			&
			(dataframe['rsi'] < 40)	#rsi_24
		)
		dataframe.loc[buy_offset_hma2, 'enter_tag'] += 'l_h_2 '
		if no_hma_2:
			conditions.append(buy_offset_hma2 & (reduce(lambda x, y: x | y, no_hma_2) == False))
		else:
			conditions.append(buy_offset_hma2)

		no_hma_3 = []


		buy_offset_hma3 = (
			(dataframe['close'] > dataframe['hma_offset_buy3'])
			&
			(dataframe['rsi'] > 60)	#rsi_13
			&
			(dataframe['rsi_45'] < 60)	#rsi_34
		)
		dataframe.loc[buy_offset_hma3, 'enter_tag'] += 's_h_3 '
		if no_hma_3:
			conditions_short.append(buy_offset_hma3 & (reduce(lambda x, y: x | y, no_hma_3) == False))
		else:
			conditions_short.append(buy_offset_hma3)

		no_hma_4 = []

		buy_offset_hma4 = (
			(dataframe['close'] > dataframe['hma_offset_buy4'])
			&
			(dataframe['rsi'] > 65)	 #rsi_4
		)
		dataframe.loc[buy_offset_hma4, 'enter_tag'] += 's_h_4 '
		if no_hma_4:
			conditions_short.append(buy_offset_hma4 & (reduce(lambda x, y: x | y, no_hma_4) == False))
		else:
			conditions_short.append(buy_offset_hma4)

		no_dema_1 = []



		buy_offset_dema1 = (
			((dataframe['close'] < dataframe['dema_offset_buy1']).rolling(2).min() > 0)
			&
			(dataframe['rsi'] < 35)	 #rsi_1
		)
		dataframe.loc[buy_offset_dema1, 'enter_tag'] += 'l_d_1 '
		if no_dema_1:
			conditions.append(buy_offset_dema1 & (reduce(lambda x, y: x | y, no_dema_1) == False))
		else:
			conditions.append(buy_offset_dema1)

		no_dema_2 = []

		buy_offset_dema2 = (
			((dataframe['close'] < dataframe['dema_offset_buy2']).rolling(3).min() > 0)
			&
			(dataframe['rsi_45'] < 50) #rsi_12
			&
			(dataframe['rsi'] < 40)	#rsi_24
		)
		dataframe.loc[buy_offset_dema2, 'enter_tag'] += 'l_dema3_12 '
		if no_dema_2:
			conditions.append(buy_offset_dema2 & (reduce(lambda x, y: x | y, no_dema_2) == False))
		else:
			conditions.append(buy_offset_dema2)

		no_dema_3 = []

		buy_offset_dema3 = (
			(dataframe['close'] > dataframe['dema_offset_buy3'])
			&
			(dataframe['rsi'] > 60)	#rsi_13
			&
			(dataframe['rsi_45'] < 60)	#rsi_34
		)
		dataframe.loc[buy_offset_dema3, 'enter_tag'] += 's_d_3 '
		if no_dema_3:
			conditions_short.append(buy_offset_dema3 & (reduce(lambda x, y: x | y, no_dema_3) == False))
		else:
			conditions_short.append(buy_offset_dema3)

		no_dema_4 = []

		buy_offset_dema4 = (
			((dataframe['close'] > dataframe['dema_offset_buy4']).rolling(3).min() > 0)
			&
			(dataframe['rsi'] > 65)	 #rsi_4
		)
		dataframe.loc[buy_offset_dema4, 'enter_tag'] += 's_dema3_34 '
		if no_dema_4:
			conditions_short.append(buy_offset_dema4 & (reduce(lambda x, y: x | y, no_dema_4) == False))
		else:
			conditions_short.append(buy_offset_dema4)

		no_tema_1 = []

		buy_offset_tema1 = (
			(dataframe['close'] < dataframe['tema_offset_buy1'])
			&
			(dataframe['rsi'] < 35) #rsi_1
		)
		dataframe.loc[buy_offset_tema1, 'enter_tag'] += 'l_t_1 '
		if no_tema_1:
			conditions.append(buy_offset_tema1 & (reduce(lambda x, y: x | y, no_tema_1) == False))
		else:
			conditions.append(buy_offset_tema1)

		no_tema_2 = []

		buy_offset_tema2 = (
			((dataframe['close'] < dataframe['tema_offset_buy2']).rolling(2).min() > 0)
			&
			(dataframe['rsi_45'] < 50) #rsi_12
			&
			(dataframe['rsi'] < 40)	#rsi_24
		)
		dataframe.loc[buy_offset_tema2, 'enter_tag'] += 'l_t_2 '
		if no_tema_2:
			conditions.append(buy_offset_tema2 & (reduce(lambda x, y: x | y, no_tema_2) == False))
		else:
			conditions.append(buy_offset_tema2)

		no_tema_3 = []

		buy_offset_tema3 = (
			(dataframe['close'] > dataframe['tema_offset_buy3'])
			&
			(dataframe['rsi'] > 60)	#rsi_13
			&
			(dataframe['rsi_45'] < 60)	#rsi_34
		)
		dataframe.loc[buy_offset_tema3, 'enter_tag'] += 's_t_3 '
		if no_tema_3:
			conditions_short.append(buy_offset_tema3 & (reduce(lambda x, y: x | y, no_tema_3) == False))
		else:
			conditions_short.append(buy_offset_tema3)

		no_tema_4 = []

		buy_offset_tema4 = (
			(dataframe['close'] > dataframe['tema_offset_buy4'])
			&
			(dataframe['rsi'] > 65)	 #rsi_4
		)
		dataframe.loc[buy_offset_tema4, 'enter_tag'] += 's_t_4 '
		if no_tema_4:
			conditions_short.append(buy_offset_tema4 & (reduce(lambda x, y: x | y, no_tema_4) == False))
		else:
			conditions_short.append(buy_offset_tema4)

		no_rsi_1 = []

		buy_offset_rsi1 = (
			((dataframe['rsi'] < (5 * self.buy_rsi1.value)).rolling(int(self.buy_rolling_rsi1.value)).min() > 0)
			&
			((dataframe['rsi'].shift() - dataframe['rsi']) > (3 * self.buy_diff_rsi1.value))
		)
		dataframe.loc[buy_offset_rsi1, 'enter_tag'] += "l_r_1 "
		if no_rsi_1:
			conditions.append(buy_offset_rsi1 & (reduce(lambda x, y: x | y, no_rsi_1) == False))
		else:
			conditions.append(buy_offset_rsi1)

		no_rsi_2 = []

		buy_offset_rsi2 = (
			((dataframe['rsi'] > (5 * self.buy_rsi2.value)).rolling(int(self.buy_rolling_rsi2.value)).min() > 0)
			&
			((dataframe['rsi'] - dataframe['rsi'].shift()) > (3 * self.buy_diff_rsi2.value))
		)
		dataframe.loc[buy_offset_rsi2, 'enter_tag'] += "s_r_2 "
		if no_rsi_2:
			conditions_short.append(buy_offset_rsi2 & (reduce(lambda x, y: x | y, no_rsi_2) == False))
		else:
			conditions_short.append(buy_offset_rsi2)

		no_rsi_3 = []

		buy_offset_rsi3 = (
			((dataframe['rsi'] < (5 * self.buy_rsi3.value)).rolling(int(self.buy_rolling_rsi3.value)).min() > 0)
			&
			((dataframe['rsi'].shift() - dataframe['rsi']) > (3 * self.buy_diff_rsi3.value))
		)
		dataframe.loc[buy_offset_rsi3, 'enter_tag'] += "l_r_3 "
		if no_rsi_3:
			conditions.append(buy_offset_rsi3 & (reduce(lambda x, y: x | y, no_rsi_3) == False))
		else:
			conditions.append(buy_offset_rsi3)

		no_rsi_4 = []

		buy_offset_rsi4 = (
			((dataframe['rsi'] > (5 * self.buy_rsi4.value)).rolling(int(self.buy_rolling_rsi4.value)).min() > 0)
			&
			((dataframe['rsi'] - dataframe['rsi'].shift()) > (3 * self.buy_diff_rsi4.value))
		)
		dataframe.loc[buy_offset_rsi4, 'enter_tag'] += "s_r_4 "
		if no_rsi_4:
			conditions_short.append(buy_offset_rsi4 & (reduce(lambda x, y: x | y, no_rsi_4) == False))
		else:
			conditions_short.append(buy_offset_rsi4)

		no_mfi_1 = []

		buy_offset_mfi_1 = (
			((dataframe['mfi'] < (5 * self.buy_mfi_1.value)).rolling(int(self.buy_rolling_mfi_1.value)).min() > 0)
			&
			((dataframe['mfi'].shift() - dataframe['mfi']) > (3 * self.buy_diff_mfi_1.value))
		)
		dataframe.loc[buy_offset_mfi_1, 'enter_tag'] += "l_m_1 "
		if no_mfi_1:
			conditions.append(buy_offset_mfi_1 & (reduce(lambda x, y: x | y, no_mfi_1) == False))
		else:
			conditions.append(buy_offset_mfi_1)

		no_mfi_2 = []

		buy_offset_mfi_2 = (
			((dataframe['mfi'] > (5 * self.buy_mfi_2.value)).rolling(int(self.buy_rolling_mfi_2.value)).min() > 0)
			&
			((dataframe['mfi'] - dataframe['mfi'].shift()) > (3 * self.buy_diff_mfi_2.value))
		)
		dataframe.loc[buy_offset_mfi_2, 'enter_tag'] += "s_m_2 "
		if no_mfi_2:
			conditions_short.append(buy_offset_mfi_2 & (reduce(lambda x, y: x | y, no_mfi_2) == False))
		else:
			conditions_short.append(buy_offset_mfi_2)

		no_mfi_3 = []

		buy_offset_mfi_3 = (
			((dataframe['mfi'] < (5 * self.buy_mfi_3.value)).rolling(int(self.buy_rolling_mfi_3.value)).min() > 0)
			&
			((dataframe['mfi'].shift() - dataframe['mfi']) > (3 * self.buy_diff_mfi_3.value))
		)
		dataframe.loc[buy_offset_mfi_3, 'enter_tag'] += "l_m_3 "
		if no_mfi_3:
			conditions.append(buy_offset_mfi_3 & (reduce(lambda x, y: x | y, no_mfi_3) == False))
		else:
			conditions.append(buy_offset_mfi_3)

		no_mfi_4 = []

		buy_offset_mfi_4 = (
			((dataframe['mfi'] > (5 * self.buy_mfi_4.value)).rolling(int(self.buy_rolling_mfi_4.value)).min() > 0)
			&
			((dataframe['mfi'] - dataframe['mfi'].shift()) > (3 * self.buy_diff_mfi_4.value))
		)
		dataframe.loc[buy_offset_mfi_4, 'enter_tag'] += "s_m_4 "
		if no_mfi_4:
			conditions_short.append(buy_offset_mfi_4 & (reduce(lambda x, y: x | y, no_mfi_4) == False))
		else:
			conditions_short.append(buy_offset_mfi_4)

		if conditions:

			no_long = []

			dataframe.loc[
				reduce(lambda x, y: x | y, conditions)
				&
				add_check,
				'enter_long'
			]= 1

			if no_long:
				dataframe.loc[
					reduce(lambda x, y: x | y, no_long),
					'enter_long'
				]= 0

		if conditions_short:

			no_short = []

			dataframe.loc[
				reduce(lambda x, y: x | y, conditions_short)
				&
				add_check,
				'enter_short',
			]= 1

			if no_short:
				dataframe.loc[
					reduce(lambda x, y: x | y, no_short),
					'enter_short',
				]= 0

		return dataframe

	def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

		dataframe['exit_tag'] = ''
		conditions = []
		conditions_short = []

		add_check = (
			(dataframe['volume'] > 0)
		)

		if self.optimize_sell_ema1:
			column_name = f"ema_{int(5 * self.sell_length_ema1.value)}"
			if column_name not in dataframe.columns:
				dataframe[column_name] = ta.EMA(dataframe, int(5 * self.sell_length_ema1.value))

		if self.optimize_sell_ema2:
			column_name = f"ema_{int(5 * self.sell_length_ema2.value)}"
			if column_name not in dataframe.columns:
				dataframe[column_name] = ta.EMA(dataframe, int(5 * self.sell_length_ema2.value))

		if self.optimize_sell_ema3:
			column_name = f"ema_{int(5 * self.sell_length_ema3.value)}"
			if column_name not in dataframe.columns:
				dataframe[column_name] = ta.EMA(dataframe, int(5 * self.sell_length_ema3.value))

		if self.optimize_sell_ema4:
			column_name = f"ema_{int(5 * self.sell_length_ema4.value)}"
			if column_name not in dataframe.columns:
				dataframe[column_name] = ta.EMA(dataframe, int(5 * self.sell_length_ema4.value))

		sell_ema_1 = (
			(dataframe['close'] > (dataframe[f"ema_{int(5 * self.sell_length_ema1.value)}"] * 0.05 * self.sell_offset_ema1.value))
		)
		dataframe.loc[sell_ema_1, 'exit_tag'] += 'l_e_u '
		conditions.append(sell_ema_1)

		sell_ema_2 = (
			(dataframe['close'] < (dataframe[f"ema_{int(5 * self.sell_length_ema2.value)}"] * 0.05 * self.sell_offset_ema2.value))
		)
		dataframe.loc[sell_ema_2, 'exit_tag'] += 'l_e_d '
		conditions.append(sell_ema_2)

		sell_ema_3 = (
		  (dataframe['close'] < (dataframe[f"ema_{int(5 * self.sell_length_ema3.value)}"] * 0.05 * self.sell_offset_ema3.value))
		)
		dataframe.loc[sell_ema_3, 'exit_tag'] += 's_e_d '
		conditions_short.append(sell_ema_3)

		sell_ema_4 = (
		  (dataframe['close'] > (dataframe[f"ema_{int(5 * self.sell_length_ema4.value)}"] * 0.05 * self.sell_offset_ema4.value))
		)
		dataframe.loc[sell_ema_4, 'exit_tag'] += 's_e_u '
		conditions_short.append(sell_ema_4)

		if conditions:
			dataframe.loc[
				reduce(lambda x, y: x | y, conditions)
				&
				add_check,
				'exit_long',
			]= 1

		if conditions_short:
			dataframe.loc[
				reduce(lambda x, y: x | y, conditions_short)
				&
				add_check,
				'exit_short',
			]= 1

		return dataframe

	def leverage(self, pair: str, current_time: datetime, current_rate: float,
				 proposed_leverage: float, max_leverage: float, entry_tag: Optional[str], side: str,
				 **kwargs) -> float:

		return self.lev

	def custom_exit(self, pair: str, trade: Trade, current_time: datetime, current_rate: float, current_profit: float, **kwargs) -> Optional[Union[str, bool]]:

		if (current_time - timedelta(minutes=int(self.timeframe_minutes)) >= trade.open_date_utc):

			dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
			current_candle = dataframe.iloc[-1].squeeze()

			current_profit = trade.calc_profit_ratio(current_candle['close'])
			
		if current_time - timedelta(minutes=int(self.timeframe_minutes * self.sell_clear_old_trade.value)) >= trade.open_date_utc:
			if (current_profit >= (-0.01 * self.sell_clear_old_trade_profit.value)):
				return "sell_old_trade"
