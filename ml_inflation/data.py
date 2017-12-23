"""
Module to help get inflation data from various sources to aid in machine learning
"""

import datetime

import pandas as pd
import pandas_datareader.data as web

import attr


@attr.s
class InflationSeries(object):
    """
    Holder for an inflation series.  Basically provides an easy way to get a pd.Series
    of an inflation component from FRED
    """
    series_code = attr.ib(validator=attr.validators.instance_of(str), type=str)
    series_name = attr.ib(validator=attr.validators.instance_of(str), type=str)

    _series = None

    def get_series(self, force_reload: bool = False) -> pd.Series:
        """
        Get the series.  Uses memoization unless force_reload is true
        :param force_reload: get the data no matter what
        :return: pd.Series
        """
        if self._series is None or force_reload:
            # going to use a really old start date
            raw = web.DataReader(self.series_code, "fred",
                                 datetime.datetime(1913, 1, 1),
                                 datetime.datetime.now())
            raw.columns = [self.series_name]
            self._series = raw.ix[:, 0]

        return self._series

    def get_monthly_change(self, force_reload: bool = False) -> pd.Series:
        """
        Get the series as a mom percentage change.
        :param force_reload: download the data no matter what
        :return: pd.Series
        """
        raw = self.get_series(force_reload)
        return raw.pct_change(1).dropna()

    @staticmethod
    def get_core():
        """
        Get core inflation
        :return: pd.Series of core inflation
        """
        return InflationSeries(series_code='CPILFESL', series_name='Core')
