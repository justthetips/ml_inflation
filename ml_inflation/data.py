import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader.data as web

import datetime
import attr


@attr.s
class InflationSeries(object):
    series_code = attr.ib(validator=attr.validators.instance_of(str), type=str)
    series_name = attr.ib(validator=attr.validators.instance_of(str), type=str)

    _series = None

    def get_series(self, force_reload: bool = False) -> pd.Series:
        if self._series is None or force_reload:
            # going to use a really old start date
            raw = web.DataReader(self.series_code, "fred",
                                          datetime.datetime(1913, 1, 1),
                                          datetime.datetime.now())
            raw.columns = [self.series_name]
            self._series = raw.ix[:,0]

        return self._series

    def get_monthly_change(self, force_reload: bool = False) -> pd.Series:
        raw = self.get_series(force_reload)
        return raw.pct_change(1).dropna()

    @staticmethod
    def get_core():
        return InflationSeries(series_code='CPILFESL', series_name='Core')







def foo():
    core = InflationSeries.get_core()
    print(core.get_series())
    print(type(core.get_series()))


if __name__ == '__main__':
    foo()