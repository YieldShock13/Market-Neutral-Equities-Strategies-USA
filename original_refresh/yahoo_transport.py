"""Yahoo I/O compatibility and bounded retries; no statistical calculations."""
import time
import pandas as pd
import yfinance as yf

def normalize_download(data, tickers):
    if len(tickers)==1 and isinstance(data.columns,pd.MultiIndex):
        symbol=tickers[0]
        for level in range(data.columns.nlevels):
            if symbol in data.columns.get_level_values(level):
                data=data.xs(symbol,axis=1,level=level)
                break
    if data.empty:
        raise RuntimeError('Yahoo returned no prices; no replacement values will be created')
    if len(tickers)==1 and 'Adj Close' not in data.columns:
        raise RuntimeError('Yahoo did not return adjusted close; raw close will not be substituted')
    return data

class YahooUnavailable(BaseException):
    pass

def install_retries():
    original=yf.Ticker
    class RetryingTicker:
        def __init__(self,*args,**kwargs):
            self._ticker=original(*args,**kwargs)
        @property
        def info(self):
            return self._retry(lambda:self._ticker.info)
        def history(self,*args,**kwargs):
            return self._retry(lambda:self._ticker.history(*args,**kwargs))
        def _retry(self,call):
            for attempt in range(6):
                try:
                    result=call()
                    if result is None or (hasattr(result,'empty') and result.empty) or (isinstance(result,dict) and not result):
                        raise RuntimeError('Empty Yahoo response')
                    return result
                except Exception as exc:
                    if attempt==5:raise YahooUnavailable("Yahoo requests remain unavailable after retries; refresh stopped without publishing") from exc
                    delay=min(300,60*(attempt+1))
                    print(f'Yahoo request retry {attempt+1}/5; waiting {delay}s: {type(exc).__name__}',flush=True)
                    time.sleep(delay)
        def __getattr__(self,name):
            return getattr(self._ticker,name)
    yf.Ticker=RetryingTicker
