"""Use only exchange sessions whose scheduled close has passed."""
import pandas as pd
import exchange_calendars as xcals

def completed_sessions(now=None, years=11):
    now=pd.Timestamp.now(tz='UTC') if now is None else pd.Timestamp(now)
    now=now.tz_localize('UTC') if now.tzinfo is None else now.tz_convert('UTC')
    first=(now-pd.DateOffset(years=years)).normalize().tz_localize(None)
    last=now.normalize().tz_localize(None)
    calendar=xcals.get_calendar('XNYS',start=first-pd.Timedelta(days=7),end=last+pd.Timedelta(days=7))
    sessions=calendar.sessions_in_range(first,last)
    completed=[s for s in sessions if calendar.session_close(s)<=now]
    index=pd.DatetimeIndex(completed)
    return index.tz_localize(None).normalize() if index.tz is not None else index.normalize()


def rolling_ten_trading_years(now=None):
    from eligibility import REQUIRED_SESSIONS
    sessions=completed_sessions(now,years=12)
    if len(sessions)<REQUIRED_SESSIONS:
        raise ValueError("Insufficient exchange-calendar coverage")
    return sessions[-REQUIRED_SESSIONS:]
