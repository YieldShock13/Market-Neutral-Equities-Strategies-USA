"""Operational cutoff only; no changes to model estimation or eligibility."""
import pandas as pd
import exchange_calendars as xcals

def latest_completed_session(now=None):
    now=pd.Timestamp.now(tz='UTC') if now is None else pd.Timestamp(now)
    now=now.tz_localize('UTC') if now.tzinfo is None else now.tz_convert('UTC')
    calendar=xcals.get_calendar('XNYS')
    dates=calendar.sessions_in_range((now-pd.Timedelta(days=14)).date(),now.date())
    completed=[date for date in dates if calendar.session_close(date)<=now]
    if not completed:
        raise RuntimeError('No completed reference exchange session found')
    date=pd.Timestamp(completed[-1])
    return date.tz_localize(None) if date.tzinfo else date
