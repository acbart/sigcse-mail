from datetime import timedelta, datetime

import pandas as pd

HOUR_LIST = ['12am', '1am', '2am', '3am', '4am', '5am',
             '6am', '7am', '8am', '9am', '10am', '11am', 
             '12pm', '1pm', '2pm', '3pm', '4pm', '5pm',
             '6pm', '7pm', '8pm', '9pm', '10pm', '11pm']
             
DAYS_OF_WEEK = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

SHORT_MONTH_NAMES = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul',
                     'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

MONTH_MAP = {
    "January": 1,
    "February": 2,
    "March": 3,
    "April": 4,
    "May": 5,
    "June": 6,
    "July": 7,
    "August": 8,
    "September": 9,
    "October": 10,
    "November": 11,
    "December": 12
}

NUM_MONTH = {v: k for k,v in MONTH_MAP.items()}

timezone_map = {'cst': -600,
                'est': -500,
                'cen': -600,
                'cdt': -500,
                'edt': -400,
                'pst': -800,
                'pdt': -700,
                'cat-2': 200,
                'gmt': 0,
                'est-10': -1000,
                'est5dst': -500,
                '-dlst': 0,
                'mdt': -600,
                'mst': -700,
                '+03d0': 300,
                'utc': 0
                }
                
def strptime_with_offset(string, format="%a, %d %b %Y %H:%M:%S"):
    '''
    Converts the string into a Pandas datetime object.
    '''
    string, timezone = string.rsplit(maxsplit=1)
    timezone = timezone.replace('--', '-')
    base_dt = datetime.strptime(string, format)
    if '-' in timezone:
        timezone = '-'+timezone.rsplit('-', maxsplit=1)[1]
    elif '+' in timezone:
        timezone = "+"+timezone.rsplit('+', maxsplit=1)[1]
    if timezone.lower() in timezone_map:
        offset = timezone_map[timezone.lower()]
    else:
        offset = int(timezone)
    delta = timedelta(hours=offset/100, minutes=offset%100)
    actual = base_dt + delta
    if actual.year < 1970:
        return pd.NaT
    else:
        return pd.to_datetime(actual)


def to_2digit_month(month):
    '''
    Args:
        month (str)
    Returns:
        str
    '''
    return '{0:0>2}'.format(month)
