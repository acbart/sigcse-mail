import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from itertools import starmap

def plt_stagger_xticks(series, stagger=20):
    plt.xticks(np.arange(0, len(series), stagger),
               series[::stagger], rotation=45)
               
from itertools import starmap

def flatten_multiple_indexes(midx, sep=''):
    fstr = sep.join(['{}'] * midx.nlevels)
    return pd.Index(starmap(fstr.format, midx))
