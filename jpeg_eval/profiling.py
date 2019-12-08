from bayes_opt.event import Events
from bayes_opt.util import load_logs

import random
import ratio
import numpy as np
import threading
import eval
from utils import *
import pandas as pd
import sys
import datetime
n = 500
time = datetime.timedelta(0)
for i in range(n):
    t1 = datetime.datetime.now()
    ratio.sorted_qtable_generate('../tmp')
    t2 = datetime.datetime.now()
    time += t2-t1
print('sorted profiling: ',time/n)
time = datetime.timedelta(0)
for i in range(n):
    t1 = datetime.datetime.now()
    ratio.bound_qtable_generate('../tmp')
    t2 = datetime.datetime.now()
    time += t2-t1
print('bounded profiling: ',time/n)

