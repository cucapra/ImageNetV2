import numpy as np
from utils import diff_fit
import csv
import pandas as pd
import sys
csv_name = sys.argv[1]
cnt = 0
df = pd.read_csv(csv_name)
#rates = np.array( df['rate'])
#indexes = np.logical_and(df['rate']>21,df['rate']<23)
#df = df[indexes]
i = 0
for index in range(len(df)):
    i+=1
    res = diff_fit(df['acc1'][index],df['rate'][index])
    if res > -0.001:
        cnt+=1
    if cnt == 10:
        break
print(i)
