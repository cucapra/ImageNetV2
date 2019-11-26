import os, sys,shutil
import threading
import csv
import pandas as pd
import ratio
import eval
import glob
from PIL import Image
import numpy as np
from scipy import stats
import random
import click
from utils import *

uncmp_root = '/mnt/tmpfs/matched_frequency_part/'
uncmp_mean = 150582
cmp_dir = '/mnt/tmpfs/sorted_cache/'
create_dir(cmp_dir) 
dir_list = os.listdir(uncmp_root)
file_list = {x:os.listdir(os.path.join(uncmp_root,x)) for x in dir_list }
qtable_root = "/mnt/tmpfs/sorted_qtables/" 
df = pd.read_csv("csv/sorted.csv")
scores=np.array((df['acc1'],df['rate']))
scores=np.swapaxes(scores,0,1)
#data = [x for x in np.load('pareto.npy')]
for i in range(501,len(scores)):
    qname = os.path.join(qtable_root,'qtable'+str(i)+'.txt')
    print(qname)
    ts = []
    partition = int(len(dir_list)/50)
    for j,k in enumerate(range(0,len(dir_list),partition)):
        ts.append( threading.Thread(target=compress,args=(dir_list[k:k+partition],file_list,cmp_dir,uncmp_root,qname)) ) 
        ts[j].start()
    for t in ts:
        t.join()
    psnr_value = get_psnr(uncmp_root,cmp_dir)
    row = ['qtable'+str(i)+'.txt',scores[i][0],scores[i][1],psnr_value]
    print(row)
    store_csv_check(row,"sorted_psnr.csv",['qname','acc1','rate', 'psnr_mean'])
