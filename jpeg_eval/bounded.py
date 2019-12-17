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
optimize_root = '/data/zhijing/flickrImageNetV2/bound_cache/'#'/mnt/tmpfs/sorted_cache/'
create_dir(optimize_root)
#cmp_dir = os.path.join(optimize_root, 'dataset_pareto_take_firstq')
cmp_dir = os.path.join(optimize_root,'cache')
create_dir(cmp_dir) 
dir_list = os.listdir(uncmp_root)
file_list = {x:os.listdir(os.path.join(uncmp_root,x)) for x in dir_list }
#qtable_root = os.path.join('/data/zhijing/flickrImageNetV2/sorted_cache/','qtables')
qtable_root = os.path.join(optimize_root, 'qtables')
create_dir(qtable_root)
#data = [x for x in np.load('pareto.npy')]
for i in range(1,1000):#len(data)):
    qname = os.path.join(qtable_root,'qtable'+str(i)+'.txt')
    print(qname)
    ratio.bound_qtable_generate(qname)
    #ratio.sorted_qtable_generate(qname)
    ts = []
    partition = int(len(dir_list)/32)
    for j,k in enumerate(range(0,len(dir_list),partition)):
        ts.append( threading.Thread(target=compress,args=(dir_list[k:k+partition],file_list,cmp_dir,uncmp_root,qname)) ) 
        ts[j].start()
    print('start all threads')
    for t in ts:
        t.join()
    print('done compression')
    cmp_mean,cmp_std = get_size(cmp_dir)
    print(cmp_mean,cmp_std)
    r = uncmp_mean/cmp_mean
    
    sys.argv = ['skip','--dataset']
    acc1,acc5 = eval.run(sys.argv.append(cmp_dir) )
    row = [i,acc1,acc5,r,cmp_mean,cmp_std]
    print(row)
    store_csv_check(row,"csv/bound.csv")

