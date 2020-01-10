import os, sys,shutil
os.chdir('../jpeg_eval/')
sys.path.insert(1, '../jpeg_eval/')
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

def store_csv(row, name):
    with open(name, 'a') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(row)
    csvFile.close()
def store_csv_check(row,name):
    if not os.path.isfile(name):
        store_csv(['i','acc1','acc5','rate','cmp_mean','cmp_std'],name)
    store_csv(row,name)
def get_size(start_path):
    total_size = []
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                total_size.append(os.path.getsize(fp))
    total_size = np.array(total_size)
    
    return total_size.mean(),total_size.std()

def create_dir(n):
    if not os.path.exists(n):
        os.makedirs(n)
uncmp_mean = 150582
main_dir = '/data/zhijing/flickrImageNetV2/stats2'
type_dir = os.listdir(main_dir)
methods = ['bound589','bys132','mab203']
for t in type_dir:
    method_dir = os.listdir(os.path.join(main_dir, t))
    for m in method_dir:
        if m not in methods:
            continue
        ind_dir = os.listdir(os.path.join(main_dir, t, m))
        for ind in ind_dir:
            cmp_dir =  os.path.join(main_dir, t, m, ind)
            cmp_mean,cmp_std = get_size(cmp_dir)
            print(cmp_mean,cmp_std)
            r = uncmp_mean/cmp_mean
            
            sys.argv = ['skip','--dataset']
            acc1,acc5 = eval.run(sys.argv.append(cmp_dir) )
            row = [cmp_dir,acc1,acc5,r,cmp_mean,cmp_std]
            print(row)
            store_csv_check(row,"csv/stats2.csv")

