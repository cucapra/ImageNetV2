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
def compress(dir_list,file_list,cmp_dir,uncmp_root,tmp_qtable):
    for dir_in in dir_list:
        if not os.path.exists(os.path.join(cmp_dir,dir_in) ):
            os.makedirs(os.path.join(cmp_dir,dir_in) )
        for file_in in file_list[dir_in]:
            file_out = os.path.join(cmp_dir,dir_in,file_in.replace('bmp','jpg'))
            execute = "./cjpeg -outfile "+file_out+" -quality 50 -qtable "+tmp_qtable+" -qslots 0,1,2 "+os.path.join(uncmp_root,dir_in,file_in)
            os.system(execute)
uncmp_root = '/mnt/tmpfs/matched_frequency_part/'
uncmp_mean = 150582
optimize_root = '/data/zhijing/flickrImageNetV2/random_cache/'#'/mnt/tmpfs/sorted_cache/'
create_dir(optimize_root)
#cmp_dir = os.path.join(optimize_root, 'dataset_pareto_take_firstq')
cmp_dir = optimize_root
create_dir(cmp_dir) 
dir_list = os.listdir(uncmp_root)
file_list = {x:os.listdir(os.path.join(uncmp_root,x)) for x in dir_list }
#qtable_root = os.path.join('/data/zhijing/flickrImageNetV2/sorted_cache/','qtables')
qtable_root = os.path.join(optimize_root, 'qtables')
create_dir(qtable_root)
#data = [x for x in np.load('pareto.npy')]
for i in range(0,1000):#len(data)):
    qname = os.path.join(qtable_root,'qtable'+str(i)+'.txt')
    print(qname)
    ratio.random_qtable_generate(qname)
    #ratio.sorted_qtable_generate(qname)
    ts = []
    partition = int(len(dir_list)/32)
    for j,k in enumerate(range(0,len(dir_list),partition)):
        ts.append( threading.Thread(target=compress,args=(dir_list[k:k+partition],file_list,cmp_dir,uncmp_root,qname)) ) 
        ts[j].start()
    for t in ts:
        t.join()
    cmp_mean,cmp_std = get_size(cmp_dir)
    print(cmp_mean,cmp_std)
    r = uncmp_mean/cmp_mean
    
    sys.argv = ['skip','--dataset']
    acc1,acc5 = eval.run(sys.argv.append(cmp_dir) )
    row = [i,acc1,acc5,r,cmp_mean,cmp_std]
    print(row)
    store_csv_check(row,"csv/random.csv")

