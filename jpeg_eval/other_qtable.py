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
import cv2
import math
def store_csv(row, name):
    with open(name, 'a') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(row)
    csvFile.close()
def store_csv_check(row,name):
    if not os.path.isfile(name):
        store_csv(['qname','acc1','acc5','rate','mpsnr','cmp_mean','cmp_std'],name)
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

def PSNR(img1, img2):
    #print(img1,img2)
    img1 = cv2.imread(img1)
    img2 = cv2.imread(img2)
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def compress(j,dir_list,file_list,cmp_dir,uncmp_root,tmp_qtable):
    psnrs = []
    for dir_in in dir_list:
        if not os.path.exists(os.path.join(cmp_dir,dir_in) ):
            os.makedirs(os.path.join(cmp_dir,dir_in) )
        for file_in in file_list[dir_in]:
            write_to = os.path.join(cmp_dir,dir_in,file_in.replace('bmp','jpg'))
            read_from = os.path.join(uncmp_root,dir_in,file_in)
            execute = "./cjpeg -outfile "+write_to+" -quality 50 -qtable "+tmp_qtable+" -qslots 0 "+ read_from           
            os.system(execute)
            psnrs.append( PSNR(read_from,write_to))
    #print(j,len(dir_list),len(psnrs))
    #print('mean,',np.array(psnrs).mean())
    psnr_dir[j] = np.array(psnrs).mean()

uncmp_root = '/data/zhijing/flickrImageNetV2/matched_frequency_part/'
uncmp_mean = 150582
optimize_root = '/data/zhijing/flickrImageNetV2/otherq_cache/'
create_dir(optimize_root)
cmp_dir = os.path.join(optimize_root, 'dataset')
create_dir(cmp_dir) 
dir_list = os.listdir(uncmp_root)
file_list = {x:os.listdir(os.path.join(uncmp_root,x)) for x in dir_list }
qtable_root = os.path.join(optimize_root,'qtables')
create_dir(qtable_root)
qnames = os.listdir('otherq')
psnr_dir = [None]*20
for name in qnames:
    qname = os.path.join('otherq',name)
    ts = []
    partition = int(len(dir_list)/20)
    for j,k in enumerate(range(0,len(dir_list),partition)):
        ts.append( threading.Thread(target=compress,args=(j,dir_list[k:k+partition],file_list,cmp_dir,uncmp_root,qname)) )
        ts[j].start()
    for t in ts:
        t.join()
    p = np.array(psnr_dir)
    mpsnr=np.mean(p[~np.isnan(p)])
    print('mpsnr',mpsnr)
    cmp_mean,cmp_std = get_size(cmp_dir)
    print(cmp_mean,cmp_std)
    r = uncmp_mean/cmp_mean
    
    sys.argv = ['skip','--dataset']
    acc1,acc5 = eval.run(sys.argv.append(cmp_dir) )
    row = [name,acc1,acc5,r,mpsnr,cmp_mean,cmp_std]
    print(row)
    store_csv_check(row,"other_qtable.csv")

