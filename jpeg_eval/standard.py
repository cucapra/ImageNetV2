import os, sys,shutil
import threading
import csv
import pandas as pd
import matplotlib.pyplot as plt
import eval
import glob
from PIL import Image
import numpy as np
from scipy import stats
from ratio import *
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
        store_csv(['i','acc1','acc5','rate','mpsnr','cmp_mean','cmp_std'],name)
    store_csv(row,name)
def read_csv(name):
    df = pd.read_csv(name) 
    dics = []
    for i,r in enumerate(df['rate']):
        dic = {str(round(r)) :(df['acc1'][i],df['acc5'][i] )}
        print(dic)
        dics.append(dic)
    return dics
#store_csv_check(['-1','0.7561', '0.9309','1','150582','0'],'standard.csv')
def get_size(start_path):
    total_size = []
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                total_size.append(os.path.getsize(fp))
    total_size = np.array(total_size)
    #a,loc,scale=stats.skewnorm.fit(np.array(total_size))
    #sigma = a/np.sqrt(1+a*a)
    #mean = loc + scale*sigma*np.sqrt(2/np.pi)
    #var = scale**2*(1-sigma**2*2/np.pi)
    #std = np.sqrt(var)
    return total_size.mean(),total_size.std()
def PSNR(img1, img2):
    print(img1,img2)
    img1 = cv2.imread(img1)
    img2 = cv2.imread(img2)
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
def get_psnr(dir_list,file_list,cmp_dir,uncmp_root):
    psnrs = []
    for dir_in in dir_list:
        for file_in in file_list[dir_in]:
            write_to = os.path.join(cmp_dir,dir_in,file_in.replace('bmp','jpg'))
            read_from = os.path.join(uncmp_root,dir_in,file_in)
            psnrs.append(PSNR(read_from,write_to))
    return np.array(psnrs).mean()

def compress(quality,dir_list,file_list,cmp_dir,uncmp_root):
    for dir_in in dir_list:
        if not os.path.exists(os.path.join(cmp_dir,dir_in) ):
            os.makedirs(os.path.join(cmp_dir,dir_in) )
        for file_in in file_list[dir_in]:
            file_out = os.path.join(cmp_dir,dir_in,file_in.replace('bmp','jpg'))
            execute = "./cjpeg -outfile "+file_out+" -quality "+quality+" "+os.path.join(uncmp_root,dir_in,file_in)
            os.system(execute)
def create_dir(n):
    if not os.path.exists(n):
        os.makedirs(n)

def run():
    uncmp_root = '/data/zhijing/flickrImageNetV2/matched_frequency_part/'#part,224
    uncmp_mean = 150582
    optimize_root = '/data/zhijing/flickrImageNetV2/cmp/matched_part/'
    create_dir(optimize_root)
    dir_list = os.listdir(uncmp_root)
    file_list = {x:os.listdir(os.path.join(uncmp_root,x)) for x in dir_list }
    
    csv_name = "standard_part.csv"
    
    for i in range(5,101,5):
        cmp_dir = os.path.join(optimize_root,'quality'+str(i))
        if not os.path.exists(cmp_dir):
            os.makedirs(cmp_dir)
        #if i == 0:
        #    shutil.copyfile('qtables/qtable.txt',tmp_qtable)

        ts = []
        partition = int(len(dir_list)/20)
        for j,k in enumerate(range(0,len(dir_list),partition)):
            ts.append( threading.Thread(target=compress,args=(str(i),dir_list[k:k+partition],file_list,cmp_dir,uncmp_root))) 
            ts[j].start()
        for t in ts:
            t.join()
        mpsnr = get_psnr(dir_list,file_list,cmp_dir,uncmp_root)
        print('mpsnr',mpsnr)
        cmp_mean,cmp_std = get_size(cmp_dir)
        print(cmp_mean,cmp_std)
        r = uncmp_mean/cmp_mean
        
        sys.argv = ['skip','--dataset']
        acc1,acc5 = eval.run(sys.argv.append(cmp_dir) )
        row = [i,acc1,acc5,r,mpsnr,cmp_mean,cmp_std]
        print(row)
        store_csv_check(row, csv_name)
        
if __name__=='__main__':
    sys.exit(run())
