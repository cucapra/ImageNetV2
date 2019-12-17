import os, sys,shutil
import re
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
from utils import *
part = 'part3'
def run():
    uncmp_root = '/data/zhijing/flickrImageNetV2/matched_frequency_'+part#part,224
    #uncmp_root = '/data/zhijing/flickrImageNetV2/'+part+'_224'
    uncmp_mean = 150582
    optimize_root = '/data/zhijing/flickrImageNetV2/cmp/matched_'+part
    #optimize_root = '/data/zhijing/flickrImageNetV2/cmp/'+part
    create_dir(optimize_root)
    cmp_dir = os.path.join(optimize_root,'cache')
    create_dir(cmp_dir)
    dir_list = os.listdir(uncmp_root)
    file_list = {x:os.listdir(os.path.join(uncmp_root,x)) for x in dir_list }
    qtable_root = "/mnt/tmpfs/sorted_qtables/" 
    csv_name = "csv/standard_"+part+".csv"

    ga_root = '/data/zhijing/flickrImageNetV2/ga_cache/'
    gad  = os.listdir(ga_root)
    gadd = [g for g in gad if g.startswith('ga_selection')]
    indexes = []
    for ga_dir in gadd:
        gas = os.listdir(os.path.join(ga_root,ga_dir,'qtables'))
        res = []
        for ga_f in gas:
            res.append( int(re.findall('\d+',ga_f)[0]) )
            
        indexes.append(os.path.join(ga_root,ga_dir,'qtables','ga'+str(max(res))+'_0.txt'))
    for qname in range(0):#indexes:
        
        if not os.path.exists(cmp_dir):
            os.makedirs(cmp_dir)
        #if i == 0:
        #    shutil.copyfile('qtables/qtable.txt',tmp_qtable)

        ts = []
        partition = int(len(dir_list)/20)
        for j,k in enumerate(range(0,len(dir_list),partition)):
            ts.append( threading.Thread(target=compress,args=(dir_list[k:k+partition],file_list,cmp_dir,uncmp_root, qname))) 
            ts[j].start()
        for t in ts:
            t.join()
        #mpsnr = get_psnr(dir_list,file_list,cmp_dir,uncmp_root)
        #print('mpsnr',mpsnr)
        cmp_mean,cmp_std = get_size(cmp_dir)
        r = uncmp_mean/cmp_mean
        
        sys.argv = ['skip','--dataset']
        acc1,acc5 = eval.run(sys.argv.append(cmp_dir) )
        fitness = diff_fit(acc1, r)
        row = [qname,acc1,acc5,r,fitness,part]
        print(row)
        store_csv_check(row, csv_name)


    indexes = np.load('pareto1000.npy')
    wrong = np.load('pareto.npy')

    res = np.array(list(set(indexes[indexes<=842])^set(wrong[wrong<=842])))
    indexes = sorted(np.hstack((res,indexes[indexes > 842])))
    
    for i in indexes:
        qname = os.path.join(qtable_root,'qtable'+str(i)+'.txt')
        print(qname)
        if not os.path.exists(cmp_dir):
            os.makedirs(cmp_dir)
        #if i == 0:
        #    shutil.copyfile('qtables/qtable.txt',tmp_qtable)

        ts = []
        partition = int(len(dir_list)/20)
        for j,k in enumerate(range(0,len(dir_list),partition)):
            ts.append( threading.Thread(target=compress,args=(dir_list[k:k+partition],file_list,cmp_dir,uncmp_root, qname))) 
            ts[j].start()
        for t in ts:
            t.join()
        #mpsnr = get_psnr(dir_list,file_list,cmp_dir,uncmp_root)
        #print('mpsnr',mpsnr)
        cmp_mean,cmp_std = get_size(cmp_dir)
        print(cmp_mean,cmp_std)
        r = uncmp_mean/cmp_mean
        
        sys.argv = ['skip','--dataset']
        acc1,acc5 = eval.run(sys.argv.append(cmp_dir) )
        fitness = diff_fit(acc1, r)
        row = ['qtable'+str(i),acc1,acc5,r,fitness,part]
        print(row)
        store_csv_check(row, csv_name)

    for i in range(5,101,5):
        cmp_dir = os.path.join(optimize_root,'quality'+str(i))
        if not os.path.exists(cmp_dir):
            os.makedirs(cmp_dir)
        if i == 0:
            shutil.copyfile('qtables/qtable.txt',tmp_qtable)

        ts = []
        partition = int(len(dir_list)/20)
        for j,k in enumerate(range(0,len(dir_list),partition)):
            ts.append( threading.Thread(target=compress_quality,args=(str(i),dir_list[k:k+partition],file_list,cmp_dir,uncmp_root))) 
            ts[j].start()
        for t in ts:
            t.join()
        #mpsnr = get_psnr(dir_list,file_list,cmp_dir,uncmp_root)
        #print('mpsnr',mpsnr)
        cmp_mean,cmp_std = get_size(cmp_dir)
        print(cmp_mean,cmp_std)
        r = uncmp_mean/cmp_mean
        
        sys.argv = ['skip','--dataset']
        acc1,acc5 = eval.run(sys.argv.append(cmp_dir) )
        fitness = diff_fit(acc1, r)
        row = [i,acc1,acc5,r,fitness,part]
        print(row)
        store_csv_check(row, csv_name)
        
if __name__=='__main__':
    sys.exit(run())
