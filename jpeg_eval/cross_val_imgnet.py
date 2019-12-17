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
part = 'imagenet'
def run():
    uncmp_root = '/data/zhijing/cs6787/imagenet_224/val/'
    uncmp_mean = 150582
    optimize_root = '/data/zhijing/cs6787/imagenet_cmp/val/'
    #optimize_root = '/data/zhijing/flickrImageNetV2/cmp/'+part
    create_dir(optimize_root)
    
    dir_list = os.listdir(uncmp_root)
    file_list = {x:os.listdir(os.path.join(uncmp_root,x)) for x in dir_list }
    csv_name = "csv/standard_"+part+".csv"


    cmp_dir = os.path.join(optimize_root,'cache')
    create_dir(cmp_dir)
    
    qtable_root = "/data/zhijing/flickrImageNetV2/sorted_cache/qtables" 
    
    indexes = np.load('pareto1000.npy')
    #wrong = np.load('pareto.npy')

    #res = np.array(list(set(indexes[indexes<=842])^set(wrong[wrong<=842])))
    #indexes = sorted(np.hstack((res,indexes[indexes > 842])))
    
    for i in indexes:
        qname = os.path.join(qtable_root,'qtable'+str(i)+'.txt')
        print(qname)

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

    

    cmp_dir = os.path.join(optimize_root,'quality'+str(i))
    if not os.path.exists(cmp_dir):
        os.makedirs(cmp_dir)
    for i in range(5,101,5):

        ts = []
        partition = int(len(dir_list)/20)
        for j,k in enumerate(range(0,len(dir_list),partition)):
            ts.append( threading.Thread(target=compress_quality,args=(str(i),dir_list[k:k+partition],file_list,cmp_dir,uncmp_root))) 
            ts[j].start()
        for t in ts:
            t.join()
        cmp_mean,cmp_std = get_size(cmp_dir)
        print(cmp_mean,cmp_std)
        r = uncmp_mean/cmp_mean
        
        sys.argv = ['skip','--dataset']
        acc1,acc5 = eval.run(sys.argv.append(cmp_dir) )
        fitness = diff_fit(acc1, r)
        row = [i,acc1,acc5,r,fitness,part]
        print(row)
        store_csv_check(row, csv_name)
#    cmp_dir = os.path.join(optimize_root,'mab_cache')
#    create_dir(cmp_dir)
#    mab_root = '/data/zhijing/flickrImageNetV2/mab_cache/'
#    create_dir(mab_root)
#    qtable_root = os.path.join(mab_root,'qtables')
#    create_dir(qtable_root)
#    
#
#    df = pd.read_csv('csv/mab_bounded.csv')
#    scores = np.array((df['rate'], df['acc1']))
#    scores = np.swapaxes(scores, 0, 1)
#    indexes = identify_pareto(scores)
#    np.save('mab.npy', indexes)
#    print(np.load('mab.npy'))
    indexes = np.load('mab.npy')
    df = pd.read_csv('csv/mab_bounded_qtable.csv')
    for index in range(indexes):
        qtable = np.array([df['q'+str(i).zfill(2)][index] for i in range(64)])
        qtable = qtable.reshape((8,8))
        write_qtable(qtable,qname=os.path.join(qtable_root, 'qtable'+str(index)+'.txt'))
    
    for i in indexes:
        qname = os.path.join(qtable_root,'qtable'+str(i)+'.txt')
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
        row = [qname,acc1,acc5,r,fitness,part]
        print(row)
        store_csv_check(row, csv_name)

    cmp_dir = os.path.join(optimize_root,'bo_cache')
    create_dir(cmp_dir)
    qtable_root = '/data/zhijing/flickrImageNetV2/bo_cache/qtables5'
   # df = pd.read_csv('csv/bayesian5.csv')
   # scores = np.array((df['rate'], df['acc1']))
   # scores = np.swapaxes(scores, 0, 1)
   # indexes = identify_pareto(scores)
   # np.save('bayesian5.npy', indexes)
    indexs = np.load('bayesian5.npy')
    for i in indexes:
        qname = os.path.join(qtable_root,'qtable'+str(i)+'.txt')
        #print(qname)
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
        row = [qname,acc1,acc5,r,fitness,part]
        print(row)
        store_csv_check(row, csv_name)


    cmp_dir = os.path.join(optimize_root,'bound_cache')
    create_dir(cmp_dir)
    qtable_root = '/data/zhijing/flickrImageNetV2/bound_cache/qtables'
#    df = pd.read_csv('csv/bound.csv')
#    scores = np.array((df['rate'], df['acc1']))
#    scores = np.swapaxes(scores, 0, 1)
#    indexes = identify_pareto(scores)
#    np.save('bound.npy', indexes)
    indexes = np.load('bound.npy')
    for i in indexes:
        qname = os.path.join(qtable_root,'qtable'+str(i)+'.txt')
        #print(qname)
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
        row = [qname,acc1,acc5,r,fitness,part]
        print(row)
        store_csv_check(row, csv_name)

        
if __name__=='__main__':
    sys.exit(run())
