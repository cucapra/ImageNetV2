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
from scipy import signal
def store_csv(row, name):
    with open(name, 'a') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(row)
    csvFile.close()
def store_csv_check(row,name):
    if not os.path.isfile(name):
        store_csv(['i','acc1','acc5','rate','cmp_mean','cmp_std'],name)
    store_csv(row,name)
def csv_to_list(name):
    df = pd.read_csv(name) 
    dics = {}
    for i,r in enumerate(df['rate']):
        dics[round(r)] = (df['acc1'][i],df['acc5'][i] )
    return df['i'][len(df)-1],dics
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
def compare_acc(s,l,diff):#cmp rate: small or equal, large
    if ((s[0] > l[0] and s[1] > l[1] + diff * 0.01) or 
        (s[0] > l[0] + diff * 0.005 and s[1] > l[1])):
        print("ok,pass! let's throw a dice to decide whether to accept it.")
        return True
    return False

def ifsuccess(mean,std,accs,uncmp_mean,store_rate_acc):
    coef = np.array([-238.80368897,69.500704,61.07984424])
    acc = np.array([accs[0]**2,accs[0],1])
    rate = (round(uncmp_mean/(mean+std/50)), round(uncmp_mean/(mean+std/50)))
    print('range of comp rate:',rate)
    if rate[0] == rate[1]: rate = [rate[0]]
    for r in rate:
        fitness = r-np.sum(acc*coef)
        #if r*accs[0] > store_rate_acc[0] or r*accs[1] > store_rate_acc[1]:
        if fitness > 0:
            if random.uniform(0,1) > 0.2:
                store_rate_acc = (r*accs[0],r*accs[1])
                return True
        else:
            if random.uniform(0,1) > 0.8:
                store_rate_acc = (r*accs[0],r*accs[1])
                return True
#        if len(store_rate_acc) == 0:
#            store_rate_acc[r] = accs
#            return True
#        elif r in store_rate_acc.keys():
#            if (compare_acc(accs,store_rate_acc[r],0) and
#                random.uniform(0,1) > 0.5):
#                store_rate_acc[r] = accs
#                return True
#        else:
#            rates = sorted(store_rate_acc.keys())
#            if r > rates[-1]:
#                if (compare_acc(store_rate_acc[rates[-1]],accs,r-rates[-1]) and
#                    random.uniform(0,1) > 0.5):
#                    store_rate_acc[r] = accs
#                    return True
#            elif r < rates[0]:
#                if (compare_acc(accs,store_rate_acc[rates[0]],rates[0]-r) and
#                    random.uniform(0,1) > 0.5):
#                    store_rate_acc[r] = accs
#                    return True
#            else:
#                l = 0#r<l
#                for k in sorted(store_rate_acc.keys(),reverse=True):
#                    if r > k: 
#                        break
#                    else: 
#                        l = k
#                print(r,l)
#                assert(l>r,'should be l > r!')
#                if (compare_acc(accs,store_rate_acc[l],l-r) and
#                    random.uniform(0,1) > 0.5):
#                    store_rate_acc[r] = accs
#                    return True
    return False
def compress(dir_list,file_list,cmp_dir,uncmp_root,tmp_qtable):
    for dir_in in dir_list:
        if not os.path.exists(os.path.join(cmp_dir,dir_in) ):
            os.makedirs(os.path.join(cmp_dir,dir_in) )
        for file_in in file_list[dir_in]:
            file_out = os.path.join(cmp_dir,dir_in,file_in.replace('bmp','jpg'))
            execute = "./cjpeg -outfile "+file_out+" -quality 50 -qtable "+tmp_qtable+" "+os.path.join(uncmp_root,dir_in,file_in)
            os.system(execute)
def create_dir(n):
    if not os.path.exists(n):
        os.makedirs(n)

def mutate(qname,write_to):
    qtable=read_qtable(qname)
    p = 0.6
    mutate_index = np.random.choice(a=[False, True], size=(3, 8, 8), p=[1-p, p])
    #qtable[mutate_index] += np.random.randint(-5,5,np.sum(mutate_index))
    arr = np.asarray(qtable,float)    
    kernel = np.array([[0,1,0],
                   [1,1,1],
                   [0,1,0]])
    for i in range(3):
        arr[i] = signal.convolve2d(arr[i], kernel, boundary='wrap', mode='same')/kernel.sum()
    qtable[mutate_index] = np.asarray(arr[mutate_index],int)
    write_qtable(qtable,write_to)

@click.command()
@click.option('--trial',default='0',type=str)
def run(trial):
    uncmp_root = '/data/zhijing/flickrImageNetV2/matched_frequency_part/'#part,224
    uncmp_mean = 150582
    optimize_root = '/data/zhijing/flickrImageNetV2/markov_cache/trial'+trial+'/'
    create_dir(optimize_root)
    cmp_dir = os.path.join(optimize_root,'dataset')
    create_dir(optimize_root)
    dir_list = os.listdir(uncmp_root)
    file_list = {x:os.listdir(os.path.join(uncmp_root,x)) for x in dir_list }
    
    store_rate_acc = (0,0)#{}
    csv_name = "markov"+trial+".csv"
    starti=-1
    if os.path.isfile(csv_name):
        starti,store_rate_acc=csv_to_list(csv_name)
        print(starti,store_rate_acc)
    #store_rate_acc[22] = (0.5809,0.802)
    #store_csv_check(['-1','0.5809','0.802','22','150582','0'],csv_name)
    qtable_root = os.path.join(optimize_root,'qtables')
    create_dir(qtable_root)
    store_qtable = os.path.join(optimize_root,'markov.txt')
    if not os.path.isfile(store_qtable):
        shutil.copyfile('qtables/qtable.txt',store_qtable)
    #tmp_qtable = 'qtables/markov.txt'
    for i in range(starti+1,500):
        if not os.path.exists(cmp_dir):
            os.makedirs(cmp_dir)
        tmp_qtable = os.path.join(qtable_root,'markov'+str(i)+'.txt')
        #perturbed_qtable_generate(store_qtable,tmp_qtable,7)
        mutate(store_qtable,tmp_qtable)
        if i == 0:
            shutil.copyfile('qtables/qtable.txt',tmp_qtable)

        ts = []
        partition = int(len(dir_list)/20)
        for j,k in enumerate(range(0,len(dir_list),partition)):
            ts.append( threading.Thread(target=compress,args=(dir_list[k:k+partition],file_list,cmp_dir,uncmp_root,tmp_qtable)) ) 
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
        if ifsuccess(cmp_mean,cmp_std,(acc1,acc5),uncmp_mean,store_rate_acc): #or i == 0:
            shutil.copyfile(tmp_qtable,store_qtable)
            store_csv_check(row, csv_name)
        
if __name__=='__main__':
    sys.exit(run(sys.argv[1:]))
