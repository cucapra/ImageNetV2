import os,sys,glob,math
from PIL import Image
import numpy as np
import re
from ast import literal_eval
import argparse
from comp_utils import *
import pandas as pd
import random
def read_qtable(qname,i=-1):
    #------old read from file--------
    #e.g [[1,2],[2,3]]
    #f = open(qname,"r")
    #a=re.sub("\s+",'',f.read())
    #qtable=np.array(literal_eval(f.read()))
    #--------new form-----------
    #same as libjpeg requirements, done by space
    qtable = np.loadtxt(qname,dtype=np.int)
    if i!=-1:
        return qtable.reshape((-1,8,8))[i]
    return qtable.reshape((-1,8,8))

def write_qtable(qtable,qname='qtable.txt'):
    f = open(qname,'w')
    str_qtable = str(np.abs(qtable.astype(int)) ).replace('[ ','').replace('[','')
    str_qtable = ' '.join(str_qtable.split())
    str_qtable = str_qtable.replace(']','\n')
    f.write(' '+str_qtable)
    f.close()
    return qtable
def bound_qtable_generate(qtable_name):
    df = pd.read_csv("csv/sorted.csv")
    scores = np.array((df['rate'],df['acc1']))
    scores = np.swapaxes(scores,0,1)
    indexs = np.load('pareto1000.npy')
    qts = []
    for ind in indexs:
        qname=("/data/zhijing/flickrImageNetV2/sorted_cache/qtables/qtable"+str(ind)+".txt")
        if df['rate'][ind] > 21 and df['rate'][ind] < 23:
            qt = read_qtable(qname,0)
            qts.append(qt.reshape((-1)))
            qts.append(np.transpose(qt).reshape((-1)))
    qtstd = np.array(qts).std(axis=0)
    qtmax = np.rint(np.clip(np.array(qts).max(axis=0)+qtstd*0.5, 1, 255))
    qtmin = np.rint(np.clip(np.array(qts).min(axis=0)-qtstd*0.5, 1, 255))
    bound_qt = np.array([np.random.randint(qtmin[i], qtmax[i]) for i in range(64)]).reshape(8,8)
    return write_qtable(bound_qt, qtable_name)


def perturbed_qtable_generate(input_name, qtable_name, step_range = 3):
    qtable = read_qtable(input_name) 
    qtable += np.random.randint(-1*step_range,step_range+1,size=qtable.shape)
    qtable = np.clip(qtable,1,255)
    write_qtable(qtable,qtable_name)
    

def sorted_qtable_generate(qtable_name,depth=3):
    qtable = np.abs(np.random.standard_normal(( depth,64) ))
    qtable = [qtable[i]/np.max(qtable[i]) for i in range(depth)]
    qtable = np.sort(qtable,axis=1)
    qtable = np.array([zigzag_reverse(qtable[i],8) for i in range(3)])
    ran = np.random.randint(0,35)
    ranmax = np.random.randint(50,150)
    qtable = np.clip(np.round(qtable*(ranmax-ran) + ran ),ran,ranmax).astype(int)
    scale = np.random.uniform(0.5,1.5)
    qtable = np.clip(np.round(qtable*scale),1,255).astype(int)
    if depth == 3:
        write_qtable(qtable,qtable_name)
        return qtable
    else:
        write_qtable(qtable[0],qtable_name)
        return qtable[0]


def random_qtable_generate(qtable_name):
    qtable = np.abs(np.random.standard_normal((3,8,8)))
    qtable = qtable/np.max(qtable) #range in [0, 1]
    qtable = np.clip(np.round(qtable*255),1,255).astype(int)
    write_qtable(qtable,qtable_name) 
    return qtable

def size_jpeg(path):
    comp = glob.glob(path)
    x = []
    for name in comp:
        x.append(os.path.getsize(name))
    return np.array(x).mean()


