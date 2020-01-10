import os, sys, threading
os.chdir('../jpeg_eval/')
sys.path.insert(1,'../jpeg_eval/')
from utils import *
import numpy as np
import pandas as pd
import train

method = 'bound'
cache_name = 'bound_cache'
#part = "standard30"
#uncmp_root = '/data/zhijing/imagenet300/uncmp/'
uncmp_mean = 150582
#cmp_dir = '/data/zhijing/imagenet300/'+part

optimize_roots = [
    '/data/zhijing/flickrImageNetV2/cmp/matched',
    '/data/zhijing/flickrImageNetV2/cmp/threshold0.7',
    '/data/zhijing/flickrImageNetV2/cmp/topimages',
    '/data/zhijing/flickrImageNetV2/cmp/imgnet_val'
]
uncmp_roots = [
    '/data/zhijing/flickrImageNetV2/matched_frequency_224/',
    '/data/zhijing/flickrImageNetV2/threshold0.7_224/',
    '/data/zhijing/flickrImageNetV2/topimages_224/',
    '/data/zhijing/cs6787/imagenet_224/val_all/' 
]


indexes = np.load(method+'.npy')
st_rate = 22.107518218126575
df = pd.read_csv('csv/'+method+'.csv')
rates = np.array(df['rate'][indexes]) - st_rate
ri = np.array([(rates[i], indexes[i]) for i in range(len(rates))], dtype=[('x', float), ('y', int)])
ri.sort(order='x')
ri = [r for r in ri if r[0]>0 ]
rate, index = ri[0]
print(rate, index)



qname = "/data/zhijing/flickrImageNetV2/"+cache_name+"/qtables/qtable"+str(index)+".txt"
print(os.path.exists(qname)) 

for (optimize_root, uncmp_root) in zip(optimize_roots, uncmp_roots):
    create_dir(optimize_root)
    print(optimize_root)
    dir_list = os.listdir(uncmp_root)
    print(len(dir_list))
    file_list = {x:os.listdir(os.path.join(uncmp_root,x)) for x in dir_list}
    cmp_dir = os.path.join(optimize_root,method+str(index))   
    create_dir(cmp_dir)
    print(cmp_dir)
    ts = []
    partition = int(len(dir_list)/32)
    
    for j,k in enumerate(range(0,len(dir_list),partition)):
        ts.append( threading.Thread(target=compress,args=(dir_list[k:k+partition],file_list,cmp_dir,uncmp_root,qname)) ) 
        ts[j].start()
    for t in ts:
        t.join()
    print('done')
