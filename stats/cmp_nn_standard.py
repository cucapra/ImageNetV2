import os, sys, threading
os.chdir('../jpeg_eval/')
sys.path.insert(1,'../jpeg_eval/')
from utils import *
import train
#part = "standard30"
#uncmp_root = '/data/zhijing/imagenet300/uncmp/'
csv_name = 'csv/cr.csv'
uncmp_mean = 150582
#cmp_dir = '/data/zhijing/imagenet300/'+part

uncmp_root ='/data/zhijing/flickrImageNetV2/threshold0.7_224/'
uncmp_root = '/data/zhijing/cs6787/imagenet_224/val_all/'


dir_list = os.listdir(uncmp_root)
print(len(dir_list))
file_list = {x:os.listdir(os.path.join(uncmp_root,x)) for x in dir_list}
quality = str(50)
#qname = "/mnt/tmpfs/bo_cache/qtables/qtable0.txt"
optimize_root = '/data/zhijing/flickrImageNetV2/cmp/imgnet_val/'
create_dir(optimize_root)
cmp_dir = os.path.join(optimize_root, 'quality50')
create_dir(cmp_dir)

ts = []
partition = int(len(dir_list)/32)
for j,k in enumerate(range(0,len(dir_list),partition)):
    ts.append( threading.Thread(target=compress_quality, args=(quality, dir_list[k:k+partition], file_list, cmp_dir, uncmp_root)) )
    ts[j].start()
for t in ts:
    t.join()


