import os, sys, threading
from utils import *
import train
#part = "standard30"
#uncmp_root = '/data/zhijing/imagenet300/uncmp/'
csv_name = 'csv/retrain_val_v2.csv'
uncmp_mean = 150582
#cmp_dir = '/data/zhijing/imagenet300/'+part
uncmp_root = '/data/zhijing/cs6787/imagenet_224/train/'
dir_list = os.listdir(uncmp_root)
print(len(dir_list))
file_list = {x:os.listdir(os.path.join(uncmp_root,x)) for x in dir_list}
quality = str(50)
#qname = "/mnt/tmpfs/bo_cache/qtables/qtable0.txt"
optimize_root = '/data/zhijing/cs6787/imagenet_cmp/quality50/'
create_dir(optimize_root)
train_dir = os.path.join(optimize_root, 'train')
create_dir(train_dir)

ts = []
partition = int(len(dir_list)/32)
for j,k in enumerate(range(0,len(dir_list),partition)):
    ts.append( threading.Thread(target=compress_quality, args=(quality, dir_list[k:k+partition], file_list, train_dir, uncmp_root)) )
    ts[j].start()
for t in ts:
    t.join()


uncmp_root = '/data/zhijing/flickrImageNetV2/matched_frequency_train/val/'
dir_list = os.listdir(uncmp_root)
print(len(dir_list))
file_list = {x:os.listdir(os.path.join(uncmp_root,x)) for x in dir_list}
val_dir = os.path.join(optimize_root, 'val')
create_dir(val_dir)

ts = []
partition = int(len(dir_list)/32)
for j,k in enumerate(range(0,len(dir_list),partition)):
    ts.append( threading.Thread(target=compress_quality, args=(quality, dir_list[k:k+partition], file_list, val_dir, uncmp_root)) )
    ts[j].start()
for t in ts:
    t.join()
cmp_mean, cmp_std = get_size(val_dir)
r = uncmp_mean/cmp_mean

sys.argv = ['skip', '-d', train_dir, '--val_name', val_dir, '--feature_extract', '-m', 'resnet', '-ep', '10', '-c', '1000']
acc1 = train.run(sys.argv)
fitness = diff_fit(acc1, r)
row=[quality, acc1, r, fitness]
print(row)
store_csv_check(row,csv_name)
#cmp_dir=uncmp_root

#cmp_mean,cmp_std = get_size(cmp_dir)
#rate = 150583/cmp_mean
#print('rate',rate)
#sys.argv = ['skip','--dataset']
#acc1,acc5 = eval.run(sys.argv.append(cmp_dir))
#fitness=diff_fit(acc1,rate)
#print('fitness', fitness)
#row = [acc1,acc5,rate,fitness,qname,part]
#store_csv_check(row,"csv/eval_standard.csv", ['acc1','acc5','rate','fit','qname','part'])
