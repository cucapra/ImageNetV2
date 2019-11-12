import os, sys, threading
from utils import *
import eval
part = "val"
uncmp_root = '/data/zhijing/flickrImageNetV2/matched_frequency_train/'+part
uncmp_mean = 150582
cmp_dir = '/data/zhijing/flickrImageNetV2/matched_frequency_train/quality25/'+part

dir_list = os.listdir(uncmp_root)
print(len(dir_list))
file_list = {x:os.listdir(os.path.join(uncmp_root,x)) for x in dir_list}
qname = "/mnt/tmpfs/bo_cache/qtables/qtable0.txt"
ts = []
partition = int(len(dir_list)/40)
for j,k in enumerate(range(0,len(dir_list),partition)):
    ts.append( threading.Thread(target=compress,args=(dir_list[k:k+partition],file_list,cmp_dir,uncmp_root,qname)) ) 
    ts[j].start()
for t in ts:
    t.join()

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
