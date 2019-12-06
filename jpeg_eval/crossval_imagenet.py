# from bayes_opt import BayesianOptimization
# from bayes_opt.observer import JSONLogger
# from bayes_opt.event import Events
# from bayes_opt.util import load_logs

import os,sys,csv,time,random,torch,threading
from multiprocessing import Pool
import numpy as np
from torchvision import datasets, transforms
import evaluate
# from utils import *
import utils
import pandas as pd
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True



def get_size(start_path):
    total_size = []
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                total_size.append(os.path.getsize(fp))
    total_size = np.array(total_size)
    
    # return total_size.mean(),total_size.std()
    return total_size

def to_bmp(in_root, in_dirs, file_list, out_dir):
    for dir_in in in_dirs:
        temp_path = os.path.join(out_dir,dir_in)
        if not os.path.exists(temp_path) and os.path.isdir(os.path.join(in_root,dir_in)):
            os.makedirs(temp_path)
        if dir_in in file_list:
            for file_in in file_list[dir_in]:
                out_name = file_in.split('.')[0]+'.bmp'
                file_out = os.path.join(temp_path,out_name)
                if not os.path.isfile(file_out):
                    print(file_in, file_out)
                    try:
                        Image.open(os.path.join(in_root,dir_in,file_in)).convert('RGB').save(file_out)
                    except OSError as e:
                        print(e)
                        continue


def compress(dir_list,file_list,cmp_dir,uncmp_root,tmp_qtable):
    for dir_in in dir_list:
        if dir_in in file_list:
            if not os.path.exists(os.path.join(cmp_dir,dir_in) ):
                os.makedirs(os.path.join(cmp_dir,dir_in) )
            count = 0
            for file_in in file_list[dir_in]:
                count += 1
                if count>10:
                    break
                file_out = os.path.join(cmp_dir,dir_in,file_in.replace('bmp','jpg'))
                # print(file_in,file_out)
                if not os.path.isfile(file_out) or os.path.getsize(file_out)==0:
                    execute = "~/libjpeg/cjpeg -outfile "+file_out+" -quality 50 -qtable "+tmp_qtable+" -qslots 0 "+os.path.join(uncmp_root,dir_in,file_in)
                    os.system(execute)


def compress_quality(quality, dir_list, file_list, cmp_dir, uncmp_root):
    for dir_in in dir_list:
        if not os.path.exists(os.path.join(cmp_dir,dir_in) ):
            os.makedirs(os.path.join(cmp_dir,dir_in) )
        for file_in in file_list[dir_in]:
            file_out = os.path.join(cmp_dir,dir_in,file_in.replace('bmp','jpg'))
            execute = "~/libjpeg/cjpeg -outfile "+file_out+" -quality "+quality+" "+os.path.join(uncmp_root,dir_in,file_in)
            os.system(execute)

def create_dir(n):
    if not os.path.exists(n):
        os.makedirs(n)

def run(args):
    ind,logs = args
    start = time.time()
    qtable_file=os.path.join(qtable_dir, 'qtable'+str(ind)+'.txt')
    
    cmp_dir = os.path.join(optimize_root, 'qtable_'+str(ind))
    create_dir(cmp_dir)

    ### compress dataset
    logs += 'Compressing...\n'
    # compress(dir_list,file_list,cmp_dir,uncmp_root,qtable_file)
    # raise Exception('fsfsfs')

    ts = []
    for j,k in enumerate(range(0,len(dir_list),partition)):
        ts.append(threading.Thread(target=compress,args=(dir_list[k:k+partition],file_list,cmp_dir,uncmp_root,qtable_file)) ) 
        ts[j].start()
    for t in ts:
        t.join()
    cmp_size = get_size(cmp_dir)
    cmp_mean,cmp_std = cmp_size.mean(), cmp_size.std()
    rate = uncmp_mean/cmp_mean
    print('--cr: {}({}/{})'.format(rate, uncmp_mean, cmp_mean))

    ### evaluate model
    logs += 'Evaluating...\n'
    sys.argv = ['skip','--dataset']
    acc1,acc5 = evaluate.run(sys.argv.append(cmp_dir))

    fitness = utils.diff_fit(acc1,rate)
    logs += '--acc1:{}, acc5:{}, fitness:{}\n'.format(acc1,acc5,fitness)
    row = [ind, acc1,acc5,rate,qtable_file]
    utils.store_csv_check(row,csv_name,['ind','acc1','acc5','rate','qname'])
    _elapse = time.time() - start
    logs += 'Time: {:.1f}h\n'.format(_elapse/3600)
    # break
    print(logs)

def run_standard(args):
    ind,logs = args
    start = time.time()
    
    cmp_dir = os.path.join(optimize_root, 'standard_qtable_'+str(ind))
    create_dir(cmp_dir)

    ### compress dataset
    logs += 'Compressing...\n'

    ts = []
    for j,k in enumerate(range(0,len(dir_list),partition)):
        ts.append( threading.Thread(target=compress_quality,args=(str(ind),dir_list[k:k+partition],file_list,cmp_dir,uncmp_root))) 
        ts[j].start()
    for t in ts:
        t.join()
    cmp_size = get_size(cmp_dir)
    cmp_mean,cmp_std = cmp_size.mean(), cmp_size.std()
    rate = uncmp_mean/cmp_mean
    print('--cr: {}({}/{})'.format(rate, uncmp_mean, cmp_mean))

    ### evaluate model
    logs += 'Evaluating...\n'
    sys.argv = ['skip','--dataset']
    acc1,acc5 = evaluate.run(sys.argv.append(cmp_dir))

    fitness = utils.diff_fit(acc1,rate)
    logs += '--acc1:{}, acc5:{}, fitness:{}\n'.format(acc1,acc5,fitness)
    row = [ind, acc1,acc5,rate,ind]
    utils.store_csv_check(row,csv_name,['ind','acc1','acc5','rate','quality'])
    _elapse = time.time() - start
    logs += 'Time: {:.1f}h\n'.format(_elapse/3600)
    # break
    print(logs)
    

### convert jpeg images to bmp
# dir_list = os.listdir('/data/ILSVRC2012/val')
# file_list = {}
# for x in dir_list:
#     temp_dir = os.path.join('/data/ILSVRC2012/val',x)
#     if os.path.isdir(temp_dir):
#         file_list[x] = os.listdir(temp_dir)
# to_bmp('/data/ILSVRC2012/val', dir_list, file_list, '/data/ILSVRC2012/val_bmp')
# raise Exception('to_bmp')

gpu_id = 0
csv_name = 'csv/crossval_imagenet.csv'
# csv_name = 'csv/crossval_imagenet_standard.csv'
optimize_root = '/data/zh272/temp/'
qtable_dir = '/data/zh272/flickrImageNetV2/sorted_cache/qtables/'
# qtable_dir = '/data/zh272/flickrImageNetV2/bo_chace/qtables5/'
metrics_file = "csv/sorted.csv"
# metrics_file = "csv/bayesian5.csv"
# metrics_file = "csv/standard.csv"
uncmp_root = '/data/ILSVRC2012/val_bmp'


os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
create_dir(optimize_root)
create_dir(qtable_dir)
df = pd.read_csv(metrics_file)
uncmp_mean,uncmp_std = 687300.03404, 1118486.0851240403
# uncmp_size = get_size(uncmp_root)
# uncmp_mean,uncmp_std = uncmp_size.mean(), uncmp_size.std()
# print(uncmp_mean,uncmp_std)
# raise Exception('ddd')

dir_list = os.listdir(uncmp_root)
file_list = {}
for x in dir_list:
    temp_dir = os.path.join(uncmp_root,x)
    if os.path.isdir(temp_dir):
        file_list[x] = os.listdir(temp_dir)

hist = set()
if os.path.isfile(csv_name):
    with open(csv_name) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if 'ind' in row:
                continue
            hist.add(int(row[0]))


partition = int(len(dir_list)/20)#20
arg_list = []

### For standard JPEG table with different quality
# for ind in range(5,101,5):
#     logs = 'Cross-validating quality: {} (CR:{},acc1:{})\n'.format(ind, df['rate'][ind], df['acc1'][ind])
#     run_standard((ind,logs))


### For customized JPEG tables
# scores = np.array((df['rate'],df['acc1']))
# scores = np.swapaxes(scores,0,1)
# indexs = np.load('pareto.npy')
indexs = np.load('pareto1000.npy')
# pool = Pool(4)
for ind in indexs:
    if df['rate'][ind] > 20 and df['rate'][ind] < 30 and ind not in hist:
    # if df['rate'][ind] <= 20 and ind not in hist:
    # if df['rate'][ind] >= 30 and ind not in hist:
    # if df['rate'][ind] <= 25 and ind not in hist:
    # if df['rate'][ind] > 25 and ind not in hist:
        logs = 'Cross-validating q-table: {} (CR:{},acc1:{})\n'.format(ind, df['rate'][ind], df['acc1'][ind])
        run((ind, logs))
        # arg_list.append((ind, logs))
# pool.map(run, arg_list)