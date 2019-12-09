# from bayes_opt import BayesianOptimization
# from bayes_opt.observer import JSONLogger
# from bayes_opt.event import Events
# from bayes_opt.util import load_logs

import os,sys,csv,time,random,torch,threading,argparse
from multiprocessing import Pool
import numpy as np
import torch.nn as nn
from torchvision import datasets, transforms
import evaluate
# from utils import *
import utils,ratio
from train2 import parse_args,initialize_model,load_data,create_optimizer,train_model
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


data_t = transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224)])
def to_bmp(in_root, in_dirs, file_list, out_dir, limit=300):
    pool = Pool(5)
    def xform(args):
        in_root,dir_in,file_in,file_out = args
        try:
            data_t(Image.open(
                os.path.join(in_root,dir_in,file_in)
            )).convert('RGB').save(file_out)
        except OSError as e:
            print(e)
    arg_list = []
    for dir_in in in_dirs:
        temp_path = os.path.join(out_dir,dir_in)
        if not os.path.exists(temp_path) and os.path.isdir(os.path.join(in_root,dir_in)):
            os.makedirs(temp_path)
        if dir_in in file_list:
            count = 0
            for file_in in file_list[dir_in]:
                count += 1
                if count>limit:
                    break
                out_name = file_in.split('.')[0]+'.bmp'
                file_out = os.path.join(temp_path,out_name)
                if not os.path.isfile(file_out):
                    # print(file_in, file_out)
                    # xform(in_root,dir_in,file_in,file_out)
                    arg_list.append((in_root,dir_in,file_in,file_out))
            # print(temp_path, count)
    pool.map(xform, arg_list)



def compress(dir_list,file_list,cmp_dir,uncmp_root,tmp_qtable,limit=10):
    for dir_in in dir_list:
        if dir_in not in file_list:
            continue
        create_dir(os.path.join(cmp_dir,dir_in))
        count = 0
        for file_in in file_list[dir_in]:
            count += 1
            if count>limit:
                break
            file_out = os.path.join(cmp_dir,dir_in,file_in.replace('bmp','jpg'))
            # print(file_in,file_out)
            if not os.path.isfile(file_out) or os.path.getsize(file_out)==0:
                execute = "~/libjpeg/cjpeg -outfile "+file_out+" -quality 50 -qtable "+tmp_qtable+" -qslots 0 "+os.path.join(uncmp_root,dir_in,file_in)
                os.system(execute)


def compress_quality(quality, dir_list, file_list, cmp_dir, uncmp_root,limit=10):
    for dir_in in dir_list:
        if dir_in not in file_list:
            continue
        create_dir(os.path.join(cmp_dir,dir_in))
        count = 0
        for file_in in file_list[dir_in]:
            count += 1
            if count>limit:
                break
            file_out = os.path.join(cmp_dir,dir_in,file_in.replace('bmp','jpg'))
            if not os.path.isfile(file_out) or os.path.getsize(file_out)==0:
                execute = "~/libjpeg/cjpeg -outfile "+file_out+" -quality "+quality+" "+os.path.join(uncmp_root,dir_in,file_in)
                os.system(execute)

def create_dir(n):
    if not os.path.exists(n):
        os.makedirs(n)


def run(args):
    ind,logs = args
    start = time.time()
    qtable_file=os.path.join(qtable_dir, 'qtable'+str(ind)+'.txt')

    ### compress train dataset
    if retrain:
        logs += 'Compressing train set...\n'
        cmp_dir_train = os.path.join(optimize_root, 'train/qtable_'+str(ind))
        create_dir(cmp_dir_train)
        ts = []
        for j,k in enumerate(range(0,len(dir_list_train),partition)):
            ts.append(threading.Thread(target=compress,args=(dir_list_train[k:k+partition],file_list_train,cmp_dir_train,uncmp_root_train,qtable_file,img_per_cls_train)))
            ts[j].start()
        for t in ts:
            t.join()
        cmp_size = get_size(cmp_dir_train)
        cmp_mean,cmp_std = cmp_size.mean(), cmp_size.std()
        rate = uncmp_mean_train/cmp_mean
        logs += '--train cr: {}({}/{})'.format(rate, uncmp_mean_train, cmp_mean)

    ### compress val dataset
    logs += 'Compressing val set...\n'
    cmp_dir = os.path.join(optimize_root, 'val/qtable_'+str(ind))
    create_dir(cmp_dir)
    # compress(dir_list,file_list,cmp_dir,uncmp_root,qtable_file)
    # raise Exception('fsfsfs')

    ts = []
    for j,k in enumerate(range(0,len(dir_list),partition)):
        ts.append(threading.Thread(target=compress,args=(dir_list[k:k+partition],file_list,cmp_dir,uncmp_root,qtable_file,img_per_cls))) 
        ts[j].start()
    for t in ts:
        t.join()
    cmp_size = get_size(cmp_dir)
    cmp_mean,cmp_std = cmp_size.mean(), cmp_size.std()
    rate = uncmp_mean/cmp_mean
    logs += '--val cr: {}({}/{})'.format(rate, uncmp_mean, cmp_mean)

    ### retraining
    if retrain:
        train_args = [
            '--model_name','resnet',
            '--data_dir',cmp_dir_train,
            '--batch_size','64',
            '--num_classes','1000',
            '--val_name',cmp_dir,
            '--num_epochs','10',
        ]
        acc1,acc5 = run_train(train_args)
    else:
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
    print(logs)

def run_standard(args):
    ind,logs = args
    start = time.time()
    
    ### compress train dataset
    if retrain:
        logs += 'Compressing train set...\n'
        cmp_dir_train = os.path.join(optimize_root, 'train/qtable_'+str(ind))
        create_dir(cmp_dir_train)
        ts = []
        for j,k in enumerate(range(0,len(dir_list_train),partition)):
            ts.append(threading.Thread(target=compress_quality,args=(str(ind),dir_list_train[k:k+partition],file_list_train,cmp_dir_train,uncmp_root_train,img_per_cls_train)))
            ts[j].start()
        for t in ts:
            t.join()
        cmp_size = get_size(cmp_dir_train)
        cmp_mean,cmp_std = cmp_size.mean(), cmp_size.std()
        rate = uncmp_mean_train/cmp_mean
        logs += '--train cr: {}({}/{})'.format(rate, uncmp_mean_train, cmp_mean)

    ### compress val dataset
    logs += 'Compressing val set...\n'
    cmp_dir = os.path.join(optimize_root, 'val/qtable_'+str(ind))
    create_dir(cmp_dir)
    ts = []
    for j,k in enumerate(range(0,len(dir_list),partition)):
        ts.append( threading.Thread(target=compress_quality,args=(str(ind),dir_list[k:k+partition],file_list,cmp_dir,uncmp_root,img_per_cls))) 
        ts[j].start()
    for t in ts:
        t.join()
    cmp_size = get_size(cmp_dir)
    cmp_mean,cmp_std = cmp_size.mean(), cmp_size.std()
    rate = uncmp_mean/cmp_mean
    logs += '--val cr: {}({}/{})'.format(rate, uncmp_mean, cmp_mean)

    ### retraining
    if retrain:
        train_args = [
            '--model_name','resnet',
            '--data_dir',cmp_dir_train,
            '--batch_size','64',
            '--num_classes','1000',
            '--val_name',cmp_dir,
            '--num_epochs','10',
        ]
        acc1,acc5 = run_train(train_args)
    else:
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


def identify_pareto(source="sorted.csv", dest='pareto'):
    df = pd.read_csv(source)
    scores=np.array((df['rate'],df['acc1']))
    scores=np.swapaxes(scores,0,1)

    # Count number of items
    population_size = scores.shape[0]
    # Create a NumPy index for scores on the pareto front (zero indexed)
    population_ids = np.arange(population_size)
    # Create a starting list of items on the Pareto front
    # All items start off as being labelled as on the Parteo front
    pareto_front = np.ones(population_size, dtype=bool)
    # Loop through each item. This will then be compared with all other items
    for i in range(population_size):
        # Loop through all other items
        for j in range(population_size):
            # Check if our 'i' pint is dominated by out 'j' point
            if all(scores[j] >= scores[i]) and any(scores[j] > scores[i]):
                # j dominates i. Label 'i' point as not on Pareto front
                pareto_front[i] = 0
                # Stop further comparisons with 'i' (no more comparisons needed)
                break
    # Return ids of scenarios on pareto front
    np.save(dest,population_ids[pareto_front])
    return population_ids[pareto_front]


def run_train(args):
    args = parse_args(args)
    args.device = torch.device("cuda:{}".format(gpu_id) if torch.cuda.is_available() else "cpu")
    model_ft, input_size = initialize_model(args,use_pretrained=True)
    image_datasets, dataloaders_dict = load_data(args, input_size) 
    
    optimizer_ft = create_optimizer(args, model_ft)
    criterion = nn.CrossEntropyLoss()
    # Train and evaluate
    model_ft, hist = train_model( args=args, model=model_ft, dataloaders=dataloaders_dict, criterion=criterion, optimizer=optimizer_ft, is_inception=(args.model_name=="inception") )
    #torch.save(model_ft.state_dict(), os.path.join(args.dir,"verification.final"))
    best_acc = (0,0)
    for acc1,acc5 in hist:
        if acc1>best_acc[0]:
            best_acc = (acc1,acc5)
    return best_acc


# ### convert jpeg images to bmp
# dir_list = os.listdir('/data/ILSVRC2012/train_bmp300')
# file_list = {}
# for x in dir_list:
#     temp_dir = os.path.join('/data/ILSVRC2012/train_bmp300',x)
#     if os.path.isdir(temp_dir):
#         file_list[x] = os.listdir(temp_dir)
# to_bmp('/data/ILSVRC2012/train_bmp300', dir_list, file_list, '/data/ILSVRC2012/train_bmp300_resize224')
# raise Exception('to_bmp')

gpu_id = 3
retrain = True
suffix = 'mab' # standard,sorted,bayesian5,bound,mab
img_per_cls_train = 100
img_per_cls = 10
subproc,procs = 0,1 # 0,1,2,3
uncmp_root = '/data/ILSVRC2012/val_bmp_resize224'
uncmp_root_train = '/data/ILSVRC2012/train_bmp300_resize224'



csv_name = 'csv/crossval_imagenet_{}.csv'.format(suffix+'_retrain' if retrain else suffix)
optimize_root = '/data/zh272/temp/{}/'.format(suffix)
if suffix == 'mab':
    metrics_file = 'csv/mab_bounded.csv'
else:
    metrics_file = "csv/{}.csv".format(suffix)

if suffix == 'sorted':
    qtable_dir = '/data/zh272/flickrImageNetV2/sorted_cache/qtables/'
    pareto_file = 'pareto1000'
elif suffix == 'bayesian5':
    qtable_dir = '/data/zh272/flickrImageNetV2/bo_chace/qtables5/'
    pareto_file = 'pareto_bayesian'
elif suffix == 'bound':
    qtable_dir = '/data/zh272/flickrImageNetV2/bound_cache/qtables/'
    pareto_file = 'pareto_bound'
elif suffix == 'mab':
    qtable_dir = '/data/zh272/flickrImageNetV2/mab_cache/qtables/'
    pareto_file = 'pareto_mab'
    # df = pd.read_csv('csv/mab_bounded_qtable.csv')
    # for index in range(len(df)):
    #     qtable = np.array([df['q'+str(i).zfill(2)][index] for i in range(64)]).reshape((8,8))
    #     ratio.write_qtable(qtable,qname=os.path.join(qtable_dir, 'qtable'+str(index)+'.txt'))
elif suffix == 'standard':
    pass
else:
    raise Exception('not finished')

os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
create_dir(optimize_root)
if suffix not in ['standard','mab']:
    create_dir(qtable_dir)
df = pd.read_csv(metrics_file)
# uncmp_mean,uncmp_std = 687300.03404, 1118486.0851240403
# uncmp_mean_train,uncmp_std_train = 662408.8940933333, 1233251.0965360408
uncmp_mean,uncmp_std = 150582.0, 0.0
uncmp_mean_train,uncmp_std_train = 150582.0, 0.0

# uncmp_size = get_size(uncmp_root)
# uncmp_mean,uncmp_std = uncmp_size.mean(), uncmp_size.std()
# print('val',uncmp_mean,uncmp_std)
# uncmp_size_train = get_size(uncmp_root_train)
# uncmp_mean_train,uncmp_std_train = uncmp_size_train.mean(), uncmp_size_train.std()
# print('train',uncmp_mean_train,uncmp_std_train)
# raise Exception('umcmp_size')

dir_list_train = os.listdir(uncmp_root_train)
file_list_train = {}
for x in dir_list_train:
    temp_dir = os.path.join(uncmp_root_train,x)
    if os.path.isdir(temp_dir):
        file_list_train[x] = os.listdir(temp_dir)

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

if suffix == 'standard':
    ### For standard JPEG table with different quality
    for ind,qua in enumerate(range(5,101,5)):
        if df['rate'][ind]>20 and df['rate'][ind]<30:
            logs = 'Cross-validating quality: {} (CR:{},acc1:{})\n'.format(qua, df['rate'][ind], df['acc1'][ind])
            run_standard((qua,logs))

else:
    ### For customized JPEG tables
    if os.path.exists(pareto_file+'.npy'):
        indexs = np.load(pareto_file+'.npy')
    else:
        indexs = identify_pareto(source=metrics_file, dest=pareto_file)
    # pool = Pool(4)
    # indexs = np.random.choice(indexs,size=5,replace=False)

    st_rate = 22.107518218126575
    rates = np.abs(np.array(df['rate'][indexs]) - st_rate)
    ri = np.array([(rates[i], indexs[i]) for i in range(len(rates))], dtype=[('x', float), ('y', int)])
    ri.sort(order='x')
    # length = len(indexs)//procs
    # for ind in indexs[subproc*length:min((subproc+1)*length, len(indexs))]:
    for rate, ind in ri[:2]:
        # if df['rate'][ind] > 20 and df['rate'][ind] < 30 and ind not in hist:
        if ind not in hist:
            logs = 'Cross-validating q-table: {} (CR:{},acc1:{})\n'.format(ind, df['rate'][ind], df['acc1'][ind])
            run((ind, logs))
    #         arg_list.append((ind, logs))
    # pool.map(run, arg_list)