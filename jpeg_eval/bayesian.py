from bayes_opt import BayesianOptimization
from bayes_opt.observer import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs

import random
import ratio
import numpy as np
import threading
import eval
from utils import *
import pandas as pd
import sys

def diff_fit(acc1, rate):
    coef = np.array([-6.40782216e-05,-1.81228974e-03,6.46255250e-01])
    rates = np.array([rate**2,rate,1])
    return acc1 - np.sum(rates*coef)

optimize_root = '/mnt/tmpfs/bo_cache/'
create_dir(optimize_root)
cmp_dir = os.path.join(optimize_root, 'dataset')
create_dir(cmp_dir) 
qtable_dir = os.path.join(optimize_root, 'qtables6')
create_dir(qtable_dir)

csv_name = 'csv/bayesian6.csv'
uncmp_root = '/mnt/tmpfs/matched_frequency_part/'
uncmp_mean = 150582
dir_list = os.listdir(uncmp_root)
file_list = {x:os.listdir(os.path.join(uncmp_root,x)) for x in dir_list }
cnt = 0
if os.path.exists(csv_name): 
    cnt = sum(1 for line in open(csv_name)) - 1
def black_box_function(**p):
#q0,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,q11,q12,q13,q14,q15,q16,q17,q18,q19,q20,q21,q22,q23,q24,q25,q26,q27,q28,q29,q30,q31,q32,q33,q34,q35,q36,q37,q38,q39,q40,q41,q42,q43,q44,q45,q46,q47,q48,q49,q50,q51,q52,q53,q54,q55,q56,q57,q58,q59,q60,q61,q62,q63):
    global cnt, file_list,dir_list,uncmp_root,uncmp_mean,cmp_dir,optimize_root,qtable_dir
    print("generating qtable ", str(cnt))
    qtable_file = os.path.join(qtable_dir,'qtable'+str(cnt)+'.txt')
    fitness = 0
    ts = []
    partition = int(len(dir_list)/20)#20
    qtable = [p['q'+str(i).zfill(2)] for i in range(64)]
    qtable = np.array(qtable).reshape((8,8))
    ratio.write_qtable(qtable,qtable_file)
    for j,k in enumerate(range(0,len(dir_list),partition)):
        ts.append( threading.Thread(target=compress,args=(dir_list[k:k+partition],file_list,cmp_dir,uncmp_root,qtable_file)) ) 
        ts[j].start()
    for t in ts:
        t.join()
    cmp_mean,cmp_std = get_size(cmp_dir)
    rate = 150582/cmp_mean

    sys.argv = ['skip','--dataset']
    acc1,acc5 = eval.run(sys.argv.append(cmp_dir) )

    fitness = diff_fit(acc1,rate) 
    print('fitness',fitness)
    row = [acc1,acc5,rate,qtable_file]
    store_csv_check(row,csv_name,['acc1','acc5','rate','qname'])
    cnt += 1
    return fitness

#standard = ratio.read_qtable('qtables/qtable.txt')[0]
# Bounded region of parameter space
df = pd.read_csv("csv/sorted.csv")
scores = np.array((df['rate'],df['acc1']))
scores = np.swapaxes(scores,0,1)
indexs = np.load('pareto.npy')
print(indexs.shape)
qts = []
for ind in indexs:
    qname=("/data/zhijing/flickrImageNetV2/sorted_cache/qtables/qtable"+str(ind)+".txt")
    if df['rate'][ind] > 21 and df['rate'][ind] < 23:
        print(ind)
        qt=ratio.read_qtable(qname,0)
        qts.append(qt.reshape((-1)))
        qts.append(np.transpose(qt).reshape((-1)))
qtmean =np.rint(np.clip(np.array(qts).mean(axis=0), 1, 255))
qtstd = np.array(qts).std(axis=0)
#print(qtmean.reshape([8,8]))
#print((np.array(qts).std(axis=0)/qtmean).reshape([8,8]))
qtmax = np.rint(np.clip(np.array(qts).max(axis=0)+qtstd*0.5, 1, 255))
qtmin = np.rint(np.clip(np.array(qts).min(axis=0)-qtstd*0.5, 1, 255))
#print(qtmax.reshape([8,8]))
#print(qtmin.reshape([8,8]))
#pbounds = {'q'+str(8*i+j).zfill(2): (max(0,standard[i][j]-10),min(255,standard[i][j]+10)) for i in range(8) for j in range(8)}

pbounds = {'q'+str(i).zfill(2): (qtmin[i], qtmax[i]) for i in range(64)}
print(qtmin.reshape(8,8))
print(pbounds)

btypes = {'q'+str(i).zfill(2): int for i in range(64)}

optimizer = BayesianOptimization(
    f=black_box_function,
    pbounds=pbounds,
    ptypes=btypes,
    random_state=1,
)


load_logs(optimizer, logs=["logs/logs3.json"]);
#logger = JSONLogger(path="logs/logs3.json") #it clears logs.json first
#optimizer.subscribe(Events.OPTMIZATION_STEP, logger)
#for ind in indexs:
#    fitness = diff_fit(scores[ind][1], scores[ind][0])
#    qname=("/data/zhijing/flickrImageNetV2/sorted_cache/qtables/qtable"+str(ind)+".txt")
#    #qname="/mnt/tmpfs/bo_cache/qtables/qtable"+str(ind)+".txt"
#    qtable=ratio.read_qtable(qname,0).reshape((-1))
#    qtable_dic = {'q'+str(i).zfill(2): qtable[i] for i in range(64)}
#    optimizer.register(qtable_dic,fitness)

optimizer.maximize(init_points=0, n_iter=500, acq="ei", xi=1e-2)
print(optimizer.max)
 
