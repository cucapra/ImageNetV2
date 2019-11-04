from bayes_opt import BayesianOptimization
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
qtable_dir = os.path.join(optimize_root, 'qtables')
create_dir(qtable_dir)


uncmp_root = '/mnt/tmpfs/matched_frequency_part/'
uncmp_mean = 150582
dir_list = os.listdir(uncmp_root)
file_list = {x:os.listdir(os.path.join(uncmp_root,x)) for x in dir_list }
cnt = 0
def black_box_function(q0,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,q11,q12,q13,q14,q15,q16,q17,q18,q19,q20,q21,q22,q23,q24,q25,q26,q27,q28,q29,q30,q31,q32,q33,q34,q35,q36,q37,q38,q39,q40,q41,q42,q43,q44,q45,q46,q47,q48,q49,q50,q51,q52,q53,q54,q55,q56,q57,q58,q59,q60,q61,q62,q63):
    global cnt, file_list,dir_list,uncmp_root,uncmp_mean,cmp_dir,optimize_root,qtable_dir
    qtable_file = os.path.join(qtable_dir,'qtable'+str(cnt)+'.txt')
    fitness = 0
    ts = []
    partition = int(len(dir_list)/20)#20
    qtable = np.array((q0,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,q11,q12,q13,q14,q15,q16,q17,q18,q19,q20,q21,q22,q23,q24,q25,q26,q27,q28,q29,q30,q31,q32,q33,q34,q35,q36,q37,q38,q39,q40,q41,q42,q43,q44,q45,q46,q47,q48,q49,q50,q51,q52,q53,q54,q55,q56,q57,q58,q59,q60,q61,q62,q63)).reshape((8,8))
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
    store_csv_check(row,"bayesian.csv",['acc1','acc5','rate','qname'])
    cnt += 1
    return fitness


# Bounded region of parameter space
pbounds = {'q'+str(i): (1,255) for i in range(64)}

optimizer = BayesianOptimization(
    f=black_box_function,
    pbounds=pbounds,
    random_state=1,
)

df = pd.read_csv("csv/sorted.csv")
scores = np.array((df['rate'],df['acc1']))
scores = np.swapaxes(scores,0,1)
indexs = [x for x in np.load('pareto.npy')]
for ind in indexs:
    fitness = diff_fit(scores[ind][1], scores[ind][0])
    qname=("/data/zhijing/flickrImageNetV2/sorted_cache/qtables/qtable"+str(ind)+".txt")
    qtable=ratio.read_qtable(qname,0).reshape((-1))
    qtable_dic = {'q'+str(i): qtable[i] for i in range(64)}
    optimizer.register(qtable_dic,fitness)
optimizer.maximize(init_points = 20, n_iter=100)
print(optimizer.max)
 
