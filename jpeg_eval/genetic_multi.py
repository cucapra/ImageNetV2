from easyga_multi import GeneticAlgorithm
import random
import ratio 
import numpy as np
import threading
import eval
from utils import *
from operator import attrgetter
import pandas as pd
import sys

trial = 'ga_selection_multi'
optimize_root = '/data/zhijing/flickrImageNetV2/ga_cache/'+trial
create_dir(optimize_root)
cmp_dir = os.path.join(optimize_root, 'dataset')
create_dir(cmp_dir) 
qtable_dir = os.path.join(optimize_root, 'qtables')
create_dir(qtable_dir)

#data = [x for x in np.load('pareto.npy') if x > 500] #945
df = pd.read_csv("sorted.csv")
scores = np.array((df['rate'],df['acc1']))
scores = np.swapaxes(scores,0,1)
#scores[scores[:,0]>=13] = 0

rank = 0
alldata=[]
print( np.count_nonzero(scores)/2 )
while len(alldata)<30 and np.count_nonzero(scores)!=0:
    indexs = identify_pareto(scores)
    data = np.hstack((indexs.reshape(-1,1), np.full((len(indexs), 1), rank) ))
    alldata = data if rank==0 else np.vstack((alldata,data))
    scores[indexs] = 0
    rank=rank+1 if rank <=5 else 5
print(len(alldata),alldata)
sys.exit()
population_size = len(alldata)#15
ga = GeneticAlgorithm(alldata,population_size=population_size,generations=30)#50
def qtable_ij(ij):
    qname = os.path.join(qtable_dir,'ga'+str(ij[0])+'_'+str(ij[1])+'.txt')
    return qname

def create_individual(data,j):    
    #ratio.sorted_qtable_generate(qname,1)
    cpf = os.path.join('/data/zhijing/flickrImageNetV2/sorted_cache/qtables/',
            'qtable'+str(data[j][0])+'.txt')
    cpt = qtable_ij((0,j))
    qtable=ratio.read_qtable(cpf)
    ratio.write_qtable(qtable[0],cpt)
    return (0,j),data[j][1]
    #return [random.randint(0, 1) for _ in range(len(data))]

ga.create_individual = create_individual


def crossover(p1, p2,i,j):
    parent = [ratio.read_qtable(qtable_ij(p1))]
    parent.append( ratio.read_qtable(qtable_ij(p2)) )
    child = [np.zeros(parent[0].shape)]*2
    #row = col = False
    #if random.randint(0,1) == 0:
    #    row = True
    #else:
    #    col = True
    #index = random.randint(2,5)
    #if row:
    #    child_1 = np.concatenate( (parent_1[:,:index], parent_2[:,index:]), axis=1)
    #    child_2 = np.concatenate( (parent_2[:,:index], parent_1[:,index:]), axis=1)
    #else:
    #    child_1 = np.concatenate( (parent_1[:index,:], parent_2[index:,:]),axis=0 )
    #    child_2 = np.concatenate( (parent_2[:index,:], parent_1[index:,:]),axis=0 )
    row = random.randint(2,5)
    col = random.randint(2,5)
    for p in range(2):
        q = random.getrandbits(1) 
        child[p][:row,:col] = parent[q][:row,:col]
        q = random.getrandbits(1) 
        child[p][:row,col:] = parent[q][:row,col:]
        q = random.getrandbits(1) 
        child[p][row:,:col] = parent[q][row:,:col]
        q = random.getrandbits(1) 
        child[p][row:,col:] = parent[q][row:,col:]
    #for p in range(2):
    #    q = random.uniform(0,1)
    #    child[p] = (parent[0]*(1-q)+parent[1]*q)
    qname = qtable_ij((i,j))
    ratio.write_qtable(child[0],qname)
    qname = qtable_ij((i,j+1))
    ratio.write_qtable(child[1],qname)
    return (i,j),(i,j+1)


ga.crossover_function = crossover

def mutate_once(qtable,place = 0): #0 for left top, 1 for right top, 2 for left down
        if place==0:
            c = r = (0,4)
        if place==1:
            c = (4,6)
            r = (0,4)
        if place==2:
            c = (0,4)
            r = (4,6)
        col = np.random.randint(c[0],c[1])
        row = np.random.randint(r[0],r[1])
        qtable[col][row] += np.random.randint(-5,6)
        return qtable

def mutate(individual,rank,write_to):
    qname=qtable_ij(individual)
    qtable=ratio.read_qtable(qname)
    if rank == -1: return qname
    if rank == 0: place_list = [0]
    if rank == 1: place_list = [0,0] 
    if rank == 2: place_list = [0,0,1]
    if rank == 3: place_list = [0,0,1,2]
    if rank == 4: place_list = [0,0,1,1,2]
    if rank == 5: place_list = [0,0,1,1,2,2]
    for place in place_list:
        qtable = mutate_once(qtable, place=place)
        scale = np.random.uniform(0.8,1.2)
        qtable = np.clip(np.round(qtable*scale),1,255).astype(int)
    qname = qtable_ij(write_to)
    ratio.write_qtable(qtable,qname)
    return write_to
ga.mutate_function = mutate



uncmp_root = '/data/zhijing/flickrImageNetV2/matched_frequency_part/'
uncmp_mean = 150582
dir_list = os.listdir(uncmp_root)
file_list = {x:os.listdir(os.path.join(uncmp_root,x)) for x in dir_list }
def fitness(individual, data):
    fitness = 0

    ts = []
    partition = int(len(dir_list)/20)#20
    qname=qtable_ij(individual)
    for j,k in enumerate(range(0,len(dir_list),partition)):
        #compress(dir_list[k:k+partition],file_list,cmp_dir,uncmp_root,qname)
        #break
        ts.append( threading.Thread(target=compress,args=(dir_list[k:k+partition],file_list,cmp_dir,uncmp_root,qname)) ) 
        ts[j].start()
    for t in ts:
        t.join()
    cmp_mean,cmp_std = get_size(cmp_dir)
    rate = 150582/cmp_mean

    sys.argv = ['skip','--dataset']
    acc1,acc5 = eval.run(sys.argv.append(cmp_dir) )
    #coef = np.array([-238.80368897,69.500704,61.07984424])
    #acc = np.array([acc1**2,acc1,1])
    #fitness = rate-np.sum(acc*coef)
    coef = np.array([-6.40782216e-05,-1.81228974e-03,6.46255250e-01])
    rates = np.array([rate**2,rate,1])
    fitness = acc1 - np.sum(rates*coef)
    print('fitness',fitness)
    row = [acc1,acc5,rate,qname]
    store_csv_check(row,trial+".csv",['acc1','acc5','rate','qname'])
    info = (acc1,rate)
    return fitness,info

ga.fitness_function = fitness
ga.run()
print(ga.best_individual())

for individual in ga.last_generation():
    print(individual[:2])


