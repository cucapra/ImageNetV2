from scipy.stats import t
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import glob
import copy
import os
import csv
os.chdir('../jpeg_eval/')

def store_csv(row, name):
    with open(name,'a') as csvf:
        writer = csv.writer(csvf)
        writer.writerow(row)
    csvf.close()
def store_csv_check(row,name):
    if not os.path.isfile(name):
        store_csv(['method','dataset','t','p','mean'],name)
    store_csv(row,name)
def identify_pareto(scores):
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
    return population_ids[pareto_front]

def get_data(group, **param):
    index = param['index']
    if group == 'ga_selection':
        ga_list = glob.glob('csv/ga_selection_*_*.csv')
        rate=[]
        acc1=[]
        for name in ga_list:
            df = pd.read_csv(name)
            rate.append(np.array(df['rate'])[-1])
            acc1.append(np.array(df['acc1'])[-1])
        return (rate[index],acc1[index])
#df  = pd.read_csv("ratio.csv")
#pl = df.plot(kind='scatter',x='Comp Rate',y='Acc', s=30, color ='Blue', label='random jpeg')
#term = 'rate'#rate
    if 'stats' in group:
        df = pd.read_csv("csv/stats2.csv")
        term1 = df['i'].str.contains(param['contains1'])
        term2 = df['i'].str.contains(param['contains2'])
        res = (df['rate'][term1][term2][index], df['acc1'][term1][term2][index])
        return res
    if 'MAB' in group:
        df = pd.read_csv("csv/mab_bounded.csv")
        return (df['rate'][index], df['acc1'][index])
    if 'sorted' in group:
        df = pd.read_csv("csv/sorted.csv")
        return (df['rate'][index], df['acc1'][index])
    if 'bound' in group:
        df = pd.read_csv('csv/'+group+'.csv')
        return (df['rate'][index], df['acc1'][index])
    if 'bayesian' in group:
        df = pd.read_csv('csv/'+group+'.csv')
        return (df['rate'][index], df['acc1'][index])
    if 'standard' in group:
        df = pd.read_csv('csv/'+group+'.csv')
        term = df['i'].astype(str).str.isnumeric()
        if param['start'] != None:
            term = df['i'].str.startswith(param['startwith']) 
        res = (df['rate'][term][index], df['acc1'][term][index])
        return res
    if 'psnr' in group:
        df = pd.read_csv('csv/sorted_psnr.csv')
        return (df['rate'][index], df['psnr_mean'][index])

catos = ['matched', 'threshold0.7', 'topimages', 'imgnet_val']
methods = ['sorted', 'bound', 'bys', 'mab']

for method in methods:
    for cato in catos:
        groups = {  
                    'stats_sort':{'index':slice(None), 'contains1':method, 'contains2': cato},
                    'stats_standard':{'index':slice(None), 'contains1': 'quality', 'contains2':cato},
        
                 }
        # Create plot
        fig = plt.figure()
        ax = fig.add_subplot(111)#axisbg="1.0")
        # np.polyfit(g2[0],g2[1], 2)
        #x = np.linspace(min(df['rate']), max(df['rate']), 1000)
        #y = [ np.sum(np.array([a**2,a,1])*coef) for a in x]
        #ax.plot(x,y)
        xmax = 0
        xmin = 0
        ys = []
        for k in groups.keys():
            x, y = get_data(k, **groups[k])
            y = np.array(y)
            print(len(y))
            y = y[y>0.5]
            y = np.sort(y)
            ys.append(y)
            #mean, var, skew = t.fit(y, df, np.mean(y), np.std(y))
            lnspc=np.linspace(min(y), max(y), 100)
            ax.plot(lnspc,stats.norm.pdf(lnspc, loc=np.mean(y), scale=np.std(y)), label = k)
            ax.hist(y,  bins=10, density=True, alpha = 0.7, label=k)   
            #use this to draw histogram of your data
        
            #a = 1 if group=='standard' or 'pareto' else 0.3
            #ax.hist(x, y, s=30, label=k)
        t,p = stats.ttest_ind(ys[0],ys[1], equal_var=False)
        print(method,cato)
        print(t,p)
        store_csv_check([method,cato,str(t),str(p),np.mean(ys[0])-np.mean(ys[1])],'csv/ttest.csv')
        t,p = stats.ttest_ind(ys[0],ys[1])
        print(t,p)
        
        #plt.title('CR pareto vs Acc')
        #plt.legend(loc=0)
        #plt.xlabel('Accuracy') 
        #plt.ylabel('#')
        #os.chdir('../plots/')
        #plt.savefig(os.path.basename(__file__).replace('.py','.png'))
        #plt.show()


