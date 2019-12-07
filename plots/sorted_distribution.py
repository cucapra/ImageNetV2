import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import glob
import copy
import os
os.chdir('../jpeg_eval/')

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

rates = np.array(pd.read_csv('csv/sorted.csv')['rate'])
sorted_index = np.logical_and(rates > 22,rates < 22.2)
sorted_index2 = copy.deepcopy(sorted_index)
sorted_index2[1000:] = False
#scores = np.array((df['rate'],df['acc1']))
#scores = np.swapaxes(scores,0,1)
#pareto = identify_pareto(scores)
groups = {  
            'sorted': { 'index': sorted_index },
            'sorted1000': { 'index': sorted_index2 },
            'standard_part1':{'index':slice(9,10), 'start':None}
            #'bayesian3': { 'index': slice(None) },
            #'bayesian5': { 'index': slice(None) },
            #'bayesian6': { 'index': slice(None) },
            #'bound': { 'index': slice(None) },
            #'MAB': { 'index': slice(None) },

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
for k in groups.keys():
    _, y = get_data(k, **groups[k])
    y2 = y.groupby(pd.cut(y, np.arange(y.min(), y.max(), (y.max()-y.min())/7))).count()
    y2[0]+=1
    ax.hist(y,bins = 10,edgecolor='k',label = 'Sorted Random Search')
    break
    #ax.scatter(x, y, s=30, label=k.replace('_part1', ''))
k = 'standard_part1'
_, y = get_data(k, **groups[k])
y = y.get_value(9)
ax.axvline(y, linestyle='dashed',color='k', linewidth=1, label = 'Standard')

#plt.title('CR pareto vs Acc')
plt.legend(loc=0, fontsize=12)
plt.tick_params(axis="x", labelsize=12)
plt.tick_params(axis="y",labelsize=12)
plt.xlabel('Accuracy',fontsize=18) 
plt.ylabel('Count', fontsize=18)
plt.tight_layout()
os.chdir('../plots/')
plt.savefig(os.path.basename(__file__).replace('.py','.png'))
plt.show()

#df = pd.read_csv("libjpeg_random.csv")
#pl = df.plot.scatter(x='Comp Rate', y='Acc', s=30, color='Blue', label='random jpeg',ax=pl);
#df = pd.read_csv("libjpeg_perturbed.csv")
#pl = df.plot.scatter(x='Comp Rate', y='Acc', s=30, color='Yellow', label='perturbed jpeg',ax=pl);


#print(df[1:])

#ax.scatter(x=df['rate'][1:], y=df['acc1'][1:], s=45, color='Green', label='standard jpeg')


#fig = pl.get_figure()
#fig.save
