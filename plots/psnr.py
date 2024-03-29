import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import glob
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
    if 'sorted' in group:
        df = pd.read_csv("csv/sorted.csv")
        return (df['rate'][index], df['acc1'][index])
    if 'standard' in group:
        df = pd.read_csv('csv/'+group+'.csv')
        term = df['i'].str.isnumeric()
        if start != None:
            term = df['i'].str.startswith(param['startwith']) 
        return (df['rate'][term][index], df['acc1'][term][index])
    if 'psnr' in group:
        df = pd.read_csv('csv/sorted_psnr.csv')
        return (df['rate'][index], df['psnr_mean'][index])
#psnr = (df['psnr_mean'],df[term])
#scores=np.array((df['psnr_mean'],df['acc1']))
#scores=np.swapaxes(scores,0,1)
#indexes=identify_pareto(scores)
#np.save('pareto',indexes)
#indexes = np.load('pareto.npy')
#g1 = (df['rate'][indexes],df['acc1'][indexes])
#g2 = (df['rate'],df['acc1'])
##g1 = (df['psnr_mean'][indexes],df[term][indexes])
#np.set_printoptions(threshold=sys.maxsize)
#df = pd.read_csv("csv/ga_selection_multi.csv")
#scores=np.array((df['rate'],df['acc1']))
#scores=np.swapaxes(scores,0,1)
#indexes=identify_pareto(scores)
#g3 = (df['rate'][indexes],df['acc1'][indexes]) #g3: pareto of ga
#df = pd.read_csv("csv/standard_part.csv")
#g4 = (df['rate'][1:20],df['acc1'][1:20])
#print(coef)



markers = ["o" , "," , "o" , "v" , "^" , "<", ">"]#.
colors = ['r','g','b','y','c', 'm', 'k']#("red","blue",'yellow','green')
pareto1000 = np.load('pareto1000.npy')
#df = pd.read_csv('csv/sorted.csv')
#scores = np.array((df['rate'],df['acc1']))
#scores = np.swapaxes(scores,0,1)
#pareto = identify_pareto(scores)
groups = {  'psnr': { 'index': slice(None), 'name': 'Sorted Random Search'},
            'psnr_pareto': { 'index': pareto1000 , 'name':'Pareto of Sorted Random Search' }
         }

# Create plot
fig = plt.figure()
ax = fig.add_subplot(111)#axisbg="1.0")
# np.polyfit(g2[0],g2[1], 2)
#x = np.linspace(min(df['rate']), max(df['rate']), 1000)
#y = [ np.sum(np.array([a**2,a,1])*coef) for a in x]
#ax.plot(x,y)
markers = [(i,j,0) for i in range(2,10) for j in range(1, 3)]
for i, k in enumerate(groups.keys()):
    x, y = get_data(k, **groups[k])
    #a = 1 if group=='standard' or 'pareto' else 0.3
    ax.scatter(x, y, s=50, label=groups[k]['name'])


#plt.title('CR pareto vs Acc')
plt.legend(loc=0, fontsize=12)
plt.tick_params(axis="x", labelsize=12)
plt.tick_params(axis="y", labelsize=12)
plt.xlabel('Compression Rate', fontsize=18) #rate
plt.ylabel('PSNR', fontsize=18)
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
