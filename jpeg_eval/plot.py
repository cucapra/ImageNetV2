import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import glob
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
ga_list = glob.glob('csv/ga_selection_*_*.csv')
print(ga_list)
rate=[]
acc1=[]
for name in ga_list:
    df = pd.read_csv(name)
    rate.append(np.array(df['rate'])[-1])
    acc1.append(np.array(df['acc1'])[-1])
g5 = (rate,acc1)
#df  = pd.read_csv("ratio.csv")
#pl = df.plot(kind='scatter',x='Comp Rate',y='Acc', s=30, color ='Blue', label='random jpeg')
term = 'rate'#rate
df = pd.read_csv("sorted_psnr.csv")
psnr = (df['psnr_mean'],df[term])
#scores=np.array((df['psnr_mean'],df['acc1']))
#scores=np.swapaxes(scores,0,1)
#indexes=identify_pareto(scores)
#np.save('pareto',indexes)
indexes = np.load('pareto.npy')
#g1 = (df['rate'],df['acc1'])
g1 = (df['psnr_mean'][indexes],df[term][indexes])
np.set_printoptions(threshold=sys.maxsize)
df = pd.read_csv("csv/ga_selection_multi.csv")
scores=np.array((df['rate'],df['acc1']))
scores=np.swapaxes(scores,0,1)
indexes=identify_pareto(scores)
g2 = (df['rate'][:64],df['acc1'][:64])
g3 = (df['rate'][indexes],df['acc1'][indexes])
df = pd.read_csv("csv/standard.csv")
g4 = (df['rate'][1:20],df['acc1'][1:20])
coef = np.polyfit(g3[0],g3[1], 2)



#print(coef)
markers = ["." , ","]# , "o" , "v" , "^" , "<", ">"]
data = (psnr,g1)
#data = (g4, g1, g2, g3,g5)
colors = ['r','g']#,'b','y','c', 'm', 'k']#("red","blue",'yellow','green')

groups = ('sorted random psnr', 'pareto psnr')
#groups = ('standard',"pareto for RS", "pareto one for RS", "pareto with GA",'individual GA')

# Create plot
fig = plt.figure()
ax = fig.add_subplot(111)#axisbg="1.0")
x = np.linspace(min(df['rate']), max(df['rate']), 1000)
y = [ np.sum(np.array([a**2,a,1])*coef) for a in x]
#ax.plot(x,y)
for data, color, group,m in zip(data, colors, groups,markers):
    x, y = data
    a = 1 if group=='standard' or 'pareto' else 0.3
    ax.scatter(x, y, alpha=a, c=color, marker = m, edgecolors='none', s=30, label=group)

#plt.plot(df['rate'][10], [df['acc1'][10]], marker='o', markersize=3, color="red")

#plt.title('CR pareto vs Acc')
plt.legend(loc=0)
plt.xlabel('psnr') #rate
plt.ylabel(term)
plt.savefig('rate_qtable.png')
plt.show()

#df = pd.read_csv("libjpeg_random.csv")
#pl = df.plot.scatter(x='Comp Rate', y='Acc', s=30, color='Blue', label='random jpeg',ax=pl);
#df = pd.read_csv("libjpeg_perturbed.csv")
#pl = df.plot.scatter(x='Comp Rate', y='Acc', s=30, color='Yellow', label='perturbed jpeg',ax=pl);


#print(df[1:])

#ax.scatter(x=df['rate'][1:], y=df['acc1'][1:], s=45, color='Green', label='standard jpeg')


#fig = pl.get_figure()
#fig.save
