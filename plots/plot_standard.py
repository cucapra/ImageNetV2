import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import glob
import os
os.chdir('../jpeg_eval/')
#df  = pd.read_csv("ratio.csv")
#pl = df.plot(kind='scatter',x='Comp Rate',y='Acc', s=30, color ='Blue', label='random jpeg')
part = sys.argv[1] if len(sys.argv) > 1 else ''
print(part)
term = 'acc1'#rate
df = pd.read_csv("csv/standard_"+part+".csv")
ga = df['i'].str.startswith("/data/")
g1 = (df['rate'][ga],df['acc1'][ga])
sort = df['i'].str.startswith("qtable")
g2 = (df[sort]['rate'], df[sort]['acc1'])
st = df['i'].str.isnumeric()
g3 = (df['rate'][st][1:], df['acc1'][st][1:])
print(g3)

coef = np.polyfit(g2[0],g2[1], 2)



#print(coef)
markers = ["v" , ",", "o" ]# "v" , "^" , "<", ">"]
data = (g2,g1,g3)
#data = (g4, g1, g2, g3,g5)
colors = ['r','g','b']#,'y','c', 'm', 'k']#("red","blue",'yellow','green')

groups = ('sorted','ga', 'standard')
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

plt.title('CR pareto vs Acc')
plt.legend(loc=0)
plt.xlabel('rate') #rate
plt.ylabel(term)
os.chdir('../plots/')
plt.savefig('other_part.png')
plt.show()

#df = pd.read_csv("libjpeg_random.csv")
#pl = df.plot.scatter(x='Comp Rate', y='Acc', s=30, color='Blue', label='random jpeg',ax=pl);
#df = pd.read_csv("libjpeg_perturbed.csv")
#pl = df.plot.scatter(x='Comp Rate', y='Acc', s=30, color='Yellow', label='perturbed jpeg',ax=pl);


#print(df[1:])

#ax.scatter(x=df['rate'][1:], y=df['acc1'][1:], s=45, color='Green', label='standard jpeg')


#fig = pl.get_figure()
#fig.save
