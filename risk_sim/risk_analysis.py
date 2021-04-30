import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

rdf = pd.read_json('risk_data.json')


filteredrdf = rdf[(rdf['maxDeployRad']==300)]

with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    print(filteredrdf)

# table of results
for k in filteredrdf['k'].unique():
    print('** k', k)
    for t in filteredrdf[np.isclose(filteredrdf['k'], k)]['Thresh'].unique():
        filrdf = filteredrdf[np.isclose(filteredrdf['k'], k) & np.isclose(filteredrdf['Thresh'], t)]
        regular_list = filrdf['Distance'].to_numpy().tolist()
        flat_list = [item for sublist in regular_list for item in sublist]
        s = ((filrdf['Num_Met_Thresh']*filrdf['Entered_PZ']*0.01)/(filrdf['Len_Others']+filrdf['Num_Met_Thresh']*filrdf['Entered_PZ']*0.01))
        print('thresh',t,'num tests',len(filrdf) ,'Mean met thresh:', filrdf['Num_Met_Thresh'].mean(), '(',filrdf['Num_Met_Thresh'].std(),'). percentage thresh entered pz:', filrdf['Entered_PZ'].mean(), '(',filrdf['Entered_PZ'].std(), ')')
        print('      mean dist', np.mean(flat_list), '(', np.std(flat_list), ') . % of flocks en pz:',  s.mean()*100,'(', s.std()*100,')')

# histogram
k = 1.0
thresh = 0.975
filteredrdf = rdf[(rdf['maxDeployRad']==300) & (np.isclose(rdf['k'], k)) & (np.isclose(rdf['Thresh'], thresh))]

s = 'k = ' + str(k) + '\nThreshold = ' + str(thresh)

regular_list = filteredrdf['Distance'].to_numpy().tolist()
flat_list = [item*10 for sublist in regular_list for item in sublist]

plt.hist(flat_list, bins = 25,alpha=1, rwidth=0.85)
plt.xlabel('Distance flock met threshold (m)')
plt.ylabel('Count')
plt.xlim(xmin=1500, xmax = 3000)
ax = plt.gca()
ax.set_axisbelow(True)
ax.yaxis.grid(color='gray', linestyle='dashed', alpha=0.7)
plt.text(1600, 36, s)

plt.show()

