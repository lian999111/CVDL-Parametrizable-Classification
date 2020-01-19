import numpy as np
import matplotlib as plt
from mpl_toolkits.mplot3d import Axes3D
import random 

def plot3D(Embeddings,ylabels):
    fig = plt.pyplot.figure()
    ax = fig.add_subplot(111, projection='3d')
    i = 0
    xs = np.asarray([])
    ys = np.asarray([])
    zs = np.asarray([])
    while i<Embeddings.size:
        xs = np.append(xs,Embeddings[i])
        ys = np.append(ys,Embeddings[i + 1])
        zs = np.append(zs,Embeddings[i + 2])
        i=i+3
        
    colors = ['red','green','blue','yellow','grey','cyan','pink','black','orange','chartreuse']    
    scatter = ax.scatter(xs, ys, zs, c = ylabels, cmap=plt.colors.ListedColormap(colors))
    legend1 = ax.legend(*scatter.legend_elements(),loc="lower left", title="Classes")
    ax.add_artist(legend1)
    
def Rand(start, end, num): 
    res = [] 
  
    for j in range(num): 
        res.append(random.randint(start, end)) 
  
    return res 
  
# Driver Code 
num = 12
start = 0
end = 20
test_vec = np.asarray(Rand(start, end, num)) 
ylabels = np.asarray(Rand(0, 9, 4))
plot3D(test_vec,ylabels)