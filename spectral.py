import numpy as np
from numpy import linalg as LA
from K_means import Kmeans,diff,calcRand
from numpy.linalg import inv
import math
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


alpha = 0.01

def load_data(filename):
    file = open(filename, 'r')
    genes = []
    indexs = []
    y = []
    for line in file:
        data = line.split()
        if data[1] == -1: continue
        indexs.append(data[0])
        genes.append(data[2:])
        y.append(data[1])
#     genes = np.unique(np.array(genes,dtype=float), axis=0)
#     genes = np.concatenate((np.array(range(genes.shape[0])).reshape(-1,1)+1, genes),axis=1)
    genes = np.array(genes,dtype=float)
    return genes,np.array(y).reshape(-1,1)

# Bulid weighted threshold-neighborhood graph
def bulid_sim_matrix(points,threshold):
    sim_matrix = np.zeros((len(points),len(points)))
    for i in range(len(points)):
        for j in range(i+1,len(points)):
            sim = similarity(points[i],points[j])
#             print(distance(points[i],points[j]),i,j)
            if sim > threshold:
                sim_matrix[i][j] = similarity(points[i],points[j])
                sim_matrix[j][i] = sim_matrix[i][j]
    return sim_matrix

# Build k-neighborhood graph
def bulid_k_neighboor_sim_matrix(points,k):
    sim_matrix = np.full((len(points),len(points)),-1.0)
    for i in range(len(points)):
        for j in range(i+1,len(points)):
            sim = similarity(points[i],points[j])
#             print(distance(points[i],points[j]),i,j)
            sim_matrix[i][j] = similarity(points[i],points[j])
            sim_matrix[j][i] = sim_matrix[i][j]
        flat = sim_matrix[i].flatten()
#         print(flat)
        flat.sort()
        kth = flat[0-k]
        for j in range(len(points)):
            if sim_matrix[i][j] < kth:
                sim_matrix[i][j] = 0
    return sim_matrix

def laplacian_matrix(points,threshold):
#     sim_matrix = bulid_sim_matrix(points,threshold)
    sim_matrix = bulid_k_neighboor_sim_matrix(points,threshold)
# #     print(sim_matrix.shape)
    degree_matrix = np.zeros((sim_matrix.shape),dtype=float)
    for i in range(sim_matrix.shape[0]):
        degree_matrix[i][i] = np.sum(sim_matrix[i])
#     print(degree_matrix.shape)
    return inv(degree_matrix).dot(degree_matrix-sim_matrix)
#     print(np.array(degree_matrix-sim_matrix).shape)
#     return np.array(degree_matrix-sim_matrix)

def bi_partition(L):
    eigenvalues, eigenvectors = LA.eig(L)
#     print(eigenvalues[0],eigenvectors.shape)
    x = np.concatenate((np.array(eigenvectors[1]).reshape(-1,1),np.array(range(eigenvectors.shape[0])).reshape(-1,1)+1),axis=1)
    x = x[x[:,0].argsort()]
#     print(x)
    index = 0
    while x[index][0] < 0:
        index += 1
    C1 = x[0:index][:,1].astype(int)
    C2 = x[index:][:,1].astype(int)
#     print(C1,C2)
    return C1, C2

def generate_z(points,threshold,K):
    L = laplacian_matrix(points,threshold)
#     print(L.shape)
    eigenvalues, eigenvectors = LA.eig(L)
#     print(eigenvectors.shape)
#     V = eigenvectors[:,:K]
    V = eigenvectors[:K].T
#     print(V.shape)
#     Z = np.concatenate((V,np.array(range(V.shape[0])).reshape(-1,1)+1),axis=1)
#     print(V)
    return V

def distance(p1, p2):
    return np.sqrt(np.sum(np.power(p1-p2,2)))

def similarity(p1,p2):
#     print((1.0+alpha)/(distance(p1,p2)+alpha))
    if(distance(p1,p2)<0): return 0
    return (1.0+alpha)/(distance(p1,p2)+alpha)

def figure_plot(data, label, x_label, y_label, title):
    fig = plt.figure().add_subplot(111)
    colors = ['black', 'red', 'lime', 'blue', 'yellow', 'magenta', 'gray', 'Teal', 'Green', 'Cyan', 'Olive']
    labels = {}
    for i in range(label.size):
        if str(label[i]) not in labels:
            labels[str(label[i])] = data[i:i+1]
        else:
            labels[str(label[i])] = np.concatenate((labels[str(label[i])], data[i:i+1]), axis=0)
    for id in labels:
        fig.scatter(labels[id][:, 0], labels[id][:, 1], c=colors[0], label=id[2:-2])
        colors.remove(colors[0])
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.show()


filename = 'iyer.txt'
# filename = 'cho.txt'

x,y = load_data(filename)
K = int(y[-1])
Z = generate_z(x,50,K)
center,clusters=Kmeans(Z,K)
accuracy,corresponding=diff(clusters,y,K)
rand=calcRand(clusters,y)
# print("Accuracy: ",accuracy)
print("Rand: ",rand)
# print("Corresponding: ",corresponding)


PCA_test = PCA(n_components = 2)  # PCA Library
res_PCA_test = PCA_test.fit_transform(x)  # Dimension reduction
figure_plot(res_PCA_test, clusters, 'Component 1', 'Component 2', filename)