import os
import numpy as np

from scipy.io import loadmat
import pickle
from dataset import configdataset
from download import download_datasets, download_features
from evaluate import compute_map

#---------------------------------------------------------------------
# Set data folder and testing parameters
#---------------------------------------------------------------------
# Set data folder, change if you have downloaded the data somewhere else
data_root = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'data')
# Check, and, if necessary, download test data (Oxford and Pairs), 
# revisited annotation, and example feature vectors for evaluation
download_datasets(data_root)
download_features(data_root)

# Set test dataset: roxford5k | rparis6k
dataset = 'rparis6k'

#---------------------------------------------------------------------
# Evaluate
#---------------------------------------------------------------------
print('>> {}: Evaluating test dataset...'.format(dataset)) 
cfg = configdataset(dataset, os.path.join(data_root, 'datasets'))
# load query and database features
print('>> {}: Loading features...'.format(dataset))    
#features = loadmat('D:/CBIR code/learnedcode/effinetB1Rmac/paris6k.mat')
features = loadmat('D:/CBIR code/learnedcode/effinetb1a/paris6k.mat')
Q = features['Q']
X = features['X']

K = 100 # approx 50 mutual nns

QUERYKNN = 10
#R = 2000
alpha = 0.9

from diffussion import *

# perform search
print('>> {}: Retrieval...'.format(dataset))
from sklearn.metrics import pairwise_distances 
from scipy.spatial.distance import cosine
sim=1-pairwise_distances(X.T,Q.T,metric='cosine')
#sim= np.dot(X.T, Q)
qsim = sim_kernel(sim).T

sortidxs = np.argsort(-qsim, axis = 1)
for i in range(len(qsim)):
    qsim[i,sortidxs[i,QUERYKNN:]] = 0
from numpy.linalg import norm
qsim = sim_kernel(qsim)

A=1-pairwise_distances(X.T,metric='cosine')
W = sim_kernel(A).T
W = topK_W(W, K)
Wn = normalize_connection_graph(W)

plain_ranks = np.argsort(-sim, axis=0)
cg_ranks =  cg_diffusion(qsim, Wn, alpha)

alg_names = ['Plain','Diffusion cg']
alg_ranks = [plain_ranks, cg_ranks]
for rn in range(len(alg_names)):
    ranks = alg_ranks[rn]
    name = alg_names[rn]
    ks = [1, 10, 100]
    gnd_t = []
    #pickle_in=open('D:/CBIR code/learnedcode/effinetB1Rmac/gnd_t.pickle','rb')
    pickle_in=open('D:/CBIR code/learnedcode/effinetb1a/gnd_t.pickle','rb')
    gnd_t=pickle.load(pickle_in)
    cfg['gnd']=gnd_t
    gnd = cfg['gnd']
    pickle_in.close()
    mapE, apsE, mprE, prsE = compute_map(ranks, gnd_t, ks)
    print(name)
    print('>> {}: mAP : E: {} '.format(dataset, np.around(mapE*100, decimals=2)))
    print('>> {}: mP@k{} : {}'.format(dataset, np.array(ks), np.around(mprE*100, decimals=2)))

from ShowImage import *
cg_ranks=np.array(cg_ranks)
cg_ranks=cg_ranks.transpose()
plain_ranks=np.array(plain_ranks)
plain_ranks=plain_ranks.transpose()
#images functuion (query number , ranked_list=[] , gnd-t=[]  , number_of_images_to_show=10)
images_functuion(q=19 ,ranked_list=plain_ranks , gnd_t=gnd ,number_of_the_first=50 ,number_of_the_last=100)