import os
import numpy as np
from scipy.io import loadmat
from scipy.sparse import csr_matrix, eye, diags
from scipy.sparse import linalg as s_linalg
import time

def sim_kernel(dot_product):
    return np.maximum(np.power(dot_product,3),0)
 
def normalize_connection_graph(G):
    W = csr_matrix(G)
    #W = W - diags(W.diagonal())
    #W[np.isnan(W)] = 0##
    #W[np.isinf(W)] = 0##
    D = np.array(1./ np.sqrt(W.sum(axis = 1)))
    D[np.isnan(D)] = 0
    D[np.isinf(D)] = 0
    D_mh = diags(D.reshape(-1))
    Wn = D_mh * W * D_mh
    return Wn

def topK_W(G, K = 100):
    sortidxs = np.argsort(-G, axis = 1)
    for i in range(G.shape[0]):
        G[i,sortidxs[i,K:]] = 0
    G = np.minimum(G, G.T)
    return G



def cg_diffusion(qsims, Wn, alpha = 0.99, maxiter = 10, tol = 1e-3):
    Wnn = eye(Wn.shape[0]) - alpha * Wn
    out_sims = []
    for i in range(qsims.shape[0]):
        #f,inf = s_linalg.cg(Wnn, qsims[i,:], tol=tol, maxiter=maxiter)
        f,inf = s_linalg.minres(Wnn, qsims[i,:], tol=tol, maxiter=maxiter)
        out_sims.append(f.reshape(-1,1))
    out_sims = np.concatenate(out_sims, axis = 1)
    ranks = np.argsort(-out_sims, axis = 0)
    
    return ranks

