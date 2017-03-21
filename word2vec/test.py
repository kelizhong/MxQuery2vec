from utils.data_util import load_pickle_object
from numpy import linalg as la
import numpy as np

def cosSimilar(inA,inB):
    inA=np.mat(inA)
    inB=np.mat(inB)
    num=float(inA*inB.T)
    denom=la.norm(inA)*la.norm(inB)
    return 0.5+0.5*(num/denom)

word2vec_path = '../data/word2vec/w2v.pkl'
w2v = load_pickle_object(word2vec_path)

print(cosSimilar(w2v['nike'], w2v['iphone6']))
print(cosSimilar(w2v['iphone'], w2v['iphone6']))
print(cosSimilar(w2v['iphone'], w2v['apple']))