import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from random import sample
from scipy.spatial import distance

from cutnorm import compute_cutnorm
import skmob
from skmob.measures import evaluation


import networkx as nx
from tqdm import tqdm

#import warnings
#warnings.filterwarnings("ignore")

import random
random.seed(3110)

def get_exp_dist(lista, paired = False, method = "weight-edge", distanze = None):

    exp = []
    js = 0

    if paired:
        insieme = lista
    else:
        insieme = itertools.combinations(lista, r =2)

    for pair in insieme:


        if method == "weight-edge":
            weights_1 = pair[0].flatten()
            weights_2 = pair[1].flatten()
        elif method == "weight-dist":
            weights_1 = (pair[0]/distanze).flatten()
            weights_2 = (pair[1]/distanze).flatten()


        massim = max(max(weights_1), max(weights_2))
        bins = np.arange(0,np.ceil(massim)  )

        values_1, base_1 = np.histogram(weights_1, bins=bins, density=1)
        values_2, base_2 = np.histogram(weights_2, bins=bins, density=1)

        js = distance.jensenshannon(np.asarray(values_1), np.asarray(values_2), np.e)

        exp.append(js)

    return exp

def get_exp_measures(lista, paired = False, method = "cutnorm"):
    exp = []

    if paired:
        insieme = lista
    else:
        insieme = itertools.combinations(lista, r =2)

    if method =="cutnorm":
        k = 0
        for pair in tqdm(insieme):
                _, cutn_sdp, _ = compute_cutnorm(pair[0], pair[1])
                exp.append(cutn_sdp)
                k+=1
        return exp

    elif method == "topo":
            exp = []

            for pair in tqdm(insieme):

                G1 = nx.from_numpy_matrix(np.matrix(pair[0]), create_using=nx.DiGraph)
                G2 = nx.from_numpy_matrix(np.matrix(pair[1]), create_using=nx.DiGraph)
                cl1 = list(nx.clustering(G1,weight='weight').values())
                cl2 = list(nx.clustering(G2,weight='weight').values())
                rmse = evaluation.rmse(cl1,cl2)
                nrmse = rmse/(max(np.max(cl1),np.max(cl2)) - min(np.min(cl1), np.min(cl2)))
                exp.append(nrmse)
            return exp


    elif method == "degree":
            exp = []

            for pair in tqdm(insieme):
                G1 = nx.from_numpy_matrix(np.matrix(pair[0]), create_using=nx.DiGraph)
                G2 = nx.from_numpy_matrix(np.matrix(pair[1]), create_using=nx.DiGraph)
                deg1 = [val for (node, val) in G1.degree(weight = "weight")]
                deg2 = [val for (node, val) in G2.degree(weight = "weight")]
                rmse = evaluation.rmse(deg1,deg2)
                nrmse = rmse/(max(np.max(deg1),np.max(deg2)) - min(np.min(deg1), np.min(deg2)))
                exp.append(nrmse)
            return exp



    elif method == "indegree":
            exp = []

            for pair in tqdm(insieme):
                G1 = nx.from_numpy_matrix(np.matrix(pair[0]), create_using=nx.DiGraph)
                G2 = nx.from_numpy_matrix(np.matrix(pair[1]), create_using=nx.DiGraph)
                deg1 = [val for (node, val) in G1.in_degree(weight = "weight")]
                deg2 = [val for (node, val) in G2.in_degree(weight = "weight")]
                rmse = evaluation.rmse(deg1,deg2)
                nrmse = rmse/(max(np.max(deg1),np.max(deg2)) - min(np.min(deg1), np.min(deg2)))
                exp.append(nrmse)
            return exp


    elif method == "outdegree":
            exp = []

            for pair in tqdm(insieme):
                G1 = nx.from_numpy_matrix(np.matrix(pair[0]), create_using=nx.DiGraph)
                G2 = nx.from_numpy_matrix(np.matrix(pair[1]), create_using=nx.DiGraph)
                deg1 = [val for (node, val) in G1.out_degree(weight = "weight")]
                deg2 = [val for (node, val) in G2.out_degree(weight = "weight")]
                rmse = evaluation.rmse(deg1,deg2)
                nrmse = rmse/(max(np.max(deg1),np.max(deg2)) - min(np.min(deg1), np.min(deg2)))
                exp.append(nrmse)
            return exp



    else:
        if method == "cpc":
            misura =  evaluation.common_part_of_commuters
            exp=[]
            for pair in insieme:
                weights_1 = (pair[0]).flatten()
                weights_2 = (pair[1]).flatten()
                m = misura(weights_1, weights_2)
                exp.append(m)
            return exp

        elif method == "rmse":
            misura = evaluation.rmse
            exp=[]
            for pair in insieme:
                weights_1 = (pair[0]).flatten()
                weights_2 = (pair[1]).flatten()
                rmse = misura(weights_1, weights_2)
                nrmse = rmse/(max(np.max(weights_1),np.max(weights_2)) - min(np.min(weights_1), np.min(weights_2)))
                exp.append(nrmse)
            return exp


if __name__ == "__main__":
    pass
