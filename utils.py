import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from random import sample
from scipy.spatial import distance

from cutnorm import compute_cutnorm
import skmob
from skmob.measures import evaluation

from grakel.utils import graph_from_networkx
from grakel.kernels import EdgeHistogram
from grakel.kernels import OddSth as kk

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
        for pair in insieme:
                if k%100==0:
                    print(k)
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
                cl1 = cl1/np.linalg.norm(cl1)
                cl2 = cl2/np.linalg.norm(cl2)
                exp.append(evaluation.rmse(cl1,cl2))
            return exp

    elif method == "topo_unweighted":
            exp = []

            for pair in tqdm(insieme):
                G1 = nx.from_numpy_matrix(np.matrix(pair[0]), create_using=nx.DiGraph)
                G2 = nx.from_numpy_matrix(np.matrix(pair[1]), create_using=nx.DiGraph)
                cl1 = list(nx.clustering(G1).values())
                cl2 = list(nx.clustering(G2).values())
                cl1 = cl1/np.linalg.norm(cl1)
                cl2 = cl2/np.linalg.norm(cl2)
                exp.append(evaluation.rmse(cl1,cl2))
            return exp

    elif method == "degree":
            exp = []

            for pair in tqdm(insieme):
                G1 = nx.from_numpy_matrix(np.matrix(pair[0]), create_using=nx.DiGraph)
                G2 = nx.from_numpy_matrix(np.matrix(pair[1]), create_using=nx.DiGraph)
                deg1 = [val for (node, val) in G1.degree(weight = "weight")]
                deg2 = [val for (node, val) in G2.degree(weight = "weight")]
                deg1 = deg1/np.linalg.norm(deg1)
                deg2 = deg2/np.linalg.norm(deg2)
                exp.append(evaluation.rmse(deg1,deg2))
            return exp

    elif method == "degree_unweighted":
            exp = []

            for pair in tqdm(insieme):
                G1 = nx.from_numpy_matrix(np.matrix(pair[0]), create_using=nx.DiGraph)
                G2 = nx.from_numpy_matrix(np.matrix(pair[1]), create_using=nx.DiGraph)
                deg1 = [val for (node, val) in G1.degree()]
                deg2 = [val for (node, val) in G2.degree()]
                deg1 = deg1/np.linalg.norm(deg1)
                deg2 = deg2/np.linalg.norm(deg2)
                exp.append(evaluation.rmse(deg1,deg2))
            return exp


    elif method == "indegree":
            exp = []

            for pair in tqdm(insieme):
                G1 = nx.from_numpy_matrix(np.matrix(pair[0]), create_using=nx.DiGraph)
                G2 = nx.from_numpy_matrix(np.matrix(pair[1]), create_using=nx.DiGraph)
                deg1 = [val for (node, val) in G1.in_degree(weight = "weight")]
                deg2 = [val for (node, val) in G2.in_degree(weight = "weight")]
                deg1 = deg1/np.linalg.norm(deg1)
                deg2 = deg2/np.linalg.norm(deg2)
                exp.append(evaluation.rmse(deg1,deg2))
            return exp

    elif method == "indegree_unweighted":
            exp = []

            for pair in tqdm(insieme):
                G1 = nx.from_numpy_matrix(np.matrix(pair[0]), create_using=nx.DiGraph)
                G2 = nx.from_numpy_matrix(np.matrix(pair[1]), create_using=nx.DiGraph)
                deg1 = [val for (node, val) in G1.in_degree()]
                deg2 = [val for (node, val) in G2.in_degree()]
                deg1 = deg1/np.linalg.norm(deg1)
                deg2 = deg2/np.linalg.norm(deg2)
                exp.append(evaluation.rmse(deg1,deg2))
            return exp

    elif method == "outdegree":
            exp = []

            for pair in tqdm(insieme):
                G1 = nx.from_numpy_matrix(np.matrix(pair[0]), create_using=nx.DiGraph)
                G2 = nx.from_numpy_matrix(np.matrix(pair[1]), create_using=nx.DiGraph)
                deg1 = [val for (node, val) in G1.out_degree(weight = "weight")]
                deg2 = [val for (node, val) in G2.out_degree(weight = "weight")]
                deg1 = deg1/np.linalg.norm(deg1)
                deg2 = deg2/np.linalg.norm(deg2)
                exp.append(evaluation.rmse(deg1,deg2))
            return exp

    elif method == "outdegree_unweighted":
            exp = []

            for pair in tqdm(insieme):
                G1 = nx.from_numpy_matrix(np.matrix(pair[0]), create_using=nx.DiGraph)
                G2 = nx.from_numpy_matrix(np.matrix(pair[1]), create_using=nx.DiGraph)
                deg1 = [val for (node, val) in G1.out_degree()]
                deg2 = [val for (node, val) in G2.out_degree()]
                deg1 = deg1/np.linalg.norm(deg1)
                deg2 = deg2/np.linalg.norm(deg2)
                exp.append(evaluation.rmse(deg1,deg2))
            return exp


    else:
        if method == "cpc":
            misura =  evaluation.common_part_of_commuters
        elif method == "rmse":
            misura = evaluation.rmse
        for pair in insieme:
            weights_1 = (pair[0]).flatten()
            weights_2 = (pair[1]).flatten()
            if method =="rmse":
                weights_1 = weights_1/np.linalg.norm(weights_1)
                weights_2 = weights_2/np.linalg.norm(weights_2)
            m = misura(weights_1, weights_2)
            exp.append(m)
        return exp


def get_exp_kernel(insieme, paired = False):

    if not paired:
        l = []
        for A in insieme:
            G = nx.from_numpy_matrix(np.matrix(A), create_using=nx.DiGraph)
            l.append(G)

        for g in l:
            for n in g.nodes():
                for i in nx.ego_graph(g,n).nodes():
                    w = 0
                    if g.has_edge(n,i):
                        w = w + g.edges()[(n,i)]["weight"]
                    g.nodes()[n]["w"] = w

        G = graph_from_networkx(l,edge_labels_tag="weight", node_labels_tag="w")


        gk = kk(normalize = True)
        print("train")
        K_train = gk.fit_transform(G)

        exp = K_train[np.triu_indices(K_train.shape[0], k=1)]

        return exp

    else: #pairwise comparison, take only one value
        exp = []
        for pair in tqdm(insieme):
            l = []

            G1 = nx.from_numpy_matrix(np.matrix(pair[0]), create_using=nx.DiGraph)
            G2 = nx.from_numpy_matrix(np.matrix(pair[1]), create_using=nx.DiGraph)
            l.append(G1)
            l.append(G2)

            for g in l:
                for n in g.nodes():
                    for i in nx.ego_graph(g,n).nodes():
                        w = 0
                        if g.has_edge(n,i):
                            w = w + g.edges()[(n,i)]["weight"]
                        g.nodes()[n]["w"] = w

            G = graph_from_networkx(l,edge_labels_tag="weight", node_labels_tag="w")

            gk = kk(normalize = True)
            K_train = gk.fit_transform(G)
            sim = K_train[0,1]
            exp.append(sim)
        return exp

if __name__ == "__main__":
    pass
