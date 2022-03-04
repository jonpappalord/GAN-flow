import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from random import sample
from scipy.spatial import distance

from cutnorm import compute_cutnorm
import skmob
from skmob.measures import evaluation



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
        for pair in insieme:    
                _, cutn_sdp, _ = compute_cutnorm(pair[0], pair[1])
                exp.append(cutn_sdp)

        return exp
    
    else:
        if method == "cpc":
            misura =  evaluation.common_part_of_commuters
        elif method == "rmse":
            misura = evaluation.rmse
        for pair in insieme:
            weights_1 = (pair[0]).flatten()
            weights_2 = (pair[1]).flatten()
            m = misura(weights_1, weights_2)
            exp.append(m)
        return exp
    
if __name__ == "__main__":
    pass
    