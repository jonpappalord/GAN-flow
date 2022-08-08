from utils import *

import numpy as np
from sklearn.model_selection import train_test_split
import itertools
from random import sample
import pickle
from scipy.spatial import distance
import time
import sys

import random
random.seed(3110)



print(len(sys.argv))





if len(sys.argv) == 1: #no arguments
    cities = ["CHI", "NYC"]
    transps = ["Bike", "Taxi"]
    models = ["Gravity", "Radiation", "MoGAN", "Random_Weighted"]


else:
    cities = [sys.argv[1]]
    transps = [sys.argv[2]]
    if sys.argv[3] == "all":
        models = ["Gravity", "Radiation", "MoGAN"]
    elif sys.argv[3] == "all+R":
        models = ["Gravity", "Radiation", "MoGAN", "Random_Weighted"]
    else:
        models = sys.argv[3:]

print(models)


FLAG_weights = False
FLAG_weights_dist = False
FLAG_cpc = False
FLAG_rmse = False
FLAG_cutnorm = True
FLAG_topo = False
FLAG_degree = False
FLAG_outdegree = False
FLAG_indegree = False




table = {
    "Gravity": "fake_set_gravity.txt",
    "Radiation": "fake_set_radiation.txt",
    "Random_Weighted": "fake_set_random_weighted.txt",
    "Random": "fake_set_random.txt",
    "MoGAN": "fake_set.txt"
}



for model in models:
    for city in cities:
        for transp in transps:
            print(model,city,transp)
            print("--------")
            print("--------")
            distanze = np.load("./DataLoading/dist_mat_" +city +".npy")
            np.fill_diagonal(distanze, 0.01)

            with open("./" +transp + city +"/"+table[model], "rb") as fp:   # Unpickling
                fake_set = pickle.load(fp)
            with open("./" +transp + city +"/v_test.txt", "rb") as fp:   # Unpickling
                v_test = pickle.load(fp)

            #v_test = v_test[:15]
            #fake_set = fake_set[:15]

            number_of_items = np.floor(len(v_test)/(1.5)).astype(int)
            uno = sample(v_test, number_of_items)
            due = sample(fake_set, number_of_items)
            mixed_set_pairs = [pair for pair in itertools.product(uno , due)]
            len(fake_set), len(v_test), len(mixed_set_pairs), number_of_items

            ##-------------------------------------------------------##
            ##-------------------------------------------------------##
            ##-------------------------------------------------------##
            if FLAG_weights:
                exp_weight_1_sim = get_exp_dist(v_test)
                print("exp_weight_1_sim")
                exp_weight_2_sim = get_exp_dist(fake_set)
                print("exp_weight_2_sim")
                exp_weight_3_sim =get_exp_dist(mixed_set_pairs, paired = True)
                print("exp_weight_3_sim")

                with open("./" + transp+city+"/experiments/weight/"+model+"/1.txt", "wb") as fp:   #Pickling
                    pickle.dump(exp_weight_1_sim, fp)
                with open("./" + transp+city+"/experiments/weight/"+model+"/2.txt", "wb") as fp:   #Pickling
                    pickle.dump(exp_weight_2_sim, fp)
                with open("./" + transp+city+"/experiments/weight/"+model+"/3.txt", "wb") as fp:   #Pickling
                    pickle.dump(exp_weight_3_sim, fp)


            if FLAG_weights_dist:
                exp_weight_dist_1_sim = get_exp_dist(v_test,method = "weight-dist", distanze = distanze)
                print("exp_weight_dist_1_sim")
                exp_weight_dist_2_sim = get_exp_dist(fake_set,method = "weight-dist", distanze = distanze)
                print("exp_weight_dist_2_sim")
                exp_weight_dist_3_sim =get_exp_dist(mixed_set_pairs, paired = True, method = "weight-dist", distanze = distanze)
                print("exp_weight_dist_3_sim")

                with open("./" + transp+city+"/experiments/weightdist/"+model+"/1.txt", "wb") as fp:   #Pickling
                    pickle.dump(exp_weight_dist_1_sim, fp)
                with open("./" + transp+city+"/experiments/weightdist/"+model+"/2.txt", "wb") as fp:   #Pickling
                    pickle.dump(exp_weight_dist_2_sim, fp)
                with open("./" + transp+city+"/experiments/weightdist/"+model+"/3.txt", "wb") as fp:   #Pickling
                    pickle.dump(exp_weight_dist_3_sim, fp)


            if FLAG_cutnorm:
                exp_cutnorm_1_sim = get_exp_measures(v_test, method = "cutnorm")
                print("exp_cutnorm_1_sim")
                exp_cutnorm_2_sim = get_exp_measures(fake_set, method = "cutnorm")
                print("exp_cutnorm_2_sim")
                exp_cutnorm_3_sim = get_exp_measures(mixed_set_pairs, paired = True, method="cutnorm")
                print("exp_cutnorm_3_sim")

                with open("./" + transp+city+"/experiments/cutnorm/"+model+"/1.txt", "wb") as fp:   #Pickling
                    pickle.dump(exp_cutnorm_1_sim, fp)
                with open("./" + transp+city+"/experiments/cutnorm/"+model+"/2.txt", "wb") as fp:   #Pickling
                    pickle.dump(exp_cutnorm_2_sim, fp)
                with open("./" + transp+city+"/experiments/cutnorm/"+model+"/3.txt", "wb") as fp:   #Pickling
                    pickle.dump(exp_cutnorm_3_sim, fp)

            if FLAG_cpc:
                exp_cpc_1_sim = get_exp_measures(v_test, method = "cpc")
                print("exp_cpc_1_sim")
                exp_cpc_2_sim = get_exp_measures(fake_set, method = "cpc")
                print("exp_cpc_2_sim")
                exp_cpc_3_sim = get_exp_measures(mixed_set_pairs, paired = True, method="cpc")
                print("exp_cpc_3_sim")

                with open("./" + transp+city+"/experiments/cpc/"+model+"/1.txt", "wb") as fp:   #Pickling
                    pickle.dump(exp_cpc_1_sim, fp)
                with open("./" + transp+city+"/experiments/cpc/"+model+"/2.txt", "wb") as fp:   #Pickling
                    pickle.dump(exp_cpc_2_sim, fp)
                with open("./" + transp+city+"/experiments/cpc/"+model+"/3.txt", "wb") as fp:   #Pickling
                    pickle.dump(exp_cpc_3_sim, fp)

            if FLAG_rmse:
                exp_rmse_1_sim = get_exp_measures(v_test, method = "rmse")
                print("exp_rmse_1_sim")
                exp_rmse_2_sim = get_exp_measures(fake_set, method = "rmse")
                print("exp_rmse_2_sim")
                exp_rmse_3_sim = get_exp_measures(mixed_set_pairs, paired = True, method="rmse")
                print("exp_rmse_3_sim")

                with open("./" + transp+city+"/experiments/rmse/"+model+"/1.txt", "wb") as fp:   #Pickling
                    pickle.dump(exp_rmse_1_sim, fp)
                with open("./" + transp+city+"/experiments/rmse/"+model+"/2.txt", "wb") as fp:   #Pickling
                    pickle.dump(exp_rmse_2_sim, fp)
                with open("./" + transp+city+"/experiments/rmse/"+model+"/3.txt", "wb") as fp:   #Pickling
                    pickle.dump(exp_rmse_3_sim, fp)

            if FLAG_kernel:
                start = time.time()
                exp_kernel_1_sim = get_exp_kernel(v_test)
                end = time.time()
                print("exp_kernel_1_sim")
                elapsed_time = (end-start)/60
                print("elapsed time in minutes: " + str(elapsed_time))

                start = time.time()
                exp_kernel_2_sim = get_exp_kernel(fake_set)
                end = time.time()
                print("exp_kernel_2_sim")
                elapsed_time = (end-start)/60
                print("elapsed time in minutes: " + str(elapsed_time))

                start = time.time()
                exp_kernel_3_sim = get_exp_kernel(mixed_set_pairs, True, uno,due)
                end = time.time()
                print("exp_kernel_3_sim")
                elapsed_time = (end-start)/60
                print("elapsed time in minutes: " + str(elapsed_time))


                with open("./" + transp+city+"/experiments/kernel/"+model+"/1.txt", "wb") as fp:   #Pickling
                    pickle.dump(exp_kernel_1_sim, fp)
                with open("./" + transp+city+"/experiments/kernel/"+model+"/2.txt", "wb") as fp:   #Pickling
                    pickle.dump(exp_kernel_2_sim, fp)
                with open("./" + transp+city+"/experiments/kernel/"+model+"/3.txt", "wb") as fp:   #Pickling
                    pickle.dump(exp_kernel_3_sim, fp)

            if FLAG_topo:

                f = open(city+transp+".txt", "a")
                start = time.time()
                exp_topo_1_sim = get_exp_measures(v_test, method = "topo")
                f.write("exp_topo_1_sim\n")
                #print("exp_topo_1_sim")
                end = time.time()
                elapsed_time = (end-start)/60
                #print("elapsed time in minutes: " + str(elapsed_time))
                f.write("elapsed time in minutes: " + str(elapsed_time)+"\n")

                start = time.time()
                exp_topo_2_sim = get_exp_measures(fake_set, method = "topo")
                f.write("exp_topo_2_sim\n")
                #print("exp_topo_2_sim")
                end = time.time()
                elapsed_time = (end-start)/60
                #print("elapsed time in minutes: " + str(elapsed_time))
                f.write("elapsed time in minutes: " + str(elapsed_time)+"\n")


                start = time.time()
                exp_topo_3_sim = get_exp_measures(mixed_set_pairs, paired = True, method="topo")
                f.write("exp_topo_3_sim\n")
                #print("exp_topo_3_sim")
                end = time.time()
                elapsed_time = (end-start)/60
                #print("elapsed time in minutes: " + str(elapsed_time))
                f.write("elapsed time in minutes: " + str(elapsed_time)+"\n")

                f.close()


                with open("./" + transp+city+"/experiments/topo/"+model+"/1.txt", "wb") as fp:   #Pickling
                    pickle.dump(exp_topo_1_sim, fp)
                with open("./" + transp+city+"/experiments/topo/"+model+"/2.txt", "wb") as fp:   #Pickling
                    pickle.dump(exp_topo_2_sim, fp)
                with open("./" + transp+city+"/experiments/topo/"+model+"/3.txt", "wb") as fp:   #Pickling
                    pickle.dump(exp_topo_3_sim, fp)




            if FLAG_degree:
                start = time.time()
                exp_degree_1_sim = get_exp_measures(v_test, method = "degree")
                print("exp_degree_1_sim")
                end = time.time()
                elapsed_time = (end-start)/60
                print("elapsed time in minutes: " + str(elapsed_time))

                start = time.time()
                exp_degree_2_sim = get_exp_measures(fake_set, method = "degree")
                print("exp_degree_2_sim")
                end = time.time()
                elapsed_time = (end-start)/60
                print("elapsed time in minutes: " + str(elapsed_time))

                start = time.time()
                exp_degree_3_sim = get_exp_measures(mixed_set_pairs, paired = True, method="degree")
                print("exp_degree_3_sim")
                end = time.time()
                elapsed_time = (end-start)/60
                print("elapsed time in minutes: " + str(elapsed_time))


                with open("./" + transp+city+"/experiments/degree/"+model+"/1.txt", "wb") as fp:   #Pickling
                    pickle.dump(exp_degree_1_sim, fp)
                with open("./" + transp+city+"/experiments/degree/"+model+"/2.txt", "wb") as fp:   #Pickling
                    pickle.dump(exp_degree_2_sim, fp)
                with open("./" + transp+city+"/experiments/degree/"+model+"/3.txt", "wb") as fp:   #Pickling
                    pickle.dump(exp_degree_3_sim, fp)





            if FLAG_indegree:
                start = time.time()
                exp_indegree_1_sim = get_exp_measures(v_test, method = "indegree")
                print("exp_indegree_1_sim")
                end = time.time()
                elapsed_time = (end-start)/60
                print("elapsed time in minutes: " + str(elapsed_time))

                start = time.time()
                exp_indegree_2_sim = get_exp_measures(fake_set, method = "indegree")
                print("exp_indegree_2_sim")
                end = time.time()
                elapsed_time = (end-start)/60
                print("elapsed time in minutes: " + str(elapsed_time))

                start = time.time()
                exp_indegree_3_sim = get_exp_measures(mixed_set_pairs, paired = True, method="indegree")
                print("exp_indegree_3_sim")
                end = time.time()
                elapsed_time = (end-start)/60
                print("elapsed time in minutes: " + str(elapsed_time))


                with open("./" + transp+city+"/experiments/indegree/"+model+"/1.txt", "wb") as fp:   #Pickling
                    pickle.dump(exp_indegree_1_sim, fp)
                with open("./" + transp+city+"/experiments/indegree/"+model+"/2.txt", "wb") as fp:   #Pickling
                    pickle.dump(exp_indegree_2_sim, fp)
                with open("./" + transp+city+"/experiments/indegree/"+model+"/3.txt", "wb") as fp:   #Pickling
                    pickle.dump(exp_indegree_3_sim, fp)




            if FLAG_outdegree:
                start = time.time()
                exp_outdegree_1_sim = get_exp_measures(v_test, method = "outdegree")
                print("exp_outdegree_1_sim")
                end = time.time()
                elapsed_time = (end-start)/60
                print("elapsed time in minutes: " + str(elapsed_time))

                start = time.time()
                exp_outdegree_2_sim = get_exp_measures(fake_set, method = "outdegree")
                print("exp_outdegree_2_sim")
                end = time.time()
                elapsed_time = (end-start)/60
                print("elapsed time in minutes: " + str(elapsed_time))

                start = time.time()
                exp_outdegree_3_sim = get_exp_measures(mixed_set_pairs, paired = True, method="outdegree")
                print("exp_outdegree_3_sim")
                end = time.time()
                elapsed_time = (end-start)/60
                print("elapsed time in minutes: " + str(elapsed_time))


                with open("./" + transp+city+"/experiments/outdegree/"+model+"/1.txt", "wb") as fp:   #Pickling
                    pickle.dump(exp_outdegree_1_sim, fp)
                with open("./" + transp+city+"/experiments/outdegree/"+model+"/2.txt", "wb") as fp:   #Pickling
                    pickle.dump(exp_outdegree_2_sim, fp)
                with open("./" + transp+city+"/experiments/outdegree/"+model+"/3.txt", "wb") as fp:   #Pickling
                    pickle.dump(exp_outdegree_3_sim, fp)




if __name__ == "__main__":
    pass
