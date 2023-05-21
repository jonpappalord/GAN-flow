# MoGAN - Generating mobility networks with GANs
## Table of contents
1. [Citing](#citing)
2. [Packages](#packages)
3. [Abstract](#abstract)
4. [Data Availability](#data-availability)
5. [Structure of the repository](#structure-of-the-repository)
6. [Analysis](#analysis)
7. [DataLoading](#dataloading)


# Citing
In this repository you can find the code for running MoGAN model and to replicate the analysis conducted in our paper.
If you use the code in this repository, please cite our paper:

*Mauro, G., Luca, M., Longa, A., Lepri, B., & Pappalardo, L. (2022). Generating mobility networks with generative adversarial networks. EPJ data science, 11(1), 58.*

```
@article{mauro2022generating,
  title={Generating mobility networks with generative adversarial networks},
  author={Mauro, Giovanni and Luca, Massimiliano and Longa, Antonio and Lepri, Bruno and Pappalardo, Luca},
  journal={EPJ data science},
  volume={11},
  number={1},
  pages={58},
  year={2022},
  publisher={Springer Berlin Heidelberg}
}
```

# Packages
For running notebooks and scripts of this project you must install the following Python packages:
```
  pytorch
  torchvision
  scikit-mobility
  seaborn
  cutnorm
```
These packages will automatically install all of the other required ones (e.g ```matplotilib```, ```geopandas```, ```scipy```, ```numpy```).


# Abstract
The increasingly crucial role of human displacements in complex societal phenomena, such as traffic congestion, segregation, and the diffusion of epidemics, is attracting the interest of scientists from several disciplines.
Here, we address mobility network generation, i.e., generating a city's entire mobility network, a weighted directed graph in which nodes are geographic locations and weighted edges represent people's movements between those locations, thus describing the entire mobility set flows within a city.
Our solution is MoGAN, a model based on Generative Adversarial Networks (GANs) to generate realistic mobility networks.
We conduct extensive experiments on public datasets of bike and taxi rides to show that MoGAN outperforms the classical Gravity and Radiation models regarding the realism of the generated networks.
Our model can be used for data augmentation and performing simulations and what-if analysis.

![data_schema](https://github.com/jonpappalord/GAN-flow/blob/main/gan_schema.png?raw=true)



# Data Availability
All of the four used dataset are openly available online. Please, if you want to download these files use the scripts we will present in folowing sections. Otherwise select referring to years 2018 and 2019. 
- Data for New York City bike sharing system ("BikeNYC" hereinafter) may be found at: https://s3.amazonaws.com/tripdata/index.html
- Data for New York City taxis ("TaxiNYC" hereinafter) may be found at: https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page
    - We used yellow taxi data
- Data for Chicago bike sharing system ("BikeCHI" hereinafter) may be found at: https://divvy-tripdata.s3.amazonaws.com/index.html
- Data for Chicago taxis ("TaxiCHI" hereinafter) may be found at: https://data.cityofchicago.org/Transportation/Taxi-Trips/wrvz-psew
    1. **Warning** these dataset is huge. Use the 2018 and 2019 view 
        - 2018: https://data.cityofchicago.org/Transportation/Taxi-Trips-2018/vbsw-zws8
        - 2019: https://data.cityofchicago.org/Transportation/Taxi-Trips-2019/h4cq-z3dy 
    2. **Warning** These data are updated frequently. The current data are different from the ones used in our analysis 


# Structure of the repository
In the **main** level of the repo you can find:
- ```Gravity.py``` , ``` Radiation.py``` and ```Weighted_Random.py```
    - These codes execute the three baseline models
- ```MoGAN.py```
    - This script execute the actual MoGAN model
- ```analysis.py``` 
    - Performs the experimental phases for the three models, using functions defined in ```utils.py```

The folder ```plots``` contains the pdf plots and the code for creating them in the notebook ```Plots.ipynb``` for the comparison with Gravity and Radiation model and in the notebook ```Plots_Randoms.ipynb``` for the comparison with Random Weighted model. Both notebooks will also calculate the tables of comparison.


The folders ```BikeCHI/```, ```BikeNYC/```, ```TaxiCHI/``` and ```TaxiNYC/``` holds the result files of the execution of the three models over the three distinct datasets like the set of synthetic elements (```fake_set.txt```) or the subfolders holding the results of the experimental phase (```experiments/```). The other subfolders content will be explained in next sections.

If you want to run the models over the precalculated adjacency matrices you just need to launch the python script of the model passing as argument the city and the mean of transport (i.e. the dataset) you want to use.

For example, if you want to generate synthetic mobility networks of bikes in NYC using MoGAN you should execute:
```sh
python MoGAN.py -NYC -Bike
```
or if you want to do that for the networks of taxis of Chicago you should run:
```sh
python MoGAN.py -CHI -Taxi
```
These script will train MoGAN or the two baseline models over the pre-calculated adjacency matrices you can find in ```adj/``` folder. And will save the results in the folder of each dataset (we also leave the pre-calculated results). After that it is possible to perform the [Analysis](#analysis) phase. Otherwise, if you want to reconstruct the adjacency matrix, you may want to perform the [DataLoading](#dataloading) phase.

    

# Analysis 
As reported in the article, we perform a massive experimental phase, implementing a tailored approach for evaluating the realism of the synthetic networks. For easiness of usage, we created the script ```Analysis.py``` for creating, per each model, the distributions of Mixed, Syntetic and Test Set, over the four datasets (see our article for more details). If you run the ```Analysis.py``` without any argument, it will replicate the analysis for all of the models, all of the cities and all of the means of transports. Otherwise, the first argument will be the city ("NYC" or "CHI"), the second the mean of transport ("Bike" or "Taxi") and the third the model ("Gravity", "Radiation", "MoGAN" or "Random_Weighted"). Given the heaviness, in terms of calculation time, of the experiments, you can set to True or False the flag of each metric you want to calculate: FLAG_weights and FLAG_weights_dist will allow the experimets of the weights and weight-distances, FLAG_cpc, FLAG_rmse and FLAG_cutnorm will allow the replication of the calculation of the CPC, RMSE and CD (attention the latter, really heavy), FLAG_topo will allow to calculate the weighted clustering coefficient (the heaviest experiment), while FLAG_degree, FLAG_outdegree and FLAG_indegree will allow to calculate the weighted (in/out) degree.
For easiness of use, we leave the results file pre-calculated.
Once one has run all the Analysis notebooks, with all the 4 combination of data sets, it is possible to run the ```Plots.ipynb``` notebook, that replicate the analysis reported in the article i.e. the weight/weight-distance distribution, and the Cut Norm, CPC and RMSE distribution.


# DataLoading 
We provide several notebooks for replicating our ETL (Extraction Transform and Load) phase. All of them are located in the ```DataLoading/``` folder. The first notebook ```DataDownload.ipynb``` contains code for downloading the 4 datasets. Please pay attention to the Chicago's Taxi dataset: these data (referring to years 2018 and 2019) can be updated, so the pre-calculated data we presented in previous sections may not be consistent. Once you have downloaded the data (automatically put in the ```DataLoading/data/```) folder. Once done that, you can execute the four DataLoading notebooks (namely ```DataLoading_BikeCHI```, ```DataLoading_BikeNYC```, ```DataLoading_BikeNYC```, ```DataLoading_TaxiNYC```). These four dataset will transform the raw data into ```TrajectoryDataFrame``` and subsequently into ```FlowDataFrame``` (objects of [scikit-mobility](https://github.com/scikit-mobility/scikit-mobility)) library. The obtained FlowDataFrames are binded to the 64-tesselletion of the zone of analysis of the two cities (see our paper for more details) and saved into the ```Filtered/``` folder. After that, the FlowDataFrames are casted into 64x64 weighted adjacency matrices (a.k.a. mobility networks) and are placed into the 
```adj``` folder. As an example in the next image you can find a visualization of the process carryied out for the Bike of Manhattan dataset.
![data_schema](https://github.com/jonpappalord/GAN-flow/blob/main/dataload.png?raw=true)



