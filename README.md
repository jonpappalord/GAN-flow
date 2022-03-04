# MoGAN - Generating mobility networs with GANs
---
## Table of contents
---
1. [Citing](#citing)
2. [Abstract](#abstract)
3. [Data Availability](#data-availability)
4. [Structure of the repository](#structure-of-the-repository)
5. [Analysis](#analysis)
6. [DataLoading](#dataloading)


# Citing
In this repository you can find the code for running MoGAN model and to replicate the analysis conducted in our paper.
If you use the code in this repository, please cite our paper:

*Mauro, G., Luca, M., Longa, A., Lepri, B., & Pappalardo, L. (2022). Generating Synthetic Mobility Networks with Generative Adversarial Networks. arXiv preprint arXiv:2202.11028.*

```
@article{mauro2022generating,
  title={Generating Synthetic Mobility Networks with Generative Adversarial Networks},
  author={Mauro, Giovanni and Luca, Massimiliano and Longa, Antonio and Lepri, Bruno and Pappalardo, Luca},
  journal={arXiv preprint arXiv:2202.11028},
  year={2022}
}
```

# Abstract
The increasingly crucial role of human displacements in complex societal phenomena, such as traffic congestion, segregation, and the diffusion of epidemics, is attracting the interest of scientists from several disciplines.
Here, we address mobility network generation, i.e., generating a city's entire mobility network, a weighted directed graph in which nodes are geographic locations and weighted edges represent people's movements between those locations, thus describing the entire mobility set flows within a city.
Our solution is MoGAN, a model based on Generative Adversarial Networks (GANs) to generate realistic mobility networks.
We conduct extensive experiments on public datasets of bike and taxi rides to show that MoGAN outperforms the classical Gravity and Radiation models regarding the realism of the generated networks.
Our model can be used for data augmentation and performing simulations and what-if analysis.
---
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

---
# Structure of the repository
In the **main** level of the repo you can find:
- ```Gravity.py``` and ``` Radiation.py```
    - These codes execute the two baseline models
- ```MoGAN.py```
    - This script execute the actual MoGAN model
- ```AnalysisGravity.ipynb ```, ```AnalysisRadiation.ipynb``` and ```AnalysisMoGAN.ipynb``` 
    - They perform the experimental phases for the three models
- ```Plots.ipynb``` 
    - Produces the plots of the experimental phase
- ```utils.py```
    - Some utilities function for the plotting phase 

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

    
---
# Analysis 
As reported in the article, we perform a massive experimental phase, implementing a tailored approach for evaluating the realism of the synthetic networks. For easiness of usage, we created three different Jupyter Notebook ( ```AnalysisMoGAN ```,  ``` AnalysisGravity``` and  ```AnalysisRadiation ```), one for each model, for creating, per each model, the distributions of Mixed, Syntetic and Test Set, over the four datasets (see our article for more details). If you want to run these experiments, execute all the cells of the notebooks. Please note that the second cell of the notebook contain two parameter, one specifying the city and one the mean of transport, for selecting the dataset (the syntax is the same as above i.e. CHI/NYC and Bike/Taxi). The results of the Analysis notebook will be stored in the folder of the desired dataset under the subfolder ```experiments/```. For easiness of use, we leave the results file pre-calculated.
Once one have runned all the Analysis notebooks, with all the 4 combination of data sets, it is possible to run the ```Plots.ipynb``` notebook, that replicate the analysis reported in the article i.e. the weight/weight-distance distribution, and the Cut Norm, CPC and RMSE distribution.

---
# DataLoading 
We provide several notebooks for replicating our ETL (Extraction Transform and Load) phase. All of them are located in the ```DataLoading/``` folder. The first notebook ```DataDownload.ipynb``` contains code for downloading the 4 datasets. Please pay attention to the Chicago's Taxi dataset: these data (referring to years 2018 and 2019) can be updated, so the pre-calculated data we presented in previous sections may not be consistent. Once you have downloaded the data (automatically put in the ```DataLoading/data/```) folder. Once done that, you can execute the four DataLoading notebooks (namely ```DataLoading_BikeCHI```, ```DataLoading_BikeNYC```, ```DataLoading_BikeNYC```, ```DataLoading_TaxiNYC```). These four dataset will transform the raw data into ```TrajectoryDataFrame``` and subsequently into ```FlowDataFrame``` (objects of [scikit-mobility](https://github.com/scikit-mobility/scikit-mobility)) library. The obtained FlowDataFrames are binded to the 64-tesselletion of the zone of analysis of the two cities (see our paper for more details). After that, the FlowDataFrame are casted into 64x64 weighted adjacency matrices (a.k.a. mobility networks) and are placed into the 
```adj``` folder. As an example in the next image you can find a visualization of the process carryied out for the Bike of Manhattan dataset.
![data_schema](https://github.com/jonpappalord/GAN-flow/blob/main/dataload.png?raw=true)



