import skmob
from skmob.utils import utils, constants
from skmob.tessellation import tilers
from skmob.utils.plot import plot_gdf

import numpy as np
import pandas as pd
import geopandas as gpd
from skmob.models.gravity import Gravity


from sklearn.model_selection import train_test_split
from random import sample
import pickle
from scipy.spatial import distance
import sys

import time

from matplotlib import pyplot as plt
plt.rcParams["figure.figsize"] = (12, 9) # (w, h)


#First arg CHI or NYC
#Second arg Bike or Taxi
city = sys.argv[1]
transp = sys.argv[2]


#######------UTILS------#########
#######------#######------########
#######------#######------########
def toFdf(adj, tess, d):
    l = []
    for i in range(0,len(adj)):
        for j in range(0,len(adj)):
            if(adj[i,j]!=0):
                l.append([d[i],d[j], adj[i,j]])
    data = pd.DataFrame(l)
    return(data)


def toAdj(fdf, tess):
    names = list(tess.tile_ID.values)
    adj = pd.DataFrame(columns=names, index = names)
    for i, row in enumerate(fdf.itertuples(), 1):
         adj.at[str(row.origin), str(row.destination)] = row.flow
    return adj


def filter_tessellation_land(tessellation, shape_file_land):
    tiles_in_land = gpd.sjoin(tessellation, shape_file_land, how='left', op='intersects')
    tiles_in_land = tiles_in_land.groupby(['tile_ID'],sort=False,as_index=False).first()
    land = tiles_in_land.dropna()[['tile_ID', 'geometry']]
    water = tiles_in_land[tiles_in_land['index_right'].isnull()][['tile_ID', 'geometry']]
    crs = {'init': 'epsg:4326'}
    land = gpd.GeoDataFrame(land, crs=crs, geometry='geometry')
    water = gpd.GeoDataFrame(water, crs=crs, geometry='geometry')
    return {"land":land, "water":water}



#######------TESSELLETION------#########
#######------#######------########
#######------#######------########

if city == "CHI":
    shape_file_land = gpd.read_file("./DataLoading/Boundaries - Community Areas (current).geojson")
    shape_file_land_filtered = shape_file_land[shape_file_land['community'].
                                            isin(['LOOP', 'NEAR SOUTH SIDE', 'NEAR NORTH SIDE', 'NEAR WEST SIDE', 'LOWER WEST SIDE', 'WEST TOWN'])]
    meters = 1405
    tess_chi = tilers.tiler.get("squared", meters=meters, base_shape=shape_file_land_filtered)
    tesselletion = tess_chi
    plot_gdf(tesselletion, style_func_args={'fillColor':'gray', 'color':'black', 'opacity': 0.2}, zoom = 9)

else:
    meters = 1840
    tesselletion = tilers.tiler.get("squared", meters=meters, base_shape="New York City")
    shape_file_land = gpd.read_file("./DataLoading/NYC_shapeNoWaterArea.geojson")

    shape_file_land_MAN = shape_file_land.iloc[[4]]

    res_inter = filter_tessellation_land(tesselletion, shape_file_land_MAN )
    tess_nyc = res_inter['land']
    tesselletion = tess_nyc
    plot_gdf(tesselletion, style_func_args={'fillColor':'gray', 'color':'black', 'opacity': 0.2}, zoom = 9)


#######------DATA------#########
#######------#######------########
#######------#######------########
with open("./"+ transp + city+ "/v_train.txt", "rb") as fp:   # Unpickling
    v_train = pickle.load(fp)
print(len(v_train))


with open("./"+ transp + city+ "/v_test.txt", "rb") as fp:   # Unpickling
    v_test = pickle.load(fp)
print(len(v_test))

#######------GRAVITY------#########
#######------#######------########
#######------#######------########
# # GRAVITY FITTED OVER A SAMPLE OF THE TRAINING SET #

a = list(enumerate(list(tesselletion.tile_ID.values)))
d = dict(a)
tiles = np.asarray(list(d.values())).astype(int)
zeros = np.zeros(tiles.size)

np.random.seed(0)
a = 0
l = []

start = time.time()

for n in sample(v_train, len(v_test)):

    if city == 'CHI':
        tesselletion = tess_chi
    else:
        tesselletion = tess_nyc

    gravity_singly_fitted = Gravity(gravity_type='singly constrained')

    lookup = dict(zip(tiles.astype(str), zeros))

    lookup_pick = dict(zip(tiles.astype(str), zeros))

    flow = toFdf(n, tesselletion, d)
    flow = flow[flow[0] != flow[1]]

    z = flow.groupby([0])[2].sum()
    for el in z.keys():
        lookup[el] += z[el]

    flow_pick = toFdf(n, tesselletion, d)
    z = flow_pick.groupby([0])[2].sum()
    for el in z.keys():
        lookup_pick[el] += z[el]

    tesselletion['tot_outflow'] = tesselletion['tile_ID'].map(lookup)
    tesselletion['relevance'] = tesselletion['tile_ID'].map(lookup_pick)
    tesselletion['relevance'] = tesselletion['relevance'] / (tesselletion['relevance'].max())
    tesselletion['relevance'] = tesselletion['relevance'] + 0.00001

    fdf = skmob.FlowDataFrame(data = flow,tessellation=tesselletion, tile_id='tile_ID', origin =0,
                              destination = 1, flow = 2)

    gravity_singly_fitted.fit(fdf, relevance_column='relevance')
    sc_fdf_fitted = gravity_singly_fitted.generate(tesselletion,
                tile_id_column='tile_ID',
                tot_outflows_column='tot_outflow',
                relevance_column= 'relevance',
                out_format='flows')
    l.append(sc_fdf_fitted)

    a+=1
    print(a)

end = time.time()

print("Elapsed time: " + str((end - start)/60) +" minutes")

print(gravity_singly_fitted.deterrence_func_args)

print(gravity_singly_fitted.destination_exp)
#######------FAKE SET------#########
#######------#######------########
#######------#######------########


fake_set = []
for fdf in l:
    adj = toAdj(fdf, tesselletion)
    adj = adj.fillna(0)
    arr = np.rint(adj.to_numpy())
    fake_set.append(arr)

with open("./"+ transp+city+ "/fake_set_gravity.txt", "wb") as fp:   # Unpickling
    pickle.dump(fake_set,fp)
