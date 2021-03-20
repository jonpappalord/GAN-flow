#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 11:40:42 2021

@author: giovanni
"""

from shapely.geometry import Point, Polygon
from shapely import wkt
import skmob
from skmob.preprocessing import filtering, detection
from skmob.utils.plot import plot_gdf
from skmob.tessellation import tilers
import pandas as pd
import geopandas as gpd
import webbrowser

import networkx as nx

pd.set_option('display.max_columns', 500)

def filter_tessellation_land(tessellation, shape_file_land):    
    tiles_in_land = gpd.sjoin(tessellation, shape_file_land, how='left', op='intersects')
    tiles_in_land = tiles_in_land.groupby(['tile_ID'],sort=False,as_index=False).first()
    #land = tiles_in_land.dropna().drop(["index_right","boro_code","boro_name","shape_area","shape_leng"],axis=1)
    #water = tiles_in_land[tiles_in_land['index_right'].isnull()].drop(["index_right","boro_code","boro_name","shape_area","shape_leng"],axis=1)    
    land = tiles_in_land.dropna()[['tile_ID', 'geometry']]
    water = tiles_in_land[tiles_in_land['index_right'].isnull()][['tile_ID', 'geometry']]     
    crs = {'init': 'epsg:4326'}
    land = gpd.GeoDataFrame(land, crs=crs, geometry='geometry')
    water = gpd.GeoDataFrame(water, crs=crs, geometry='geometry')     
    return {"land":land, "water":water}


def wkt_loads(x):
    try:
        return wkt.loads(x)
    except Exception:
        return None


meters = 500



### NYC ####

tessellation = tilers.tiler.get("squared", meters=meters, base_shape="New York City, New York")
shape_file_land = gpd.read_file("NYC_shapeNoWaterArea.geojson")


res_inter = filter_tessellation_land(tessellation, shape_file_land)
tess_nyc = res_inter['land']

m_t = plot_gdf(res_inter['land'])
m_t.save("NYC.html")
webbrowser.open("NYC.html")

########################


df = pd.read_csv('yellow_tripdata_2020-01.csv')

flusso = df.groupby(["PULocationID", "DOLocationID"]).size().reset_index(name="flow")
flusso = flusso.rename(columns={'PULocationID': 'origin', 'DOLocationID':'destination'})

flusso = flusso[flusso.origin <= 263]
flusso = flusso[flusso.destination <= 263]


fdf = skmob.FlowDataFrame(data = flusso,tessellation=tessellation, tile_id='tile_ID', origin ='origin', destination = 'destination', flow = 'flow')

# f = fdf.to_matrix()




#m = fdf.plot_tessellation()
m = fdf.plot_flows(flow_color='red')
m.save("thursday.html")
webbrowser.open("thursday.html")


############## Try changes of ID ############################

tess_nyc['lat'] = tess_nyc['geometry'].centroid.y
tess_nyc['lon'] = tess_nyc['geometry'].centroid.x

tess_nyc = tess_nyc.sort_values(["lat", "lon"], ascending = (False, True))
tess_nyc = tess_nyc.reset_index(drop = True)

hash_table = pd.Series(tess_nyc.index, index = tess_nyc.tile_ID).to_dict()

flusso = df.groupby(["PULocationID", "DOLocationID"]).size().reset_index(name="flow")
flusso = flusso.rename(columns={'PULocationID': 'origin', 'DOLocationID':'destination'})

#In flusso there are Ids not present in the 




#already ordered, first by longitude and then by latitude


###################
###################
###################
###################
###################
###################
###################
###################
###################
###################
###################
###################
###################
###################
###################
###################
###################
###################

### PORTO ####

tess_por = tilers.tiler.get("squared", meters=meters, base_shape="Porto, Área Metropolitana do Porto, North, Portugal")


df_por = pd.read_csv('porto_trajectories_all.csv')
df_por = df_por.drop_duplicates()


mask = df_por.trajectory_id.duplicated(keep=False)
prova = df_por[mask]
prova = df_por.drop_duplicates(subset=['trajectory_id'])
df_por = prova



df_por['geometry_source'] = df_por.source_point.apply(wkt_loads)
df_por['geometry_target'] = df_por.target_point.apply(wkt_loads)



gdf_1 = gpd.GeoDataFrame(df_por, crs="EPSG:4326", geometry=df_por['geometry_source'])
gdf_2 = gpd.GeoDataFrame(df_por, crs="EPSG:4326", geometry=df_por['geometry_target'])




res_merge_source = gpd.sjoin(tess_por, gdf_1, how='right', op='contains')
res_merge_source.dropna(inplace=True)
res_merge_source.reset_index(drop=True, inplace=True)



res_merge_dest = gpd.sjoin(tess_por, gdf_2, how='right', op='contains')
res_merge_dest.dropna(inplace=True)
res_merge_dest.reset_index(drop=True,inplace=True)



fdf = res_merge_dest.merge(res_merge_source, how='inner', on='trajectory_id')


fdf = fdf[['tile_ID_x','tile_ID_y']]

flusso = fdf.groupby(["tile_ID_x", "tile_ID_y"]).size().reset_index(name="flow")
flusso = flusso.rename(columns={'tile_ID_x': 'origin', 'tile_ID_y':'destination'})



tess_por['lat'] = tess_por['geometry'].centroid.y
tess_por['lon'] = tess_por['geometry'].centroid.x
tess_por = tess_por.sort_values(["lat", "lon"], ascending = (False, True))
tess_por = tess_por.reset_index(drop = True)
hash_table = pd.Series(tess_por.index, index = tess_por.tile_ID).to_dict()
sorted_dict = {k: hash_table[k] for k in sorted(hash_table)}


flusso['origin'] = flusso['origin'].map(sorted_dict)
flusso['destination'] = flusso['destination'].map(sorted_dict)


fdf_porto = skmob.FlowDataFrame(data = flusso,tessellation=tess_por, tile_id='tile_ID', origin ='origin', destination = 'destination', flow = 'flow')


#fdf_porto = fdf_porto[fdf_porto['flow'] > 10]  
m = fdf_porto.plot_flows(flow_color='red')
m.save("final.html")
webbrowser.open("final.html")

###

edgeList = fdf_porto.values.tolist()
G = nx.DiGraph()

for i in range(len(edgeList)):
    G.add_edge(edgeList[i][0], edgeList[i][1], weight=int(edgeList[i][2]))

G.edges()


A = nx.adjacency_matrix(G, weight='flow')



OD_porto = fdf_porto.to_matrix()





###################
###################
###################
###################
###################
###################
###################
###################
###################
###################
###################
###################
###################
###################
###################
###################
###################
###################


### NYC ####



tessellation = tilers.tiler.get("squared", meters=meters, base_shape="New York City, New York")
shape_file_land = gpd.read_file("NYC_shapeNoWaterArea.geojson")


res_inter = filter_tessellation_land(tessellation, shape_file_land)
tess_nyc = res_inter['land']
tess_nyc = tess_nyc.reset_index(drop = True)



tess_nyc = tessellation


df_nyc = pd.read_csv('201802-citibike-tripdata.csv.zip')
#df_nyc = df_nyc.drop_duplicates()

df_nyc['id'] = df_nyc.apply(lambda x: hash(tuple(x)), axis = 1)



df_nyc['geometry_source'] = [Point(xy) for xy in zip(df_nyc['start station longitude'],df_nyc['start station latitude'])]
df_nyc['geometry_target'] = [Point(xy) for xy in zip(df_nyc['end station longitude'],df_nyc['end station latitude'])]



columns = ['tripduration', 'start station name', 'start station latitude', 'start station longitude', 'end station name', 'end station latitude', 'end station longitude', 'usertype', 'birth year', 'gender', 'starttime', 'stoptime', 'bikeid']
df_nyc = df_nyc.drop(columns, axis = 1) 




gdf_1 = gpd.GeoDataFrame(df_nyc, crs="EPSG:4326", geometry=df_nyc['geometry_source']).drop(['geometry_target', 'geometry_source'],axis =1)
gdf_2 = gpd.GeoDataFrame(df_nyc, crs="EPSG:4326", geometry=df_nyc['geometry_target']).drop(['geometry_target', 'geometry_source'],axis =1)



res_merge_source = gpd.sjoin(tess_nyc, gdf_1, how='right', op='contains')
res_merge_source.dropna(inplace=True)
res_merge_source.reset_index(drop=True, inplace=True)



res_merge_dest = gpd.sjoin(tess_nyc, gdf_2, how='right', op='contains')
res_merge_dest.dropna(inplace=True)
res_merge_dest.reset_index(drop=True,inplace=True)





fdf = res_merge_dest.merge(res_merge_source, how='inner', on=['id'])
fdf = fdf[['tile_ID_x','tile_ID_y']]


flusso = fdf.groupby(["tile_ID_x", "tile_ID_y"]).size().reset_index(name="flow")
flusso = flusso.rename(columns={'tile_ID_x': 'origin', 'tile_ID_y':'destination'})




tess_nyc['lat'] = tess_nyc['geometry'].centroid.y
tess_nyc['lon'] = tess_nyc['geometry'].centroid.x
tess_nyc = tess_nyc.sort_values(["lat", "lon"], ascending = (False, True))
tess_nyc = tess_nyc.reset_index(drop = True)

hash_table = pd.Series(tess_nyc.index, index = tess_nyc.tile_ID).to_dict()
sorted_dict = {k: hash_table[k] for k in sorted(hash_table)}


tess_nyc['sorted_tile_ID'] = tess_nyc['tile_ID'].map(sorted_dict)


m_t = plot_gdf(tessellation, popup_features=['tile_ID'])
m_t.save("NYC.html")
webbrowser.open("NYC.html")





a = set(sorted(flusso.destination.unique()))
b = set(sorted(flusso.origin.unique()))
c = a.union(b)

d = set(sorted(tess_nyc.tile_ID))



flusso['origin'] = flusso['origin'].map(sorted_dict)
flusso['destination'] = flusso['destination'].map(sorted_dict)





fdf_nyc = skmob.FlowDataFrame(data = flusso,tessellation=tess_nyc, tile_id='tile_ID', origin ='origin', destination = 'destination', flow = 'flow')

OD_nyc = fdf_nyc.to_matrix()
