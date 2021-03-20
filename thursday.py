#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 12:27:42 2021

@author: giovanni
"""


import skmob
from skmob.preprocessing import filtering, detection
from skmob.utils.plot import plot_gdf
from skmob.tessellation import tilers
import pandas as pd
import geopandas as gpd
import webbrowser



url_tess = 'https://raw.githubusercontent.com/fmaletski/nyc-taxi-map/master/data/zones.geojson'
tessellation = gpd.read_file(url_tess).rename(columns={'OBJECTID': 'tile_ID'})
tessellation.tile_ID = pd.to_numeric(tessellation.tile_ID, errors='coerce')


df = pd.read_csv('yellow_tripdata_2020-01.csv')

flusso = df.groupby(["PULocationID", "DOLocationID"]).size().reset_index(name="flow")
flusso = flusso.rename(columns={'PULocationID': 'origin', 'DOLocationID':'destination'})

flusso = flusso[flusso.origin <= 263]
flusso = flusso[flusso.destination <= 263]

a = set(sorted(flusso.destination.unique()))
b = set(sorted(flusso.origin.unique()))
c = a.union(b)

d = set(sorted(tessellation.tile_ID))


fdf = skmob.FlowDataFrame(data = flusso,tessellation=tessellation, tile_id='tile_ID', origin ='origin', destination = 'destination', flow = 'flow')


#m = fdf.plot_tessellation()
m = fdf.plot_flows(flow_color='red')
m.save("thursday.html")

m_t = fdf.plot_tessellation()
m_t.save("tess.html")


#############################Bikes


df = pd.read_csv("201802-citibike-tripdata.csv.zip")


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




tessellation = tilers.tiler.get("squared", meters=500, base_shape="New York City, New York")
shape_file_land = gpd.read_file("NYC_shapeNoWaterArea.geojson")


res_inter = filter_tessellation_land(tessellation, shape_file_land)
m_t = plot_gdf(res_inter['land'])
m_t.save("NYC.html")
webbrowser.open("NYC.html")
