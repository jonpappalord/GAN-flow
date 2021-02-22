#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 12:52:21 2021

@author: giovanni
"""
import skmob
from skmob.preprocessing import filtering, detection
from skmob.utils.plot import plot_gdf
from skmob.tessellation import tilers
import pandas as pd
import geopandas as gpd
import webbrowser



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




meters = 500



### NYC ####

tessellation = tilers.tiler.get("squared", meters=meters, base_shape="New York City, New York")
shape_file_land = gpd.read_file("NYC_shapeNoWaterArea.geojson")


res_inter = filter_tessellation_land(tessellation, shape_file_land)
m_t = plot_gdf(res_inter['land'])
m_t.save("NYC.html")
webbrowser.open("NYC.html")



### BCN ###  vangdata.carto.com

tessellation = tilers.tiler.get("squared", meters=meters, base_shape="Barcelona, Spain")
shape_file_land = gpd.read_file("shapefiles_barcelona_distrito.geojson")

res_inter = filter_tessellation_land(tessellation, shape_file_land)

m_t = plot_gdf(res_inter['land'])
m_t.save("BCN.html")
webbrowser.open("BCN.html")



### MAD ### https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/madrid-districts.geojson

tessellation = tilers.tiler.get("squared", meters=meters, base_shape="Madrid, Spain")
url = "https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/madrid-districts.geojson"
shape_file_land = gpd.read_file(url)

res_inter = filter_tessellation_land(tessellation, shape_file_land)

m_t = plot_gdf(res_inter['land'])
m_t.save("MAD.html")
webbrowser.open("MAD.html")


### LON ###  Hole in the City. Makes sense?

tessellation = tilers.tiler.get("squared", meters=meters, base_shape="London, England")
shape_file_land = gpd.read_file("london_boroughs_proper.geojson")

res_inter = filter_tessellation_land(tessellation, shape_file_land)

m_t = plot_gdf(res_inter['land'])
m_t.save("LON.html")
webbrowser.open("LON.html")






### ITALY ###

url = "https://raw.githubusercontent.com/openpolis/geojson-italy/master/comuni.geojson"
shape_file_land = gpd.read_file(url)
################


### CZ ###
tessellation = tilers.tiler.get("squared", meters=meters, base_shape="Catanzaro, CZ, Italy")

m_t = plot_gdf(tessellation)
m_t.save("CZ.html")
webbrowser.open("CZ.html")

### Milan ###

tessellation = tilers.tiler.get("squared", meters=meters, base_shape="Milano, MI, Italy")

m_t = plot_gdf(tessellation)
m_t.save("MI.html")
webbrowser.open("MI.html")

### RM ###

tessellation = tilers.tiler.get("squared", meters=meters, base_shape="Roma, RM, Italy")

m_t = plot_gdf(tessellation)
m_t.save("RM.html")
webbrowser.open("RM.html")

### Pisa ###

tessellation = tilers.tiler.get("squared", meters=meters, base_shape="Pisa, PI, Italy")

m_t = plot_gdf(tessellation)
m_t.save("PI.html")
webbrowser.open("PI.html")

