#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 12:27:42 2021

@author: giovanni
"""


import skmob
from skmob.preprocessing import filtering, detection
import pandas as pd
import geopandas as gpd


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














skmob.utils.constants.NY_FLOWS_2011
