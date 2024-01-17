#pylint: skip-file

import json
import geopandas as gpd
from data import DataReader
import altair as alt
import plotly.express as px
import pandas as pd

class MapRenderer:

    @staticmethod
    def open_geojson():
        with open("resources/geojson/br_states.json") as json_data:
            d = json.load(json_data)
        return d
    
    @staticmethod
    def gen_map(geodata, color_column, gdf, color_scheme='yelloworangered'):
        '''
        Generates a map with the given geodata and color_column
        '''
        fig = px.choropleth(geodata, geojson=gdf, locations='estado', color=color_column,
                           color_continuous_scale="Viridis",
                           scope="south america",
                          )
        return fig


    @staticmethod
    def get_srag_2021_choro():
        df_2021 = DataReader.state_counts(DataReader.get_srag_2021())
        df_2021.rename(columns={'SG_UF_NOT': 'estado'}, inplace=True)
        gdf = MapRenderer.open_geojson()
        return MapRenderer.gen_map(df_2021, 'counts', gdf)
    
    
    @staticmethod
    def get_srag_2023_choro():
        df_2023 = DataReader.state_counts(DataReader.get_srag_2023())
        df_2023.rename(columns={'SG_UF_NOT': 'estado'}, inplace=True)
        gdf = MapRenderer.open_geojson()
        return MapRenderer.gen_map(df_2023, 'counts', gdf)
    
    

    @staticmethod
    def topojson_version_2023():
        df_2023 = DataReader.state_counts_normalized(DataReader.get_srag_2023())
        return MapRenderer.get_chart(df_2023)


    @staticmethod
    def topojson_version_2021():
        df_2021 = DataReader.state_counts_normalized(DataReader.get_srag_2021())
        return MapRenderer.get_chart(df_2021)

    @staticmethod
    def get_chart(df):
        topodf = alt.topo_feature(
            "https://raw.githubusercontent.com/fititnt/gis-dataset-brasil/master/uf/topojson/uf.json",
            "uf",
        )
        return alt.Chart(topodf).mark_geoshape(stroke="white", strokeWidth=2).encode(
            color=alt.Color(
                "normalized:Q",
                scale=alt.Scale(scheme="tealblues"),
                legend=alt.Legend(title="Num of Cases"),
            ),
            tooltip=[alt.Tooltip("properties.uf:O", title="UF"), alt.Tooltip("total:Q", title="Num of Cases"), alt.Tooltip("normalized:Q", title="Num of Cases per 100k")],
        ).transform_lookup(
            lookup="id",
            from_=alt.LookupData(
                df, "SG_UF_NOT", ["total", 'normalized']
            ),
        ).properties(width=1280, height=720).project(
            type="mercator", scale=1000, center=[-54, -15]).add_params(alt.selection_interval())
        
    @staticmethod
    def make_html_maps(data, year):
        json_gdf = MapRenderer.open_geojson()

        data_geo = alt.Data(values=json_gdf['features'])
        data['id'] = data['SG_UF_NOT']

        pts = alt.selection_point( fields=['id'])

        bar = alt.Chart(data).mark_bar().encode(
            x=alt.X('normalized:Q', title="Num of Cases per 100k"),
            y=alt.Y('id:N', title="UF"),
            tooltip=[alt.Text('normalized:Q')],
            color=alt.condition(pts, alt.ColorValue("steelblue"), alt.ColorValue("grey"))
        ).add_params(pts)
       
        map = alt.Chart(data_geo).transform_filter(pts).add_params(pts).mark_geoshape(stroke="white", strokeWidth=2).encode(
            color=alt.Color(
                "normalized:Q",
                scale=alt.Scale(scheme="tealblues"),
                legend=alt.Legend(title="Num of Cases"),
            ),
            tooltip=[alt.Tooltip("id:O", title="UF"), alt.Tooltip("total:Q", title="Num of Cases"), alt.Tooltip("normalized:Q", title="Num of Cases per 100k")],
        ).transform_lookup(
            lookup="id",
            from_=alt.LookupData(
                data, "id", ["total", 'normalized']
            ),
        ).project(
            type="mercator").add_params(alt.selection_interval()).properties(width=1280, height=720)

        chart = alt.hconcat(bar, map, center=True, spacing=10, background='white', padding=10, title=f"SRAG {year}" , bounds='full', autosize=alt.AutoSizeParams(type='fit', contains='padding'))
        
        chart.save(f'resources/geojson/{year}.html')