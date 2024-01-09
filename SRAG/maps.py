#pylint: skip-file

import json
import geopandas as gpd
from SRAG.data import DataReader
import altair as alt

class MapRenderer:

    @staticmethod
    def open_geojson():
        with open("resources/geojson/br_states.json") as json_data:
            d = json.load(json_data)
        return gpd.GeoDataFrame.from_features((d))
    
    @staticmethod
    def gen_map(geodata, color_column, title, tooltip, color_scheme='yelloworangered'):
        '''
        Generates a map with the given geodata and color_column
        '''
        return alt.Chart(geodata, title=title).mark_geoshape(
            stroke='darkgray'
        ).encode(
            alt.Color(color_column, 
                    type='quantitative', 
                    scale=alt.Scale(scheme=color_scheme),
                    title = "Num of Cases"),
            
            tooltip=tooltip
        ).properties(
            width=800,
            height=800
        )
    
    @staticmethod
    def get_srag_2021_choro():
        df_2021 = DataReader.state_counts(DataReader.get_srag_2021())
        gdf = MapRenderer.open_geojson()
        gdf = gdf.merge(df_2021, left_on='UF_05', right_on='SG_UF_NOT', how='inner')
        return MapRenderer.gen_map(gdf, 'counts', 'SRAG Cases in 2021', ['SG_UF_NOT', 'counts'])
    
    @staticmethod
    def get_srag_2023_choro():
        df_2023 = DataReader.state_counts(DataReader.get_srag_2023())
        gdf = MapRenderer.open_geojson()
        gdf = gdf.merge(df_2023, left_on='UF_05', right_on='SG_UF_NOT', how='inner')
        return MapRenderer.gen_map(gdf, 'counts', 'SRAG Cases in 2023', ['SG_UF_NOT', 'counts']).save('map.html')

    @staticmethod
    def topojson_version_2023():
        df_2023 = DataReader.state_counts(DataReader.get_srag_2023())
        african_countries = alt.topo_feature(
        "https://raw.githubusercontent.com/fititnt/gis-dataset-brasil/master/uf/topojson/uf.json",
        "uf",
        )
        return alt.Chart(african_countries).mark_geoshape(stroke="white", strokeWidth=2).encode(
            color=alt.Color(
                "counts:Q",
                scale=alt.Scale(scheme="yelloworangered"),
                legend=alt.Legend(title="Num of Cases"),
            ),
            tooltip=["id:O", "counts:Q"],
        ).transform_lookup(
            lookup="id",
            from_=alt.LookupData(
                df_2023, "SG_UF_NOT", ["counts"]
            ),
        ).properties(width=800, height=800)


    @staticmethod
    def topojson_version_2021():
        df_2023 = DataReader.state_counts(DataReader.get_srag_2023())
        african_countries = alt.topo_feature(
        "https://raw.githubusercontent.com/fititnt/gis-dataset-brasil/master/uf/topojson/uf.json",
        "uf",
        )
        return alt.Chart(african_countries).mark_geoshape(stroke="white", strokeWidth=2).encode(
            color=alt.Color(
                "counts:Q",
                scale=alt.Scale(scheme="yelloworangered"),
                legend=alt.Legend(title="Num of Cases"),
            ),
            tooltip= [alt.Tooltip("properties.uf:O", title="UF"), alt.Tooltip("counts:Q", title="Num of Cases")],
        ).transform_lookup(
            lookup="id",
            from_=alt.LookupData(
                df_2023, "SG_UF_NOT", ["counts"]
            ),
        ).properties(width=800, height=800)
