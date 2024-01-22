#pylint: skip-file
import json
import geopandas as gpd
from data import DataReader
import altair as alt
import plotly.express as px
import pandas as pd
from pretrainingbias.pre_training_bias import PreTrainingBias

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
    def make_html_maps(year):
        json_gdf = MapRenderer.open_geojson()
        if year == '2023':
            df = DataReader.get_srag_2023()
            uf_normalized_data = DataReader.state_counts_normalized(DataReader.get_srag_2023())
        else:
            df = DataReader.get_srag_2021()
            uf_normalized_data = DataReader.state_counts_normalized(DataReader.get_srag_2021())

        data_geo = alt.Data(values=json_gdf['features'])
        uf_normalized_data['id'] = uf_normalized_data['SG_UF_NOT']

        pts = alt.selection_point( fields=['id'])

        bar = alt.Chart(uf_normalized_data).mark_bar().encode(
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
                uf_normalized_data, "id", ["total", 'normalized']
            ),
        ).project(
            type="mercator").add_params(alt.selection_interval()).properties(width=1280, height=720)

        _ptb = PreTrainingBias()
        
        ci_perm, ci_orig = _ptb.get_class_imbalance_permutation_values(df, 'CS_SEXO', 100)
        ci_chart = MapRenderer.get_metric_dispersion(ci_perm, ci_orig, 'Class Imbalance', pts)
        
        kl_perm, kl_orig = _ptb.get_kl_divergence_permutation_values(df, 'UTI', 'CS_SEXO', 'M', 100)
        kl_chart = MapRenderer.get_metric_dispersion(kl_perm, kl_orig, 'KL Divergence', pts)
        
        ks_perm, ks_orig = _ptb.get_ks_permutation_values(df, 'UTI', 'CS_SEXO', 'M', 100)
        ks_chart = MapRenderer.get_metric_dispersion(ks_perm, ks_orig, 'KS', pts)
        
        chart = alt.hconcat(bar, map)
        metrics_charts = alt.hconcat(ci_chart, kl_chart, ks_chart)
        chart = alt.vconcat(chart, metrics_charts, center=True, spacing=10, background='white', title=f"SRAG {year}" , bounds='full', autosize=alt.AutoSizeParams(type='fit', contains='padding'))
        chart.save(f'resources/geojson/{year}.html')

    @staticmethod
    def get_metric_dispersion(permutations, original, metric_name, pts):
        """ Compute and show metric chart """
        # permutations_kl, original_kl = _ptb.get_kl_divergence_permutation_values(df, target, col, "Privileged", permutations_kl)
        df_permutations = pd.DataFrame(permutations, columns=[metric_name])
        df_permutations = df_permutations.sort_values(metric_name).reset_index(drop=True)
        df_permutations['index'] = df_permutations.index
        c = alt.Chart(df_permutations).mark_area(
                        color="lightblue",
                    interpolate='step-after',
                    line=True
                ).encode(
                    alt.X("index", title="Permutation Index"),
                    alt.Y(metric_name, title=f"{metric_name} Value"))

        original_line = alt.Chart(pd.DataFrame({metric_name: [original]})).mark_rule(color='red').encode( y=metric_name)
        # st.altair_chart(c + original_kl_line, use_container_width=True)
        return c + original_line
        
# MapRenderer.make_html_maps(DataReader.state_counts_normalized(DataReader.get_srag_2021()), '2021')
MapRenderer.make_html_maps('2023')