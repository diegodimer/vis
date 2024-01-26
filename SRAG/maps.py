#pylint: skip-file
import json
import geopandas as gpd
from data import DataReader
import altair as alt
import plotly.express as px
import pandas as pd
from pretrainingbias.pre_training_bias import PreTrainingBias

class MapRenderer:
    """ Class to render the maps"""
    
    @staticmethod
    def open_geojson():
        """ Open the geojson file"""
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

        data_reader = DataReader(year)
        df =  data_reader.get_dataframe()
        uf_normalized_data = data_reader.state_counts_normalized()
   
        data_geo = alt.Data(values=json_gdf['features'])
        uf_normalized_data['id'] = uf_normalized_data['SG_UF_NOT']

        pts = alt.selection_point( fields=['id'])

        bar = alt.Chart(uf_normalized_data).mark_bar().encode(
            x=alt.X('normalized:Q', title="Num of Cases per 100k"),
            y=alt.Y('id:N', title="UF"),
            tooltip=[alt.Text('normalized:Q')],
            color=alt.condition(pts, alt.ColorValue("steelblue"), alt.ColorValue("grey"))
        ).add_params(pts)

        map = alt.Chart(data_geo).mark_geoshape(stroke="white", strokeWidth=2).encode(
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
            type="mercator").add_params(pts).properties(width=1280, height=720)

        map = map.encode(
            opacity=alt.condition(pts, alt.value(1), alt.value(0.3))
        )

        dfs = data_reader.state_dataframes()
        _ptb = PreTrainingBias()


        ci_perm = {}
        ci_orig = {}
        kl_perm = {}
        kl_orig = {}
        ks_perm = {}
        ks_orig = {}
        for state in dfs:
            dfs[state]['CS_RACA_PRIVILEGED'] = dfs[state]['CS_RACA'].map({1: 1, 2: 0, 3: 0, 4:0, 5:0})
            ci_perm[state], ci_orig[state] = _ptb.get_class_imbalance_permutation_values(dfs[state], 'CS_RACA_PRIVILEGED', 100)
            kl_perm[state], kl_orig[state] = _ptb.get_kl_divergence_permutation_values(dfs[state], 'UTI', 'CS_RACA_PRIVILEGED', 1, 100)
            ks_perm[state], ks_orig[state] = _ptb.get_ks_permutation_values(dfs[state], 'UTI', 'CS_RACA_PRIVILEGED', 1, 100)

        ci_df_perm = pd.DataFrame(ci_perm)
        melted_ci_df_perm = pd.melt(ci_df_perm, value_vars=ci_df_perm.columns)
        melted_ci_df_perm.columns = ['id', 'Class Imbalance']
        melted_ci_df_perm = melted_ci_df_perm.sort_values("Class Imbalance").reset_index(drop=True)
        melted_ci_df_perm['index'] = melted_ci_df_perm.index

        ci_orig_df = pd.DataFrame(ci_orig, index=['Class Imbalance'])

        kl_df_perm = pd.DataFrame(kl_perm)
        melted_kl_df_perm = pd.melt(kl_df_perm, value_vars=kl_df_perm.columns)
        melted_kl_df_perm.columns = ['id', 'KL Divergence']
        melted_kl_df_perm = melted_kl_df_perm.sort_values("KL Divergence").reset_index(drop=True)
        melted_kl_df_perm['index'] = melted_kl_df_perm.index

        kl_orig_df = pd.DataFrame(kl_orig, index=['KL Divergence'])

        ks_df_perm = pd.DataFrame(ks_perm)
        melted_ks_df_perm = pd.melt(ks_df_perm, value_vars=ks_df_perm.columns)
        melted_ks_df_perm.columns = ['id', 'KS']
        melted_ks_df_perm = melted_ks_df_perm.sort_values("KS").reset_index(drop=True)
        melted_ks_df_perm['index'] = melted_ks_df_perm.index

        ks_orig_df = pd.DataFrame(ks_orig, index=['KS'])

        data_reader.df['CS_RACA_PRIVILEGED'] = data_reader.df['CS_RACA'].map({1: 1, 2: 0, 3: 0, 4:0, 5:0})

        ci_chart = MapRenderer.get_metric_dispersion(melted_ci_df_perm, ci_orig_df, 'Class Imbalance', pts)
        kl_chart = MapRenderer.get_metric_dispersion(melted_kl_df_perm, kl_orig_df, 'KL Divergence', pts)
        ks_chart = MapRenderer.get_metric_dispersion(melted_ks_df_perm, ks_orig_df, 'KS', pts)


        chart = alt.hconcat(bar, map)
        metrics_charts = alt.hconcat(ci_chart, kl_chart, ks_chart)
        chart = alt.vconcat(chart, metrics_charts, center=True, spacing=10, background='white', title=f"SRAG {year}" , bounds='full', autosize=alt.AutoSizeParams(type='fit', contains='padding'))

        race_kl = MapRenderer.get_map(data_geo, year, "KL", data_reader.kl_per_region('CS_RACA_PRIVILEGED', 1), 'viridis', 'CS_RACA')
        race_ks = MapRenderer.get_map(data_geo, year, "KS", data_reader.ks_per_region('CS_RACA_PRIVILEGED', 1), 'redyellowgreen', 'CS_RACA')
        race_ci = MapRenderer.get_map(data_geo, year, "CI", data_reader.ci_per_region('CS_RACA_PRIVILEGED'), 'plasma', 'CS_RACA')
        alt.hconcat(race_kl, race_ks, race_ci).resolve_scale(color='independent').save(f'resources/maps/race-{year}.html')

        sex_kl = MapRenderer.get_map(data_geo, year, "KL", data_reader.kl_per_region('CS_SEXO', 1), 'viridis', 'CS_SEXO')
        sex_ks = MapRenderer.get_map(data_geo, year, "KS", data_reader.ks_per_region('CS_SEXO', 1), 'redyellowgreen', 'CS_SEXO')
        sex_ci = MapRenderer.get_map(data_geo, year, "CI", data_reader.ci_per_region('CS_SEXO'), 'plasma', 'CS_SEXO')
        alt.hconcat(sex_kl, sex_ks, sex_ci).resolve_scale(color='independent').save(f'resources/maps/sex-{year}.html')

        chart.save(f'resources/geojson/{year}-2.html')

    @staticmethod
    def get_metric_dispersion(permutations, original, metric_name, pts):
        """ Compute and show metric chart """
        # permutations_kl, original_kl = _ptb.get_kl_divergence_permutation_values(df, target, col, "Privileged", permutations_kl)
        c = alt.Chart(permutations).mark_area(
                        color="lightblue",
                    interpolate='step-after',
                    line=True
                ).encode(
                    alt.X("index", title="Permutation Index"),
                    alt.Y(metric_name, title=f"{metric_name} Value"),
                    tooltip=[alt.Tooltip("id:O", title="UF"), alt.Tooltip(metric_name, title=f"{metric_name} Value")],
                    ).transform_filter(
                    pts  # Filter data based on the selection
                    ).add_params(pts)

        # original_line = alt.Chart(pd.DataFrame({metric_name: [original]})).mark_rule(color='red').encode( y=metric_name)
        original = original.reset_index().melt('index', var_name='column', value_name=metric_name)
        original['index'] = original.index
        original['id'] = original['column']
        # Create a rule for each column
        original_line = alt.Chart(original).mark_rule().encode(
            y=metric_name,
            color='id:N',
            tooltip=[alt.Tooltip("id:O", title="UF"), alt.Tooltip(metric_name, title=f"{metric_name} Value")],
            size=alt.value(2)
        ).transform_filter(pts).add_params(pts)
        return c + original_line

    @staticmethod
    def get_map(data_geo, year, metric_name, metric_df, color, attribute):
        map = alt.Chart(data_geo).mark_geoshape(stroke="white", strokeWidth=2).encode(
            color=alt.Color(
                f"{metric_name}:Q",
                scale=alt.Scale(scheme=color),
                legend=alt.Legend(title=f"{metric_name} Value", padding=0, orient='bottom', gradientLength=800),
            ),
            tooltip=[alt.Tooltip("id:O", title="UF"), alt.Tooltip(f"{metric_name}:Q", title=f"{metric_name} value")],
        ).transform_lookup(
            lookup="id",
            from_=alt.LookupData(
                metric_df, "id", [metric_name]
            ),
        ).project(
            type="mercator").properties(width=1280, height=720)

        map.save(f"resources/geojson/{metric_name}-{year}-{attribute}-metrics.png")

        return map
