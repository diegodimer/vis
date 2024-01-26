""" This file is used to clean the data from the csv file and generate a processed csv file"""""
import os
import pandas as pd

from pretrainingbias.pre_training_bias import PreTrainingBias

class DataReader:
    """ Class to read the data from the csv file and generate a processed csv file"""
    year = ""
    df = None

    def __init__(self, year):
        """ Initialize the data reader"""
        self.year = year
        if year == '2021':
            self.csv_file = "resources/datasets/INFLUD21-01-05-2023.csv"
            self.target_csv_file = "resources/datasets/PROCESSED_INFLUD21-01-05-2023.csv"
            self.df = self.pre_process_srag()
        else:
            self.csv_file = "resources/datasets/INFLUD23-16-10-2023.csv"
            self.target_csv_file = "resources/datasets/PROCESSED_INFLUD23-16-10-2023.csv"
            self.df = self.pre_process_srag()

    def get_dataframe(self) -> pd.DataFrame:
        """ Returns the dataframe"""
        return self.df

    def pre_process_srag(self):
        """ Preprocess the SRAG data from 2021 """
        if os.path.isfile(self.target_csv_file):
            return pd.read_csv(self.target_csv_file)

        with open("resources/datasets/columns.txt", 'r', encoding='utf-8') as f:
            columns = [line.rstrip() for line in f]

        df = pd.read_csv(self.csv_file, sep=';', quotechar='"', encoding='utf-8', low_memory=False)
        self.df = df.filter(items=columns)
        self.beautify_dataframe()
        self.df.to_csv(self.target_csv_file)
        return self.df

    def beautify_dataframe(self):
        """ Beautify the dataframe """
        self.df.dropna(subset = ['UTI', "CS_RACA", "CS_SEXO", "VACINA_COV"], inplace=True)
        self.df.fillna(9, inplace=True)
        self.df.drop(self.df.loc[self.df['UTI']==9].index, inplace=True)
        self.df.drop(self.df.loc[self.df['CS_SEXO']=="I"].index, inplace=True)
        self.df.drop(self.df.loc[self.df['CS_RACA']==9].index, inplace=True)
        self.df.drop(self.df.loc[self.df['DT_NASC']==9].index, inplace=True)
        self.df.drop(self.df.loc[self.df['DT_SIN_PRI']==9].index, inplace=True)
        self.df.drop(self.df.loc[self.df['VACINA_COV']==9].index, inplace=True)
        self.df.dropna(inplace=True)

        self.df['DT_SIN_PRI'] = pd.to_datetime(self.df['DT_SIN_PRI'], format='%d/%m/%Y')
        self.df['DT_SIN_PRI'] = self.df['DT_SIN_PRI'].astype(int)
        self.df['DT_SIN_PRI'] = self.df['DT_SIN_PRI'] / 10**9
        self.df['DT_NASC'] = pd.to_datetime(self.df['DT_NASC'], format='%d/%m/%Y')
        self.df['DT_NASC'] = self.df['DT_NASC'].astype(int)
        self.df['DT_NASC'] = self.df['DT_NASC'] / 10**9
        self.df['CS_SEXO'] = self.df['CS_SEXO'].map({'F': 0, 'M': 1})
        self.df['UTI'] = self.df['UTI'].map({1: 1, 2: 0})
        self.df['VACINA_COV'] = self.df['VACINA_COV'].map({1: 1, 2: 0})

    def state_counts(self):
        """ Returns a dataframe with the number of cases per state """
        return self.df.groupby('SG_UF_NOT').size().reset_index(name='counts')

    def state_data(self) -> dict:
        """ Returns a dictionary with the dataframes for each state """
        dfs = {}
        for state in self.df['SG_UF_NOT'].unique():
            new_df = self.df.loc[self.df['SG_UF_NOT'] == state].copy()

            dfs[state] = new_df
        return dfs

    def region_data(self) -> dict:
        """ Returns a dictionary with the dataframes for each region """
        regions = {
            'sul': ['PR', 'SC', 'RS'],
            'sudeste': ['SP', 'RJ', 'MG', 'ES'],
            'centro_oeste': ['MS', 'MT', 'GO', 'DF'],
            'nordeste': ['MA', 'PI', 'CE', 'RN', 'PE', 'PB', 'SE', 'AL', 'BA'],
            'norte': ['AC', 'AP', 'AM', 'PA', 'RO', 'RR', 'TO']
        }
        dfs = {}
        for region, states in regions.items():
            new_df = self.df.loc[self.df['SG_UF_NOT'].isin(states)].copy()
            dfs[region] = new_df
        return dfs

    def state_counts_normalized(self) -> pd.DataFrame:
        """ Returns a dataframe with the number of cases per state normalized by the population"""
        population = pd.read_csv("resources/datasets/IBGE2022.csv", sep=';', quotechar='"', encoding='utf-8')
        new_df = self.df.groupby('SG_UF_NOT').size().reset_index(name='total').copy()
        new_df['population'] = new_df['SG_UF_NOT'].map(population.set_index('UF')['POPULACAO'])
        new_df['normalized'] = new_df['total']/new_df['population'] * 100000
        return new_df

    def kl_divergence_per_state(self, attribute, privileged_group=1) -> pd.DataFrame:
        """ Returns a dictionary with the KL divergence for each state """
        dfs = {}
        ptb = PreTrainingBias()
        for state in self.df['SG_UF_NOT'].unique():
            new_df = self.df.loc[self.df['SG_UF_NOT'] == state].copy()
            dfs[state] = ptb.kl_divergence(new_df, 'VACINA_COV', attribute, privileged_group)
        df = pd.DataFrame(dfs, index=['KL'])
        df = pd.melt(df, value_vars=df.columns)
        df.columns = ['id', 'KL']
        return df

    def ks_per_state(self, attribute, privileged_group=1) -> pd.DataFrame:
        """ Returns a dictionary with the KS for each state """
        dfs = {}
        ptb = PreTrainingBias()
        for state in self.df['SG_UF_NOT'].unique():
            new_df = self.df.loc[self.df['SG_UF_NOT'] == state].copy()
            dfs[state] = ptb.ks(new_df, 'VACINA_COV', attribute, privileged_group)
        df = pd.DataFrame(dfs, index=['KS'])
        df = pd.melt(df, value_vars=df.columns)
        df.columns = ['id', 'KS']
        return df

    def ci_per_state(self, attribute) -> pd.DataFrame:
        """ Returns a dictionary with the class imbalance for each state """
        dfs = {}
        ptb = PreTrainingBias()
        for state in self.df['SG_UF_NOT'].unique():
            new_df = self.df.loc[self.df['SG_UF_NOT'] == state].copy()
            dfs[state] = ptb.class_imbalance(new_df, attribute)
        df = pd.DataFrame(dfs, index=['CI'])
        df = pd.melt(df, value_vars=df.columns)
        df.columns = ['id', 'CI']
        return df

    def ci_per_region(self, attribute) -> pd.DataFrame:
        """ Returns a dictionary with the class imbalance for each region"""
        regions = {
            'sul': ['PR', 'SC', 'RS'],
            'sudeste': ['SP', 'RJ', 'MG', 'ES'],
            'centro_oeste': ['MS', 'MT', 'GO', 'DF'],
            'nordeste': ['MA', 'PI', 'CE', 'RN', 'PE', 'PB', 'SE', 'AL', 'BA'],
            'norte': ['AC', 'AP', 'AM', 'PA', 'RO', 'RR', 'TO']
        }
        dfs = {}
        ptb = PreTrainingBias()
        for region, states in regions.items():
            new_df = self.df.loc[self.df['SG_UF_NOT'].isin(regions[region])].copy()
            region_ci = ptb.class_imbalance(new_df, attribute)
            for state in states:
                dfs[state] = region_ci
        df = pd.DataFrame(dfs, index=['CI'])
        df = pd.melt(df, value_vars=df.columns)
        df.columns = ['id', 'CI']
        return df

    def ks_per_region(self, attribute, privileged_group) -> pd.DataFrame:
        """ Returns a dictionary with the KS for each region"""
        regions = {
            'sul': ['PR', 'SC', 'RS'],
            'sudeste': ['SP', 'RJ', 'MG', 'ES'],
            'centro_oeste': ['MS', 'MT', 'GO', 'DF'],
            'nordeste': ['MA', 'PI', 'CE', 'RN', 'PE', 'PB', 'SE', 'AL', 'BA'],
            'norte': ['AC', 'AP', 'AM', 'PA', 'RO', 'RR', 'TO']
        }
        dfs = {}
        ptb = PreTrainingBias()
        for region, states in regions.items():
            new_df = self.df.loc[self.df['SG_UF_NOT'].isin(regions[region])].copy()
            region_ks = ptb.ks(new_df, 'VACINA_COV', attribute, privileged_group)
            for state in states:
                dfs[state] = region_ks
        df = pd.DataFrame(dfs, index=['KS'])
        df = pd.melt(df, value_vars=df.columns)
        df.columns = ['id', 'KS']
        return df

    def kl_per_region(self, attribute, privileged_group) -> pd.DataFrame:
        """ Returns a dictionary with the KL divergence for each region"""
        regions = {
            'sul': ['PR', 'SC', 'RS'],
            'sudeste': ['SP', 'RJ', 'MG', 'ES'],
            'centro_oeste': ['MS', 'MT', 'GO', 'DF'],
            'nordeste': ['MA', 'PI', 'CE', 'RN', 'PE', 'PB', 'SE', 'AL', 'BA'],
            'norte': ['AC', 'AP', 'AM', 'PA', 'RO', 'RR', 'TO']
        }
        dfs = {}
        ptb = PreTrainingBias()
        for region, states in regions.items():
            new_df = self.df.loc[self.df['SG_UF_NOT'].isin(regions[region])].copy()
            region_kl = ptb.kl_divergence(new_df, 'VACINA_COV', attribute, privileged_group)
            for state in states:
                dfs[state] = region_kl
        df = pd.DataFrame(dfs, index=['KL'])
        df = pd.melt(df, value_vars=df.columns)
        df.columns = ['id', 'KL']
        return df

    def state_dataframes(self) -> pd.DataFrame:
        """ Returns a dictionary with the dataframes for each state """
        dfs = {}
        for state in self.df['SG_UF_NOT'].unique():
            dfs[state] = self.df.loc[self.df['SG_UF_NOT'] == state].copy()
        return dfs
