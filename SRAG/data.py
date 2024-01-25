""" This file is used to clean the data from the csv file and generate a processed csv file"""""
import os
import pandas as pd

from pretrainingbias.pre_training_bias import PreTrainingBias

class DataReader:
    """ Class to read the data from the csv file and generate a processed csv file"""
    @staticmethod
    def get_srag_2021():
        """ Returns the SRAG data from 2021 """
        return DataReader.pre_process_srag_2021()

    @staticmethod
    def get_srag_2023():
        """ Returns the SRAG data from 2023 """
        return DataReader.pre_process_srag_2023()

    @staticmethod
    def pre_process_srag_2021():
        """ Preprocess the SRAG data from 2021 """
        if os.path.isfile("resources/datasets/PROCESSED_INFLUD21-01-05-2023.csv"):
            return pd.read_csv("resources/datasets/PROCESSED_INFLUD21-01-05-2023.csv")

        csv_file = "resources/datasets/INFLUD21-01-05-2023.csv"
        with open("resources/datasets/columns.txt", 'r', encoding='utf-8') as f:
            columns = [line.rstrip() for line in f]

        df = pd.read_csv(csv_file, sep=';', quotechar='"', encoding='utf-8', low_memory=False)
        df = df.filter(items=columns)
        DataReader.beautify_dataframe(df)
        df.to_csv("resources/datasets/PROCESSED_INFLUD21-01-05-2023.csv")
        return df

    @staticmethod
    def pre_process_srag_2023():
        """ Preprocess the SRAG data from 2023 """
        if os.path.isfile("resources/datasets/PROCESSED_INFLUD23-16-10-2023.csv"):
            return pd.read_csv("resources/datasets/PROCESSED_INFLUD23-16-10-2023.csv")

        csv_file = "resources/datasets/INFLUD23-16-10-2023.csv"
        with open("resources/datasets/columns.txt", 'r', encoding='utf-8') as f:
            columns = [line.rstrip() for line in f]

        df = pd.read_csv(csv_file, sep=';', quotechar='"', encoding='utf-8')
        df = df.filter(items=columns)
        DataReader.beautify_dataframe(df)
        df.to_csv("resources/datasets/PROCESSED_INFLUD23-16-10-2023.csv")
        return df

    @staticmethod
    def beautify_dataframe(df):
        """ Beautify the dataframe """
        df.dropna(subset = ['UTI', "CS_RACA", "CS_SEXO"], inplace=True)
        df.fillna(9, inplace=True)
        df.drop(df.loc[df['UTI']==9].index, inplace=True)
        df.drop(df.loc[df['CS_SEXO']=="I"].index, inplace=True)
        df.drop(df.loc[df['CS_RACA']==9].index, inplace=True)
        df.drop(df.loc[df['DT_NASC']==9].index, inplace=True)
        df.drop(df.loc[df['DT_SIN_PRI']==9].index, inplace=True)
        df.dropna(inplace=True)
        
        df['DT_SIN_PRI'] = pd.to_datetime(df['DT_SIN_PRI'], format='%d/%m/%Y')
        df['DT_SIN_PRI'] = df['DT_SIN_PRI'].astype(int) 
        df['DT_SIN_PRI'] = df['DT_SIN_PRI'] / 10**9
        df['DT_NASC'] = pd.to_datetime(df['DT_NASC'], format='%d/%m/%Y')
        df['DT_NASC'] = df['DT_NASC'].astype(int) 
        df['DT_NASC'] = df['DT_NASC'] / 10**9
        df['CS_SEXO'] = df['CS_SEXO'].map({'F': 0, 'M': 1})
        df['UTI'] = df['UTI'].map({1: 1, 2: 0})

    @staticmethod
    def state_counts(df):
        """ Returns a dataframe with the number of cases per state """
        return df.groupby('SG_UF_NOT').size().reset_index(name='counts')

    @staticmethod
    def state_data(df: pd.DataFrame) -> dict:
        """ Returns a dictionary with the dataframes for each state """
        dfs = {}
        for state in df['SG_UF_NOT'].unique():
            new_df = df.loc[df['SG_UF_NOT'] == state].copy()

            dfs[state] = new_df
        return dfs
    
    @staticmethod
    def state_counts_normalized(df):
        """ Returns a dataframe with the number of cases per state normalized by the population"""
        population = pd.read_csv("resources/datasets/IBGE2022.csv", sep=';', quotechar='"', encoding='utf-8')
        df = df.groupby('SG_UF_NOT').size().reset_index(name='total')
        df['population'] = df['SG_UF_NOT'].map(population.set_index('UF')['POPULACAO'])
        df['normalized'] = df['total']/df['population'] * 100000
        return df

    @staticmethod
    def kl_divergence_per_state(df):
        """ Returns a dictionary with the KL divergence for each state """
        dfs = {}
        ptb = PreTrainingBias()
        for state in df['SG_UF_NOT'].unique():
            new_df = df.loc[df['SG_UF_NOT'] == state].copy()
            dfs[state] = ptb.kl_divergence(new_df, 'UTI', 'CS_RACA_PRIVILEGED', 1)
        df = pd.DataFrame(dfs, index=['KL'])
        df.columns = ['id', 'KL']
        return df
    
    @staticmethod
    def ks_per_state(df):
        """ Returns a dictionary with the KS for each state """
        dfs = {}
        ptb = PreTrainingBias()
        for state in df['SG_UF_NOT'].unique():
            new_df = df.loc[df['SG_UF_NOT'] == state].copy()
            dfs[state] = ptb.ks(new_df, 'UTI', 'CS_RACA_PRIVILEGED', 1)
        df = pd.DataFrame(dfs, index=['KS'])
        df.columns = ['id', 'KS']
        return df
    
    @staticmethod
    def ci_per_state(df):
        """ Returns a dictionary with the class imbalance for each state """
        dfs = {}
        ptb = PreTrainingBias()
        for state in df['SG_UF_NOT'].unique():
            new_df = df.loc[df['SG_UF_NOT'] == state].copy()
            dfs[state] = ptb.class_imbalance(new_df, 'CS_RACA_PRIVILEGED')
        df = pd.DataFrame(dfs, index=['CI'])
        df.columns = ['id', 'CI']
        return df
    
    @staticmethod
    def state_dataframes(df):
        """ Returns a dictionary with the dataframes for each state """
        dfs = {}
        for state in df['SG_UF_NOT'].unique():
            dfs[state] = df.loc[df['SG_UF_NOT'] == state]
        return dfs
