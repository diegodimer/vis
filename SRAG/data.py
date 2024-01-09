""" This file is used to clean the data from the csv file and generate a processed csv file"""""
import os
import pandas as pd

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

        df = pd.read_csv(csv_file, sep=';', quotechar='"', encoding='utf-8')
        DataReader.beautify_dataframe(df)
        df = df.filter(items=columns)
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
        DataReader.beautify_dataframe(df)
        df = df.filter(items=columns)
        df.to_csv("resources/datasets/PROCESSED_INFLUD23-16-10-2023.csv")
        return df

    @staticmethod
    def beautify_dataframe(df):
        """ Beautify the dataframe """
        df.dropna(subset = ['UTI', "CS_RACA", "CS_SEXO"], inplace=True)
        df.fillna(9, inplace=True)
        df.drop(df.loc[df['UTI']==9].index, inplace=True)
        df.drop(df.loc[df['CS_SEXO']=="I"].index, inplace=True)
        df.drop(df.loc[df['CS_RACA']=="9"].index, inplace=True)
        df.dropna(inplace=True)

    @staticmethod
    def state_counts(df):
        """ Returns a dataframe with the number of cases per state """
        return df.groupby('SG_UF_NOT').size().reset_index(name='counts')
