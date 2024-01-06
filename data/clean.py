""" This file is used to clean the data from the csv file and generate a processed csv file"""""
import pandas as pd

class DataReader:
    """ Class to read the data from the csv file and generate a processed csv file"""
    def __init__(self, columns_file, csv_file):
        self.columns_file = columns_file
        self.csv_file = csv_file

    @staticmethod
    def pre_process_srag_2021():
        """ Preprocess the SRAG data from 2021 """
        csv_file = "resources/datasets/INFLUD21-01-05-2023.csv"
        with open("resources/columns.txt", 'r', encoding='utf-8') as f:
            columns = [line.rstrip() for line in f]

        df = pd.read_csv(csv_file, sep=';', quotechar='"', encoding='utf-8')
        df = df.filter(items=columns)
        df.to_csv("resources/datasets/PROCESSED_INFLUD21-01-05-2023.csv")
        return df

    @staticmethod
    def pre_process_srag_2023():
        """ Preprocess the SRAG data from 2023 """
        csv_file = "resources/datasets/INFLUD23-16-10-2023.csv"
        with open("resources/columns.txt", 'r', encoding='utf-8') as f:
            columns = [line.rstrip() for line in f]

        df = pd.read_csv(csv_file, sep=';', quotechar='"', encoding='utf-8')
        df = df.filter(items=columns)
        df.to_csv("resources/datasets/PROCESSED_INFLUD23-16-10-2023.csv")
        return df

    def beautify_dataframe(self, df):
        """ Beautify the dataframe """
        # SG_UF_NOT is the code to use for the state on the chloropleth map
        
DataReader.pre_process_srag_2021()
DataReader.pre_process_srag_2023()
