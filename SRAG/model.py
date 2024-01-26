""" Module to train the model for a given year"""
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, f1_score
from sklearn.neural_network import MLPClassifier
import os

from SRAG.data import DataReader

class ModelTrainer:
    """ Class to train the model for a given year"""
    year = ""
    region_data = {}
    models = {}
    dataReader = None

    def __init__(self, year):
        """ Initialize the model trainer"""
        self.year = year
        self.data_reader = DataReader(year)
        self.region_data = self.data_reader.region_data()
        self.load_all_models()

    def train_and_save_regional_model_for_year(self, df: pd.DataFrame, region: str, model):
        """ Train a model for all states, writing the saved model to a file"""
        x = df.drop(columns=['UTI', 'SG_UF_NOT', 'ID_MUNICIP', 'SG_UF_INTE', 'SG_UF', "ID" ])
        y = df['UTI']
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        # n_i = len(x_train.axes[1])
        # n_o = 1
        # n_s = len(x_train.axes[0])
        # alpha = 5

        # number_of_hidden_neurons = round(n_s / (alpha * (n_i + n_o)))
        # # round(number_of_hidden_neurons)
        # model = MLPClassifier(solver="adam", hidden_layer_sizes=(number_of_hidden_neurons,), random_state=42)
        model.fit(x_train, y_train)
        pred_model = model.predict(x_test)
        print(f"Confusion Matrix for {region}: ")
        print(confusion_matrix(y_test, pred_model))
        print(f"Classification Report for {region}: ")
        print(classification_report(y_test, pred_model))

        filename = f'resources/models/{self.year}/{region}.sav'
        self.models[region] = model
        with open(filename, 'wb') as f:
            pickle.dump(model, f)

    def generate_regional_models(self):
        """ Generate a model for each region"""
        for region, df in self.region_data.items():
            model = RandomForestClassifier(n_estimators=500, random_state=42)
            self.train_and_save_regional_model_for_year(df, region, model)

    def load_all_models(self) -> dict:
        """ Load all models from a file"""
        for region in self.region_data:
            filename = f'resources/models/{self.year}/{region}.sav'
            if os.path.isfile(filename):
                with open(filename, 'rb') as f:
                    loaded_model = pickle.load(f)
                self.models[region] = loaded_model
            else:
                need_to_train = True
        if need_to_train:
            self.generate_regional_models()

    def predict_for_region(self, model_region, predicted_region) -> list:
        """ Predicts the UTI for a given region"""
        model = self.models[model_region]
        target_data = self.region_data[predicted_region]

        x = target_data.drop(columns=['UTI', 'SG_UF_NOT', 'ID_MUNICIP', 'SG_UF_INTE', 'SG_UF', "ID" ])
        y = target_data['UTI']
        _, x_test, _, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

        y_pred = model.predict(x_test)
        print(f"Confusion Matrix for {predicted_region} on model trained for {model_region}: ")
        print(confusion_matrix(y_test, y_pred))
        print(f"Classification Report for {predicted_region} on model trained for {model_region}: ")
        print(classification_report(y_test, y_pred))
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        return [round(acc, 4), round(f1, 4)]