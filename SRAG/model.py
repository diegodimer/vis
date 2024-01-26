""" Module to train the model for a given year"""
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, f1_score



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

    def train_and_save_regional_model_for_year(self, df: pd.DataFrame, region: str):
        """ Train a model for all states, writing the saved model to a file"""
        x = df.drop(columns=['UTI', 'SG_UF_NOT', 'ID_MUNICIP', 'SG_UF_INTE', 'SG_UF', "ID" ])
        y = df['UTI']
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        grid = RandomForestClassifier(n_estimators=500  )
        grid.fit(x_train, y_train)

        pred_grid = grid.predict(x_test)
        print(f"Confusion Matrix for {region}: ")
        print(confusion_matrix(y_test, pred_grid))
        print(f"Classification Report for {region}: ")
        print(classification_report(y_test, pred_grid))

        filename = f'resources/models/{self.year}/{region}.sav'

        with open(filename, 'wb') as f:
            pickle.dump(grid, f)

    def generate_regional_models(self):
        """ Generate a model for each region"""
        for region, df in self.region_data.items():
            self.train_and_save_regional_model_for_year(df, region)

    def load_all_models(self) -> dict:
        """ Load all models from a file"""
        for region in self.region_data:
            filename = f'resources/models/{self.year}/{region}.sav'
            with open(filename, 'rb') as f:
                loaded_model = pickle.load(f)
            self.models[region] = loaded_model

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