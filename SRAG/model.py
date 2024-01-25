#pylint: skip-file
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from SRAG.data import DataReader
from sklearn.model_selection import GridSearchCV
import pandas as pd


class ModelTrainer:
    
    # def train_data_for_each_state(self, df: pd.DataFrame, year: str):
    #     """ Train a model for each state, writing the saved model to a file"""
    #     param_grid = {
    #         'bootstrap': [True],
    #         'max_depth': [80],
    #         'max_features': [2],
    #         'min_samples_leaf': [3],
    #         'min_samples_split': [8],
    #         'n_estimators': [100]
    #     }
        
    #     for state in df:
    #         X = df[state].drop(columns=['UTI', 'SG_UF_NOT', 'ID_MUNICIP', 'SG_UF_INTE', 'SG_UF', "ID"])
    #         y = df[state]['UTI']
    #         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    #         grid = GridSearchCV(RandomForestClassifier(), param_grid, refit = True, verbose=0)
    #         grid.fit(X_train, y_train)
    #         pred_grid = grid.predict(X_test)
    #         print(f"Confusion Matrix for {state}: ")
    #         print(confusion_matrix(y_test, pred_grid))
    #         print(f"Classification Report for {state}: ")
    #         print(classification_report(y_test, pred_grid))
    #         filename = f'resources/models/{year}/{state}.sav'
    #         pickle.dump(grid, open(filename, 'wb'))
            
    def train_data_for_all_states(self, df: pd.DataFrame, year: str, region: str):
        """ Train a model for all states, writing the saved model to a file"""
   
        sys.stdout=open(f"resources/models/{year}/{region}-logs.txt","w")
        X = df.drop(columns=['UTI', 'SG_UF_NOT', 'ID_MUNICIP', 'SG_UF_INTE', 'SG_UF', "ID" ])
        y = df['UTI']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        grid = RandomForestClassifier(n_estimators=500  )
        grid.fit(X_train, y_train)
        
        pred_grid = grid.predict(X_test)
        print(f"Confusion Matrix for {region}: ")
        print(confusion_matrix(y_test, pred_grid))
        print(f"Classification Report for {region}: ")
        print(classification_report(y_test, pred_grid))
        filename = f'resources/models/{year}/{region}.sav'
        pickle.dump(grid, open(filename, 'wb'))
        sys.stdout.close()
        
    def brasil_regions(self, df: pd.DataFrame, year: str):
        sul = ['PR', 'SC', 'RS']
        sudeste = ['SP', 'RJ', 'MG', 'ES']
        centro_oeste = ['MS', 'MT', 'GO', 'DF']
        nordeste = ['MA', 'PI', 'CE', 'RN', 'PE', 'PB', 'SE', 'AL', 'BA']
        norte = ['AC, AP', 'AM', 'PA', 'RO', 'RR', 'TO']
        
        df_sul = df.loc[df['SG_UF_NOT'].isin(sul)]
        self.train_data_for_all_states(df_sul, year, 'sul')
        df_sudeste = df.loc[df['SG_UF_NOT'].isin(sudeste)]
        self.train_data_for_all_states(df_sudeste, year, 'sudeste')
        df_centro_oeste = df.loc[df['SG_UF_NOT'].isin(centro_oeste)]
        self.train_data_for_all_states(df_centro_oeste, year, 'centro_oeste')
        df_nordeste = df.loc[df['SG_UF_NOT'].isin(nordeste)]
        self.train_data_for_all_states(df_nordeste, year, 'nordeste')
        df_norte = df.loc[df['SG_UF_NOT'].isin(norte)]
        self.train_data_for_all_states(df_norte, year, 'norte')
        
        
import sys


b = ModelTrainer()

b.brasil_regions(DataReader.get_srag_2023(), '2023')





# b.brasil_regions(DataReader.get_srag_2023(), '2023')

