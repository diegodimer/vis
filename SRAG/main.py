#pylint: skip-file

from SRAG.model import ModelTrainer
import pandas as pd

# Sex 2021
# maior discrepancia de KL: nordeste e sul
# maior discrepancia de KS: nordeste e sul
# maior discrepancia de CI: norte e nordeste

# Sex 2023
# maior discrepancia de KL: nordeste e sul
# maior discrepancia de KS: nordeste e sul
# maior discrepancia de CI: norte e sul

# Race 2023
# maior discrepancia de KL: nordeste e sul + nordeste e sudeste + nordeste e centro-oeste
# maior discrepancia de KS: nordeste e sul + nordeste e sudeste + nordeste e centro-oeste
# maior discrepancia de CI: sul e sudeste

# Race 2021
# maior discrepancia de KL: norte e sul + sudeste e norte + centro-oeste e norte
# maior discrepancia de KS: norte e sul + sudeste e norte + centro-oeste e norte
# maior discrepancia de CI: sul e sudeste

model_2021 = ModelTrainer("2021")
model_2023 = ModelTrainer("2023")
acc_2021 = {}
acc_2023 = {}
f1_2021 = {}
f1_2023 = {}

print("DATA FROM 2021")

""" for each region, we predict the UTI with the models for all the other regions """
for predicted_region in model_2021.region_data:
    acc_region = {}
    f1_region = {}
    for model_region in model_2021.region_data:
        print(f"Predicting {predicted_region} with model for {model_region}")
        acc_model, f1_model = model_2021.predict_for_region(model_region, predicted_region)
        acc_region[f"model {model_region}"] = acc_model
        f1_region[f"model {model_region}"] = f1_model
    acc_2021[predicted_region] = acc_region
    f1_2021[predicted_region] = f1_region

pd.DataFrame(acc_2021).to_csv("resources/datasets/acc-2021.csv")
pd.DataFrame(f1_2021).to_csv("resources/datasets/f1-2021.csv")

print("\n\nDATA FROM 2023")

for predicted_region in model_2023.region_data:
    acc_region = {}
    f1_region = {}
    for model_region in model_2023.region_data:
        print(f"Predicting {predicted_region} with model for {model_region}")
        acc_model, f1_model = model_2023.predict_for_region(model_region, predicted_region)
        acc_region[f"model {model_region}"] = acc_model
        f1_region[f"model {model_region}"] = f1_model
    acc_2023[predicted_region] = acc_region
    f1_2023[predicted_region] = f1_region

pd.DataFrame(acc_2023).to_csv("resources/datasets/acc-2023.csv")
pd.DataFrame(f1_2023).to_csv("resources/datasets/f1-2023.csv")    