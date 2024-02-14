#pylint: skip-file

from SRAG.model import ModelTrainer
import pandas as pd
import altair as alt


model_2021 = ModelTrainer("2021", 'UTI')
model_2022 = ModelTrainer("2022", 'VACINA_COV')
model_2023 = ModelTrainer("2023", 'VACINA_COV')
acc_2021 = {}
acc_2022 = {}
acc_2023 = {}
f1_2021 = {}
f1_2022 = {}
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

print("\n\nDATA FROM 2022")

for predicted_region in model_2022.region_data:
    acc_region = {}
    f1_region = {}
    for model_region in model_2022.region_data:
        print(f"Predicting {predicted_region} with model for {model_region}")
        acc_model, f1_model = model_2022.predict_for_region(model_region, predicted_region)
        acc_region[f"model {model_region}"] = acc_model
        f1_region[f"model {model_region}"] = f1_model
    acc_2022[predicted_region] = acc_region
    f1_2022[predicted_region] = f1_region

pd.DataFrame(acc_2022).to_csv("resources/datasets/acc-2022.csv")
pd.DataFrame(f1_2022).to_csv("resources/datasets/f1-2022.csv")    

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

for year in ['2021', '2022', '2023']:
    for model_region in model_2021.region_data:
        for region in model_2021.region_data:
            # number of false positives in the region for the race attribute
            df = pd.read_csv(f"resources/datasets/{year}/{model_region}/{region}.csv")
            df['CS_RACA_PRIVILEGED'] = df['CS_RACA'].map({1: 1, 2: 0, 3: 0, 4:0, 5:0})
            
            false_positives = df.loc[(df['predicted'] == 1) & (df['actual'] == 0)]
            false_negatives = df.loc[(df['predicted'] == 0) & (df['actual'] == 1)]
            true_positives = df.loc[(df['predicted'] == 1) & (df['actual'] == 1)]
            true_negatives = df.loc[(df['predicted'] == 0) & (df['actual'] == 0)]
            
            num_privileged = df.value_counts("CS_RACA_PRIVILEGED")[1]
            num_unprivileged = df.value_counts("CS_RACA_PRIVILEGED")[0]
            
            false_positive_per_race = false_positives.groupby('CS_RACA_PRIVILEGED').size()
            false_negative_per_race = false_negatives.groupby('CS_RACA_PRIVILEGED').size()
            true_positive_per_race = true_positives.groupby('CS_RACA_PRIVILEGED').size()
            true_negative_per_race = true_negatives.groupby('CS_RACA_PRIVILEGED').size()
            
            # sum_false_positive_race = false_positive_per_race[0] + false_positive_per_race[1]
            # sum_false_negative_race = false_negative_per_race[0] + false_negative_per_race[1]
            # sum_true_positive_race  = true_positive_per_race[0] + true_positive_per_race[1]
            # sum_true_negative_race  = true_negative_per_race[0] + true_negative_per_race[1]
            
            dic_race = { 
                "false positive" : { 'unprivileged': (false_positive_per_race[0]/num_unprivileged), 'privileged': (false_positive_per_race[1]/num_privileged)},
                "false negative": { 'unprivileged':  (false_negative_per_race[0]/num_unprivileged), 'privileged': (false_negative_per_race[1]/num_privileged)},
                "true positive": { 'unprivileged':   (true_positive_per_race[0]/num_unprivileged),  'privileged': (true_positive_per_race[1]/num_privileged)},
                "true negative": { 'unprivileged':   (true_negative_per_race[0]/num_unprivileged),  'privileged': (true_negative_per_race[1]/num_privileged)}
            }
            
            df_race = pd.melt(pd.DataFrame(dic_race).reset_index(), id_vars='index',  var_name='Output', value_name='count')
            
            # Plotting using Altair
            alt.Chart(df_race).mark_bar().encode(
                x=alt.X('index:O', axis=alt.Axis(title='Class')),
                y=alt.Y('count:Q', axis=alt.Axis(title='')),
                color=alt.Color('index:N', legend=alt.Legend(title='Class')),
                column= alt.Column('Output:N'),
            ).properties(width=200, height='container', title=f'{year} model trained on {model_region}, inference on region {region} - Race (Normalized)').save(f"resources/charts/{year}/Normalized-race-{year}-model-{model_region}-region-{region}.png")
                        
            false_positive_per_sex = false_positives.groupby('CS_SEXO').size()
            false_negative_per_sex = false_negatives.groupby('CS_SEXO').size()
            true_positive_per_sex = true_positives.groupby('CS_SEXO').size()
            true_negative_per_sex = true_negatives.groupby('CS_SEXO').size()
            
            num_privileged_sex = df.value_counts("CS_SEXO")[1]
            num_unprivileged_sex = df.value_counts("CS_SEXO")[0]
            
            # sum_false_positive_sex = false_positive_per_sex[0] + false_positive_per_sex[1]
            # sum_false_negative_sex = false_negative_per_sex[0] + false_negative_per_sex[1]
            # sum_true_positive_sex  = true_positive_per_sex[0] + true_positive_per_sex[1]
            # sum_true_negative_sex  = true_negative_per_sex[0] + true_negative_per_sex[1]
            
            dic_sex = { 
                "false positive" : { 'unprivileged': (false_positive_per_sex[0]/num_unprivileged_sex), 'privileged': (false_positive_per_sex[1]/num_privileged_sex)},
                "false negative": { 'unprivileged':  (false_negative_per_sex[0]/num_unprivileged_sex), 'privileged': (false_negative_per_sex[1]/num_privileged_sex)},
                "true positive": { 'unprivileged':   (true_positive_per_sex[0]/num_unprivileged_sex),  'privileged': (true_positive_per_sex[1]/num_privileged_sex)},
                "true negative": { 'unprivileged':   (true_negative_per_sex[0]/num_unprivileged_sex),  'privileged': (true_negative_per_sex[1]/num_privileged_sex)}
            }
            
            df_sex = pd.melt(pd.DataFrame(dic_sex).reset_index(), id_vars='index',  var_name='Output', value_name='count')


            alt.Chart(df_sex).mark_bar().encode(
                x=alt.X('index:O', axis=alt.Axis(title='Class')),
                y=alt.Y('count:Q', axis=alt.Axis(title='')),
                color=alt.Color('index:N', legend=alt.Legend(title='Class')),
                column= alt.Column('Output:N'), 
            ).properties(width=200, height='container',title=f'{year} model trained on {model_region}, inference on region {region} - Sex (Normalized)').save(f"resources/charts/{year}/Normalized-sex-{year}-model-{model_region}-region-{region}.png")