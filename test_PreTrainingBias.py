import pandas as pd
from PreTrainingBias import PreTrainingBias

#  Heart Disease dataset extracted from https://www.kaggle.com/datasets/rishidamarla/heart-disease-prediction?resource=download

df = pd.read_csv("Heart_Disease_Prediction.csv")
pt = PreTrainingBias()
df_only_male = df.loc[df['Sex'] == 1]
df_only_female = df.loc[df['Sex'] == 0]


def test_CI_minusOne():
    # should return -1 as it only has female representation (unprivileged)
    assert (pt.class_imbalance_per_label(df_only_female, "Sex", 1) == -1.0)


def test_CI_plusOne():
    # should return 1 as it only has male representation (unprivileged)
    assert (pt.class_imbalance_per_label(df_only_male, "Sex", 1) == 1.0)


def test_CI_zero():
    # should return 0 as we have 1 female and 1 male (representation is equal)
    df_mixed = pd.concat([df_only_male.sample(1), df_only_female.sample(1)])
    assert (pt.class_imbalance_per_label(df_mixed, 'Sex', 1) == 0)
    del df_mixed


def test_KL_value():
    # should return a value greater than 0 as 80% of the female has presence of heart disease, while only 10% of male has. Data is unbalanced for the two demographic groups
    df_mixed = pd.concat([df_only_female.loc[df_only_female['Heart Disease'] == 'Presence'].sample(8), df_only_female.loc[df_only_female['Heart Disease'] == 'Absence'].sample(2),
                          df_only_male.loc[df_only_male['Heart Disease'] == 'Presence'].sample(1), df_only_male.loc[df_only_male['Heart Disease'] == 'Absence'].sample(9)])
    assert (round(pt.KL_divergence(df_mixed, 'Heart Disease', 'Sex', 1), 4) == 1.1457)
    del df_mixed


def test_KL_zero():
    # should return 0 as it has the same demographic distribution for male and female
    df_mixed = pd.concat([df_only_female.loc[df_only_female['Heart Disease'] == 'Presence'].sample(3), df_only_female.loc[df_only_female['Heart Disease'] == 'Absence'].sample(3),
                          df_only_male.loc[df_only_male['Heart Disease'] == 'Presence'].sample(3), df_only_male.loc[df_only_male['Heart Disease'] == 'Absence'].sample(3)])
    assert (pt.KL_divergence(df_mixed, 'Heart Disease', 'Sex', 1) == 0.0)
    del df_mixed


def test_KL_valueTwo():
    # KL example from https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-data-bias-metric-kl-divergence.html
    df_mixed = pd.concat([df_only_female.loc[df_only_female['Heart Disease'] == 'Presence'].sample(7), df_only_female.loc[df_only_female['Heart Disease'] == 'Absence'].sample(3),
                          df_only_male.loc[df_only_male['Heart Disease'] == 'Presence'].sample(2), df_only_male.loc[df_only_male['Heart Disease'] == 'Absence'].sample(8)])
    assert (round(pt.KL_divergence(df_mixed, 'Heart Disease', 'Sex', 1), 4) == 0.5341)
    del df_mixed


def test_KS_value():
    # KS using new dataset where the output can have three labels, example from https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-data-bias-metric-kolmogorov-smirnov.html
    d = {'Sex': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'Result': ['Rejected', 'Rejected', 'Rejected', 'Rejected', 'Waitlisted', 'Waitlisted', 'Waitlisted',
                                                                                         'Waitlisted', 'Accepted', 'Accepted', 'Rejected', 'Rejected', 'Waitlisted', 'Accepted', 'Accepted', 'Accepted', 'Accepted', 'Accepted', 'Accepted', 'Accepted']}
    df_college = pd.DataFrame(data=d)
    # round to get 0.5 instead of 0.49999999999999994
    assert (round(pt.KS(df_college, 'Result', 'Sex', 1), 1) == 0.5)


def test_KS_valueTwo():
    # KS with 70% of woman with heart disease, 20% of men, it should be 0.5 (max {(0.7-0.2), (0.8-0.3)})
    df_mixed = pd.concat([df_only_female.loc[df_only_female['Heart Disease'] == 'Presence'].sample(7), df_only_female.loc[df_only_female['Heart Disease'] == 'Absence'].sample(3),
                          df_only_male.loc[df_only_male['Heart Disease'] == 'Presence'].sample(2), df_only_male.loc[df_only_male['Heart Disease'] == 'Absence'].sample(8)])
    assert (pt.KS(df_mixed, 'Heart Disease', 'Sex', 1) == 0.5)
    del df_mixed


def test_KS_one():
    # KS with 100% of woman with heart disease, 0% of men, it should be 1 (max {(1-0), (1-0)})
    df_mixed = pd.concat([df_only_female.loc[df_only_female['Heart Disease'] == 'Presence'].sample(10),
                          df_only_male.loc[df_only_male['Heart Disease'] == 'Absence'].sample(10)])
    assert (pt.KS(df_mixed, 'Heart Disease', 'Sex', 1) == 1.0)
    del df_mixed


def test_KS_zeroPointOne():
    # KS with 20% of woman with heart disease, 30% of men. max { (0.2 - 0.3), (0.8 - 0.7) }. should be 0.1
    df_mixed = pd.concat([df_only_female.loc[df_only_female['Heart Disease'] == 'Presence'].sample(2),
                          df_only_female.loc[df_only_female['Heart Disease'] == 'Absence'].sample(
                              8),
                          df_only_male.loc[df_only_male['Heart Disease'] == 'Presence'].sample(
                              3),
                          df_only_male.loc[df_only_male['Heart Disease'] == 'Absence'].sample(7)])
    # was giving 0.10000000000000009, round to get the correct value
    assert (round(pt.KS(df_mixed, 'Heart Disease', 'Sex', 1), 1) == 0.1)
    del df_mixed


def test_CDDL_one():
    # CDDL, should return 1
    # +1: when there no rejections in facet a or subgroup and no acceptances in facet d or subgroup
    d = {'Sex':     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         'Result':   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
         'dummy':    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
    dfs = pd.DataFrame(data=d)
    assert (pt.CDDL(dfs, 'Result', 1, 'Sex', 1, 'dummy') == 1)


def test_CDDL_lowValue():
    # Positive values indicate there is a demographic disparity as facet d or subgroup has a greater proportion of the rejected outcomes in the dataset than of the accepted outcomes. The higher the value the less favored the facet and the greater the disparity.
    d = {'Sex':     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         'Result':   [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, ],
         'dummy':    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
    dfs = pd.DataFrame(data=d)
    assert (round(pt.CDDL(dfs, 'Result', 1, 'Sex', 1, 'dummy'), 4) == 0.5495)


def test_CDDL_oneTwo():
    # CDDL, should return 1 as there's no result=0 in the facet a (1). Dummy is inserted here as I don't want to create a correlated value for examples
    # Positive values indicate there is a demographic disparity as facet d or subgroup has a greater proportion of the rejected outcomes in the dataset than of the accepted outcomes. The higher the value the less favored the facet and the greater the disparity.
    d = {'Sex':     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         'Result':   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         'dummy':    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
    dfs = pd.DataFrame(data=d)
    assert (pt.CDDL(dfs, 'Result', 1, 'Sex', 1, 'dummy') == 1.0)


def test_CDDL_zero():
    d = {'Sex':     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         'Result':   [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
         'dummy':    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
    dfs = pd.DataFrame(data=d)
    assert (pt.CDDL(dfs, 'Result', 1, 'Sex', 1, 'dummy') == 0)


def test_CDDL_zeroTwo():
    d = {'Sex':     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         'Result':   [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
         'dummy':    [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]}
    dfs = pd.DataFrame(data=d)
    assert (pt.CDDL(dfs, 'Result', 1, 'Sex', 1, 'dummy') == 0)


def test_CDDL_oneThree():
    d = {'Sex':     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         'Result':  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         'dummy':   [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]}
    dfs = pd.DataFrame(data=d)
    assert (pt.CDDL(dfs, 'Result', 1, 'Sex', 1, 'dummy') == 1.0)


def test_CDDL_fromWork():
    d = {'Sex':     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         'Result':   [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1],
         'dummy':    [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]}
    # example from work
    dfs = pd.DataFrame(data=d)
    assert (round(pt.CDDL(dfs, 'Result', 1, 'Sex', 1, 'dummy'), 4) == 0.2381)
