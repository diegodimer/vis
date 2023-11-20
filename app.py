"""main streamlit app file"""
import altair as alt
import pandas as pd
import streamlit as st

from pre_training_bias import PreTrainingBias

def show_feature_config(df):
    """show feature configuration"""
    st.markdown("### Feature configuration")
    col = st.selectbox("Select the feature", df.columns)
    feat_type = st.radio(
        'Choose the feature type:', ['Binary', 'Categorical', 'Numerical'])
    group_variable = ""
    if feat_type == 'Numerical':
        try:
            values = st.slider('Select the privileged class',
                                df[col].min(), df[col].max(), (df[col].min(), df[col].max()))
            df[f"{col}_privileged"] = df[col].apply(
                lambda x: "Privileged" if values[0] <= x and values[1] >= x else "Unprivileged")
            new_col = f"{col}_privileged"
        except ValueError:
            st.error(
                f"{col} is not a numerical feature. Please select another feature type or feature.")

    elif feat_type == 'Binary':
        if len(df[col].unique()) > 2:
            st.error(
                f"{col} is not a binary feature. Please select another feature type or feature.")
        else:
            value = st.radio('Select the privileged class', df[col].unique())
            df[f"{col}_privileged"] = df[col].apply(
                lambda x: "Privileged" if value == x else "Unprivileged")
            new_col = f"{col}_privileged"

    elif feat_type == 'Categorical':
        values = st.multiselect(
            'Select the privileged classes', df[col].unique())

        df[f"{col}_privileged"] = df[col].apply(
            lambda x: "Privileged" if x in values else "Unprivileged")

        new_col = f"{col}_privileged"

    metrics = st.multiselect(
        'Select the metrics to visualize', ['Class Imbalance', 'KL Divergence', 'KS', 'CDDL'])
    target = st.selectbox("Select the target variable (attribute to be predicted)", df.columns)
    positive_outcome = st.selectbox(
        "Select the positive outcome", df[target].unique())
    if 'CDDL' in metrics:
        group_variable = st.selectbox(
            "Select the group variable (for CDDL)", df.columns)

    return col, new_col, metrics, target, positive_outcome, group_variable

def compute_metrics(df, new_col, metrics, target, positive_outcome, group_variable):
    """compute metrics"""
    ptb = PreTrainingBias()
    if 'Class Imbalance' in metrics:
        st.markdown("### Class Imbalance")
        try:
            st.text(ptb.class_imbalance(df=df, label=new_col))
        except ValueError:
            st.error(
                "Invalid value for class imbalance. Check feature configuration")

    if 'KL Divergence' in metrics:
        st.markdown("### KL Divergence")
        try:
            st.text(ptb.kl_divergence(df=df, target=target, protected_attribute=new_col,
                                        privileged_group="Privileged"))
        except ValueError:
            st.error(
                "Invalid value for KL Divergence. Check feature configuration")

    if 'KS' in metrics:
        st.markdown("### KS")
        try:
            st.text(ptb.ks(df=df, target=target,
                            protected_attribute=new_col, privileged_group="Privileged"))
        except ValueError:
            st.error("Invalid value for KS. Check feature configuration")

    if 'CDDL' in metrics:
        st.markdown("### CDDL")
        try:
            st.text(ptb.cddl(df=df, target=target, positive_outcome=positive_outcome,
                            protected_attribute=new_col, privileged_group="Privileged", 
                            group_variable=group_variable))
        except ValueError:
            st.error("Invalid value for CDDL. Check feature configuration")

def show_distribution_charts(df, col, new_col):
    """show distribution charts"""
    if new_col:
        st.markdown("### Class distribution")
        c = (
            alt.Chart(df)
            .mark_bar()
            .encode(alt.X(new_col, title=""), alt.Y("count()", title="Count"))
        )
        st.altair_chart(c, use_container_width=True)

    if st.checkbox("Show original class distribution", value=False):
        st.altair_chart((
            alt.Chart(df)
            .mark_bar()
            .encode(x=col, y="count()")
        ), use_container_width=True)

def main():
    """main function"""
    st.markdown("# Bias comparison on uploaded dataset")

    new_col = ""
    group_variable = ""
    with st.sidebar:
        st.header("Configuration")

        input_data = st.file_uploader(
            label="Upload a CSV file to use as input data", type={"csv"})
        if input_data is not None:
            st.markdown("### Sample from Input Data")
            df = pd.read_csv(input_data)
            st.write(df[:10])

    if input_data is not None:
        col1, col2, col3 = st.columns(3)
        with col1:
            col, new_col, metrics, target, positive_outcome, group_variable = show_feature_config(df)

        with col2:
            compute_metrics(df, new_col, metrics, target, positive_outcome, group_variable)

        with col3:
            show_distribution_charts(df, col, new_col)


if __name__ == "__main__":
    st.set_page_config(
        page_title="Streamlit VIS demo", page_icon=":incoming_envelope:", layout="wide"
    )
    main()
    with st.sidebar:
        st.markdown("---")
