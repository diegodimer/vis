"""main streamlit app file"""
import altair as alt
import pandas as pd
import streamlit as st

from pre_training_bias import PreTrainingBias

def show_feature_config():
    """show feature configuration"""
    df = st.session_state.df
    st.markdown("### Feature configuration")
    col = st.selectbox("Select the feature", df.columns)
    feat_type = st.radio(
        'Choose the feature type:', ['Binary', 'Categorical', 'Numerical'])

    if feat_type == 'Numerical':
        try:
            values = st.slider('Select the privileged class',
                                df[col].min(), df[col].max(), (df[col].min(), df[col].max()))
            df[f"{col}_privileged"] = df[col].apply(
                lambda x: "Privileged" if values[0] <= x and values[1] >= x else "Unprivileged")
            st.session_state.new_col = f"{col}_privileged"
        except KeyError:
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
            st.session_state.new_col = f"{col}_privileged"

    elif feat_type == 'Categorical':
        values = st.multiselect(
            'Select the privileged classes', df[col].unique())

        df[f"{col}_privileged"] = df[col].apply(
            lambda x: "Privileged" if x in values else "Unprivileged")

        st.session_state.new_col = f"{col}_privileged"

    st.multiselect('Select the metrics to visualize', 
                   ['Class Imbalance', 'KL Divergence', 'KS', 'CDDL'], key="metrics")
    st.selectbox("Select the target variable (attribute to be predicted)", df.columns, key="target")
    st.selectbox("Select the positive outcome", df[st.session_state['target']].unique(),
                 key="positive_outcome")
    if 'CDDL' in st.session_state['metrics']:
        st.selectbox("Select the group variable (for CDDL)", df.columns, key="group_variable")
    
    st.session_state.col = col


def compute_metrics():
    """
    Compute various metrics for pre-training bias.
    """
    ptb = PreTrainingBias()
    df = st.session_state.df
    col = st.session_state.new_col
    metrics = st.session_state.metrics
    target = st.session_state.target
    positive_outcome = st.session_state.positive_outcome
    
    if 'Class Imbalance' in metrics:
        st.markdown("### Class Imbalance")
        try:
            st.text(ptb.class_imbalance(df=df, label=col))
        except ValueError:
            st.error(
                "Invalid value for class imbalance. Check feature configuration")

    if 'KL Divergence' in metrics:
        st.markdown("### KL Divergence")
        try:
            st.text(ptb.kl_divergence(df=df, target=target, protected_attribute=col,
                                        privileged_group="Privileged"))
        except ValueError:
            st.error(
                "Invalid value for KL Divergence. Check feature configuration")

    if 'KS' in metrics:
        st.markdown("### KS")
        try:
            st.text(ptb.ks(df=df, target=target,
                            protected_attribute=col, privileged_group="Privileged"))
        except ValueError:
            st.error("Invalid value for KS. Check feature configuration")

    if 'CDDL' in metrics:
        group_variable = st.session_state.group_variable
        st.markdown("### CDDL")
        try:
            st.text(ptb.cddl(df=df, target=target, positive_outcome=positive_outcome,
                            protected_attribute=col, privileged_group="Privileged",
                            group_variable=group_variable))
        except ValueError:
            st.error("Invalid value for CDDL. Check feature configuration")

def show_distribution_charts():
    """
        Show distribution charts.
    """
    df = st.session_state.df
    col = st.session_state.col
    new_col = st.session_state.new_col

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


def get_input_data():
    """
    Get input data from the user.
    """
    input_data = st.file_uploader(
            label="Upload a CSV file to use as input data", type={"csv"})

    if st.checkbox("Advanced settings", value=False, key="advanced_mode"):
        st.text_input("Column separator", value=",", key="col_sep",
            help="The column separator to use")
        st.text_input("Decimal separator", value=".", key="dec_sep",
            help="The decimal separator to use")
        st.text_input("Encoding", value="utf-8", key="encoding",
            help="The encoding to use")
        st.number_input("Header rows", value=0, key="header_row",
            help="The number of header rows to skip",)
    else:
        st.session_state.col_sep = ","
        st.session_state.dec_sep = "."
        st.session_state.encoding = "utf-8"
        st.session_state.header_row = 0

    if input_data is not None:
        st.markdown("### Sample from Input Data")
        df = pd.read_csv(input_data, sep=st.session_state.col_sep,
                         decimal=st.session_state.dec_sep,
                         encoding=st.session_state.encoding,
                         header=st.session_state.header_row)
        st.write(df[:10])
        st.session_state.df = df

def main():
    """main function"""
    st.markdown("# Bias comparison on uploaded dataset")

    with st.sidebar:
        st.header("Configuration")
        get_input_data()

    if 'df' in st.session_state:
        col1, col2, col3 = st.columns(3)
        with col1:
            show_feature_config()
        with col2:
            try:
                compute_metrics()
            except ValueError:
                pass
        with col3:
            try:
                show_distribution_charts()
            except ValueError:
                pass

if __name__ == "__main__":
    st.set_page_config(
        page_title="Streamlit VIS demo", page_icon=":incoming_envelope:", layout="wide"
    )
    main()
    with st.sidebar:
        st.markdown("---")
