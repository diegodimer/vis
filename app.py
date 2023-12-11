"""main streamlit app file"""
import altair as alt
import pandas as pd
import streamlit as st

from pre_training_bias import PreTrainingBias

def show_feature_config():
    """show feature configuration"""
    df = st.session_state['df']
    st.markdown("### Feature configuration")
    col = st.selectbox("Select the feature", df.columns)
    feat_type = st.radio(
        'Choose the feature type:', ['Binary', 'Categorical', 'Numerical'])

    if feat_type == 'Numerical':
        try:
            values = st.slider('Select the privileged class',
                                df[col].min(), df[col].max(), (df[col].min(), df[col].max()))
            df[f"{col}_privileged"] = df[col].apply(
                lambda x: "Privileged" if values[0] <= x <= values[1] else "Unprivileged")
            st.session_state['new_col'] = f"{col}_privileged"
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
            st.session_state['new_col'] = f"{col}_privileged"

    elif feat_type == 'Categorical':
        values = st.multiselect(
            'Select the privileged classes', df[col].unique())

        df[f"{col}_privileged"] = df[col].apply(
            lambda x: "Privileged" if x in values else "Unprivileged")

        st.session_state['new_col'] = f"{col}_privileged"

    st.multiselect('Select the metrics to visualize',
                   ['Class Imbalance', 'KL Divergence', 'KS', 'CDDL'], key="metrics")
    st.selectbox("Select the target variable (attribute to be predicted)", df.columns, key="target")
    st.selectbox("Select the positive outcome", df[st.session_state['target']].unique(),
                 key="positive_outcome")
    if 'CDDL' in st.session_state['metrics']:
        st.selectbox("Select the group variable (for CDDL)", df.columns, key="group_variable")

    st.session_state['col'] = col

def compute_metrics():
    """
    Compute various metrics for pre-training bias.
    """
    _ptb = PreTrainingBias()
    df = st.session_state['df']
    col = st.session_state['new_col'] 
    metrics = st.session_state['metrics']
    target = st.session_state['target']
    positive_outcome = st.session_state['positive_outcome']

    if 'Class Imbalance' in metrics:
        st.markdown("### Class Imbalance")
        try:
            show_class_imbalance(_ptb, df, col)
        except ValueError:
            st.error(
                "Invalid value for class imbalance. Check feature configuration")
        except KeyError:
            st.error(
                "Invalid value for class imbalance. Check feature configuration")

    if 'KL Divergence' in metrics:
        st.markdown("### KL Divergence")
        permutations_kl = st.number_input("Num permutations", 0, 10000, key="permutations_kl")
        try:
            show_kl_divergence(_ptb, df, col, target, permutations_kl)
        except ValueError:
            st.error(
                "Invalid value for KL Divergence. Check feature configuration")
        except KeyError:
            st.error(
                "Invalid value for class imbalance. Check feature configuration")

    if 'KS' in metrics:
        st.markdown("### KS")
        permutations_ks = st.number_input("Num permutations", 0, 10000, key="permutations_ks")
        try:
            show_ks(_ptb, df, col, target, permutations_ks)
        except ValueError:
            st.error("Invalid value for KS. Check feature configuration")
        except KeyError:
            st.error(
                "Invalid value for class imbalance. Check feature configuration")
    if 'CDDL' in metrics:
        group_variable = st.session_state['group_variable']
        permutations_cddl = st.number_input("Num permutations", 0, 10000, key="permutations_cddl")
        st.markdown("### CDDL")
        try:
            show_cddl(_ptb, df, col, target, positive_outcome, group_variable, permutations_cddl)
        except ValueError:
            st.error("Invalid value for CDDL. Check feature configuration")
        except KeyError:
            st.error(
                "Invalid value for class imbalance. Check feature configuration")

@st.cache_data
def show_cddl(_ptb, df, col, target, positive_outcome, group_variable, permutation_cddl):
    """ Compute and show CDDL chart """
    permutations_cddl, original_cddl = _ptb.get_cddl_permutation_values(df,
                                                                        target,
                                                                        positive_outcome,
                                                                        col,
                                                                        "Privileged", 
                                                                        group_variable,
                                                                        permutation_cddl)
    df_permutations_cddl = pd.DataFrame(permutations_cddl, columns=['CDDL'])
    df_permutations_cddl = df_permutations_cddl.sort_values('CDDL').reset_index(drop=True)
    df_permutations_cddl['index'] = df_permutations_cddl.index

    c = alt.Chart(df_permutations_cddl).mark_area(
                    color="lightblue",
                interpolate='step-after',
                line=True).encode(
                alt.X("index", title="Permutation Index"),
                alt.Y("CDDL", title="CDDL Value"))
    original_cddl_line = alt.Chart(pd.DataFrame({'cddl': [original_cddl]})).mark_rule(color='red').encode( y='cddl')
    st.altair_chart(c + original_cddl_line, use_container_width=True)

@st.cache_data
def show_ks(_ptb, df, col, target, permutations_ks):
    """ Compute and show KS chart """
    permutations_ks, original_ks = _ptb.get_ks_permutation_values(df, target, col, "Privileged", permutations_ks)
    df_permutations_ks = pd.DataFrame(permutations_ks, columns=['ks'])
    df_permutations_ks = df_permutations_ks.sort_values('ks').reset_index(drop=True)
    df_permutations_ks['index'] = df_permutations_ks.index
    c = alt.Chart(df_permutations_ks).mark_area(
                    color="lightblue",
                interpolate='step-after',
                line=True
            ).encode(
                alt.X("index", title="Permutation Index"),
                alt.Y("ks", title="KS Value"))
            
    original_ks_line = alt.Chart(pd.DataFrame({'ks': [original_ks]})).mark_rule(color='red').encode( y='ks')
    st.altair_chart(c + original_ks_line, use_container_width=True)

@st.cache_data
def show_kl_divergence(_ptb, df, col, target, permutations_kl):
    """ Compute and show KL divergence chart """
    permutations_kl, original_kl = _ptb.get_kl_divergence_permutation_values(df, target, col, "Privileged", permutations_kl)
    df_permutations_kl = pd.DataFrame(permutations_kl, columns=['kl'])
    df_permutations_kl = df_permutations_kl.sort_values('kl').reset_index(drop=True)
    df_permutations_kl['index'] = df_permutations_kl.index
    c = alt.Chart(df_permutations_kl).mark_area(
                    color="lightblue",
                interpolate='step-after',
                line=True
            ).encode(
                alt.X("index", title="Permutation Index"),
                alt.Y("kl", title="KL Value"))
        
    original_kl_line = alt.Chart(pd.DataFrame({'kl': [original_kl]})).mark_rule(color='red').encode( y='kl')
    st.altair_chart(c + original_kl_line, use_container_width=True)

@st.cache_data
def show_class_imbalance(_ptb, df, col):
    """ Compute and show class imbalance chart """
    permutations, _ = _ptb.get_class_imbalance_permutation_values(df=df, label=col, n_repetitions=1000)
    df_permutations = pd.DataFrame(permutations, columns=['ci'])
    df_permutations = df_permutations.sort_values('ci').reset_index(drop=True)
    df_permutations['index'] = df_permutations.index
    c = alt.Chart(df_permutations).mark_area(
                    color="lightblue",
                interpolate='step-after',
                line=True
            ).encode(
                alt.X("index", title="Permutation Index"),
                alt.Y("ci", title="Class Imbalance Value"))
    st.altair_chart(c, use_container_width=True)

def show_distribution_charts():
    """
        Show distribution charts.
    """
    df = st.session_state['df']
    col = st.session_state['col']
    new_col = st.session_state['new_col']
    if st.checkbox("Show class distribution", value=False):
        if new_col:
            st.markdown("### Class distribution")
            c = (
                alt.Chart(df)
                .mark_bar()
                .encode(alt.X(new_col, title=""), alt.Y("count()", title="Count"), color=f"{st.session_state['target']}:N")
            )
            st.altair_chart(c, use_container_width=True)

        if st.checkbox("Show original class distribution", value=False):
            st.altair_chart((
                alt.Chart(df)
                .mark_bar()
                .encode(x=col, y="count()",  color=f"{st.session_state['target']}:N")
            ), use_container_width=True)

def get_input_data(file, advanced_settings):
    """
    Get input data from the user.
    """
    input_data = file

    if advanced_settings:
        st.text_input("Column separator", value=",", key="col_sep",
            help="The column separator to use")
        st.text_input("Decimal separator", value=".", key="dec_sep",
            help="The decimal separator to use")
        st.text_input("Encoding", value="utf-8", key="encoding",
            help="The encoding to use")
        st.number_input("Header rows", value=0, key="header_row",
            help="The number of header rows to skip",
            min_value=0)
        if st.checkbox("Custom header", value=False, key="custom_header"):
            st.text_input("Header", value="", key="names",
                help="Comma-separated values containing the names to use for the columns")
        else:
            st.session_state['names'] = None

    else:
        st.session_state['col_sep'] = ","
        st.session_state['dec_sep'] = "."
        st.session_state['encoding'] = "utf-8"
        st.session_state['header_row'] = 0
        st.session_state['names']=None

    if input_data is not None:
        df = read_file(input_data, st.session_state['col_sep'], st.session_state['dec_sep'], 
                       st.session_state['encoding'], st.session_state['header_row'], 
                       st.session_state['names'])
        if st.toggle("Show sample data", value=False):
            st.markdown("### Sample from Input Data")
            st.write(df[:10])
        st.session_state['df'] = df

@st.cache_data
def read_file(input_data, sep, dec, encoding, header_row, names):
    """ Read file (to enable cache at upload level)"""
    df = pd.read_csv(input_data, sep=sep,
                        decimal=dec,
                        encoding=encoding,
                        header=header_row,
                        names=names.split(",") if names else None)
    return df

def main():
    """main function"""
    st.markdown("# Bias comparison on uploaded dataset")
    st.session_state['new_col'] = ""
    
    with st.sidebar:
        st.file_uploader(label="Upload a CSV file to use as input data", type={"csv"}, key="file")
        st.header("Configuration")
        st.toggle("Advanced settings", value=False, key="advanced_mode")
        get_input_data(st.session_state.file, st.session_state.advanced_mode)

    if 'df' in st.session_state:
        col1, col2, col3 = st.columns(3)
        with col1:
            show_feature_config()
        with col2:
            compute_metrics()
        with col3:
            show_distribution_charts()


if __name__ == "__main__":
    st.set_page_config(
        page_title="Streamlit VIS demo", page_icon=":incoming_envelope:", layout="wide"
    )
    main()
    with st.sidebar:
        st.markdown("---")
