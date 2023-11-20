import altair as alt
import pandas as pd
import streamlit as st



def main():
    st.title("Bias comparison on uploaded dataset")
    new_col = ""
    with st.sidebar:
        st.header("Configuration")
    
        input_data = st.file_uploader(label="Upload a CSV file to use as input data", type=({"csv"}))
        if input_data is not None:
            st.markdown("### Sample from Input Data")
            # st.dataframe(input_data)
            if input_data is not None:
                df = pd.read_csv(input_data)
                st.write(df[:10])
    if input_data is not None:

        col1, col2 = st.columns(2)
        with col1: 
            col = st.selectbox("Select the feature", df.columns)
            feat_type = st.radio('Choose the feature type:', ['Binary', 'Categorical', 'Numerical'])
            
            if feat_type == 'Numerical':
                try:
                    values = st.slider('Select the privilleged class',df[col].min(), df[col].max(), (df[col].min(), df[col].max()))
                    df[f"{col}_privilleged"] = df[col].apply(lambda x: "Privilleged" if values[0] <= x and values[1] >= x else "Unprivilleged")
                    new_col = f"{col}_privilleged"
                except:
                    st.error(f"{col} is not a numerical feature. Please select another feature type or feature.")
            
            elif feat_type == 'Binary':
                if(len(df[col].unique()) > 2): 
                    st.error(f"{col} is not a binary feature. Please select another feature type or feature.")
                    return
                value = st.radio('Select the privilleged class', df[col].unique())
                df[f"{col}_privilleged"] = df[col].apply(lambda x: "Privilleged" if value == x else "Unprivilleged")
                new_col = f"{col}_privilleged"
            
            elif feat_type == 'Categorical':
                values = st.multiselect('Select the privilleged classes', df[col].unique())
   
                df[f"{col}_privilleged"] = df[col].apply(lambda x: "Privilleged" if x in values else "Unprivilleged")

                new_col = f"{col}_privilleged"

            metrics = st.multiselect('Select the metrics to visualize', ['Class Imbalance', 'KL Divergence', 'KS', 'CDDL'])
        
        with col2:
            if new_col:    
                st.markdown("### Class distribution")
                c = (
                    alt.Chart(df)
                    .mark_bar()
                    .encode(x=new_col, y="count()")
                )
                st.altair_chart(c, use_container_width=True)

            if st.checkbox("Show original class distribution", value=False):
                st.altair_chart((
                alt.Chart(df)
                .mark_bar()
                .encode(x=col, y="count()")
            ), use_container_width=True)
            




if __name__ == "__main__":
    st.set_page_config(
        page_title="Streamlit VIS demo", page_icon=":incoming_envelope:"
    )
    main()
    with st.sidebar:
        st.markdown("---")

        