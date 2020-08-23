import shutil
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import umap
from category_encoders.one_hot import OneHotEncoder
# from cyvcf2 import VCF
from MulticoreTSNE import MulticoreTSNE as TSNE
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer
from sklearn.neighbors import KNeighborsClassifier
from snps import SNPs

warnings.filterwarnings("ignore")
st.set_option('deprecation.showfileUploaderEncoding', False)


def main():
    # Render the readme as markdown using st.markdown.
    readme_text = st.markdown(get_file_content_as_string("intro.md"))
    st.title("Project Video")
    st.video(None)
    # Once we have the dependencies, add a selector for the app mode on the sidebar.
    st.sidebar.title("Model Settings")
    # select which set of SNPs to explore
    aisnp_set = st.sidebar.radio(
        "Set of model type:",
        ("Facial classification", "Text classification"),
    )

    if aisnp_set == "Facial classification":
        resize = st.sidebar.number_input(label="Enter an image resize size. Ex. 256", step=1, min_value=1, value=256)
        crop = st.sidebar.number_input(label="Enter an image center crop size. Ex. 256 (enter 1 for None)", step=1,
                                       min_value=1, value=256)
        if crop > resize:
            st.sidebar.error("crop size should be less than or equal to resize size")

        normalize = st.sidebar.text_input(label="Enter your desired normalization in the format: [[0.5,0.5,0.5],[0.5,"
                                                "0.5,0.5]]", value="[[0.5,0.5,0.5],[0.5,0.5,0.5]]")

    elif aisnp_set == "Text classification":
        tokenizer = st.sidebar.radio(
            "Tokenizer:", ("T5", "Bert")
        )
        max_length = st.sidebar.number_input(label="Enter text max length", step=1,
                                             min_value=1, value=256)

    classes = st.sidebar.text_input(label="Enter the output classes in a comma separated ascending social class "
                                          "order. Ex. \"1,2,0,3\" ")
    # upload the file
    user_file = st.sidebar.file_uploader("Upload your model in .pth format:")
    # Collapsable user AISNPs DataFrame
    if user_file is not None:
        try:
            with st.spinner("Uploading your model..."):
                with open("user_snps_file.txt", "w") as file:
                    user_file.seek(0)
                    shutil.copyfileobj(user_file, file)
        except Exception as e:
            st.error(
                f"Sorry, there was a problem processing your model file.\n {e}"
            )

    st.sidebar.button("Submit")

    # filter and encode the user record
    #     user_record, aisnps_1kg = filter_user_genotypes(userdf, aisnps_1kg)
    #     user_encoded = encoder.transform(user_record)
    #     X_encoded = np.concatenate((X_encoded, user_encoded))
    #     del userdf
    #
    #     # impute the user record and reduce the dimensions
    #     user_imputed = impute_missing(X_encoded)
    #     user_reduced = reducer.transform([user_imputed])
    #     # fit the knn before adding the user sample
    #     knn.fit(X_reduced, dfsamples[population_level])
    #
    #     # concat the 1kg and user reduced arrays
    #     X_reduced = np.concatenate((X_reduced, user_reduced))
    #     dfsamples.loc["me"] = ["me"] * 3
    #
    #     # plot
    #     plotly_3d = plot_3d(X_reduced, dfsamples, population_level)
    #     st.plotly_chart(plotly_3d, user_container_width=True)
    #
    #     # predict the population for the user sample
    #     user_pop = knn.predict(user_reduced)[0]
    #     st.subheader(f"Your predicted {population_level}")
    #     st.text(f"Your predicted population using KNN classifier is {user_pop}")
    #     # show the predicted probabilities for each population
    #     st.subheader(f"Your predicted {population_level} probabilities")
    #     user_pop_probs = knn.predict_proba(user_reduced)
    #     user_probs_df = pd.DataFrame(
    #         [user_pop_probs[0]], columns=knn.classes_, index=["me"]
    #     )
    #     st.dataframe(user_probs_df)
    #
    #     show_user_gts = st.sidebar.checkbox("Show Your Genotypes")
    #     if show_user_gts:
    #         user_table_title = "Genotypes of Ancestry-Informative SNPs in Your Sample"
    #         st.subheader(user_table_title)
    #         st.dataframe(user_record)
    #
    # else:
    #     # plot
    #     plotly_3d = plot_3d(X_reduced, dfsamples, population_level)
    #     st.plotly_chart(plotly_3d, user_container_width=True)

    # Collapsable 1000 Genomes sample table
    # show_1kg = st.sidebar.checkbox("Show 1k Genomes Genotypes")
    # if show_1kg is True:
    #     table_title = (
    #         "Genotypes of Ancestry-Informative SNPs in 1000 Genomes Project Samples"
    #     )
    #     with st.spinner("Loading 1k Genomes DataFrame"):
    #         st.subheader(table_title)
    #         st.dataframe(aisnps_1kg)

    # Render the readme as markdown using st.markdown.
    readme_text = st.markdown(get_file_content_as_string("details.md"))


@st.cache
def get_file_content_as_string(mdfile):
    """Convenience function to convert file to string

    :param mdfile: path to markdown
    :type mdfile: str
    :return: file contents
    :rtype: str
    """
    mdstring = ""
    with open(mdfile, "r") as f:
        for line in f:
            mdstring += line
    return mdstring


def get_1kg_samples():
    """Download the sample information for the 1000 Genomes Project

    :return: DataFrame of sample-level population information
    :rtype: pandas DataFrame
    """
    onekg_samples = "data/integrated_call_samples_v3.20130502.ALL.panel"
    dfsamples = pd.read_csv(onekg_samples, sep="\t")
    dfsamples.set_index("sample", inplace=True)
    dfsamples.drop(columns=["Unnamed: 4", "Unnamed: 5"], inplace=True)
    dfsamples.columns = ["population", "super population", "gender"]
    return dfsamples


@st.cache(show_spinner=True)
def encode_genotypes(df):
    """One-hot encode the genotypes

    :param df: A DataFrame of samples with genotypes as columns
    :type df: pandas DataFrame
    :return: pandas DataFrame of one-hot encoded columns for genotypes and OHE instance
    :rtype: pandas DataFrame, OneHotEncoder instance
    """
    ohe = OneHotEncoder(cols=df.columns, handle_missing="return_nan")
    X = ohe.fit_transform(df)
    return pd.DataFrame(X, index=df.index), ohe


def dimensionality_reduction(X, algorithm="PCA"):
    """Reduce the dimensionality of the AISNPs
    :param X: One-hot encoded 1kG AISNPs.
    :type X: pandas DataFrame
    :param algorithm: The type of dimensionality reduction to perform.
        One of {PCA, UMAP, t-SNE}
    :type algorithm: str
    :returns: The transformed X DataFrame, reduced to 3 components by <algorithm>,
    and the dimensionality reduction Transformer object.
    """
    n_components = 3

    if algorithm == "PCA":
        reducer = PCA(n_components=n_components)
    elif algorithm == "t-SNE":
        reducer = TSNE(n_components=n_components, n_jobs=4)
    elif algorithm == "UMAP":
        reducer = umap.UMAP(
            n_components=n_components, min_dist=0.2, metric="dice", random_state=42
        )
    else:
        return None, None

    X_reduced = reducer.fit_transform(X.values)

    return pd.DataFrame(X_reduced, columns=["x", "y", "z"], index=X.index), reducer


@st.cache(show_spinner=True)
def filter_user_genotypes(userdf, aisnps_1kg):
    """Filter the user's uploaded genotypes to the AISNPs

    :param userdf: The user's DataFrame from SNPs
    :type userdf: pandas DataFrame
    :param aisnps_1kg: The DataFrame containing snps for the 1kg project samples
    :type aisnps_1kg: pandas DataFrame
    :return: The user's DataFrame of AISNPs as columns, The 1kg DataFrame with user appended
    :rtype: pandas DataFrame
    """
    user_record = pd.DataFrame(index=["your_sample"], columns=aisnps_1kg.columns)
    for snp in user_record.columns:
        try:
            user_record[snp] = userdf.loc[snp]["genotype"]
        except KeyError:
            continue
    aisnps_1kg = aisnps_1kg.append(user_record)
    return user_record, aisnps_1kg


@st.cache(show_spinner=True)
def impute_missing(aisnps_1kg):
    """Use scikit-learns KNNImputer to impute missing genotypes for AISNPs

    :param aisnps_1kg: DataFrame of all samples including user's encoded genotypes.
    :type aisnps_1kg: pandas DataFrame
    :return: DataFrame with nan values filled in my KNNImputer
    :rtype: pandas DataFrame
    """
    imputer = KNNImputer(n_neighbors=9)
    imputed_aisnps = imputer.fit_transform(aisnps_1kg)
    return np.rint(imputed_aisnps[-1])


def plot_3d(X_reduced, dfsamples, pop):
    """Display the 3d scatter plot.

    :param X_reduced: DataFrame of all samples feature-space features.
    :type X_reduced: pandas DataFrame
    :param dfsamples: DataFrame witih sample-level info on each 1kg sample.
    :type dfsamples: pandas DataFrame
    :param pop: The population resolution to plot
    :type pop: str
    :return: plotly figure
    :rtype: plotly figure
    """
    X = np.hstack((X_reduced, dfsamples))
    columns = [
        "component_1",
        "component_2",
        "component_3",
        "population",
        "super population",
        "gender",
    ]
    df = pd.DataFrame(X, columns=columns, index=dfsamples.index)
    color_discrete_map = {"me": "rgb(0,0,0)"}
    df["size"] = 16
    if "me" in dfsamples.index.tolist():
        df["size"].loc["me"] = 75

    fig = px.scatter_3d(
        df,
        x="component_1",
        y="component_2",
        z="component_3",
        color=pop,
        color_discrete_map=color_discrete_map,
        symbol=pop,
        height=600,
        size="size",
        opacity=0.95,
        color_discrete_sequence=["#008fd5", "#fc4f30", "#e5ae38", "#6d904f", "#810f7c"],
    )
    if "me" not in dfsamples.index.tolist():
        fig.update_traces(marker=dict(size=2))

    return fig


if __name__ == "__main__":
    main()
