import warnings
import streamlit as st
import torch

warnings.filterwarnings("ignore")
st.set_option('deprecation.showfileUploaderEncoding', False)
st.beta_set_page_config("Pytorch Bias Detection Engine")


def main():
    # Render the readme as markdown using st.markdown.
    readme_text = st.markdown(get_file_content_as_string("intro.md"))
    st.title("Project Video")
    st.video(None)
    # Once we have the dependencies, add a selector for the app mode on the sidebar.
    st.sidebar.title("Model Settings")
    # select which set of SNPs to explore
    model_type = st.sidebar.radio(
        "Set of model type:",
        ("Facial classification", "Text classification"),
    )

    if model_type == "Facial classification":

        classes = st.sidebar.text_input(label="Enter the output classes in a comma separated ascending social class "
                                              "order. Ex. \"1,2,0,3\" ")
        # upload the file
        user_file = st.sidebar.file_uploader("Upload your model in .pt format:")

    elif model_type == "Text classification":
        tokenizer = st.sidebar.radio(
            "Tokenizer:", ("T5", "Bert")
        )
        max_length = st.sidebar.number_input(label="Enter text max length", step=1,
                                             min_value=1, value=256)
        classes = st.sidebar.text_input(label="Enter the output classes in a comma separated ascending social class "
                                              "order. Ex. \"1,2,0,3\" ")
        # upload the file
        user_file = st.sidebar.file_uploader("Upload your model in .pt format:")

    model = st.text_area(
        label="Enter your model deceleration here! Be sure to include all the imports you need and format it like you "
              "would in python",
        height=500, value="#Imports go here\n"
                          "class classifier(nn.Module):\n" +
                          "    \n" +
                          "    #define all the layers used in model\n" +
                          "    def __init__():\n" +
                          "        \n" +
                          "        #Constructor\n" +
                          "        super().__init__()          \n" +
                          "        \n" +
                          "     \n" +
                          "    def forward():\n" +
                          "        \n" +
                          "        return outputs")

    init = st.text_area(
        label="Instantiate your model"
              "would in python", height=200, value="def instantiate_model():\n" +
                                                   "  \n" +
                                                   "  #hyperparameters go here\n" +
                                                   "\n" +
                                                   "  #instantiate the model\n" +
                                                   "  model = classifier()\n" +
                                                   "  return model")

    preprocess = st.text_area(
        label="Enter your text preprocessing here! Be sure to include all the imports you need and format it like you "
              "would in python", height=200, value="def preprocess(input_string):\n" +
                                                   "\n" +
                                                   "  # define your preprocessing function here\n" +
                                                   "\n" +
                                                   " return preprocessed_input_string")

    if st.button("Submit"):
        exec(model)
        exec(init)
        exec(preprocess)
        with st.spinner("Uploading your model..."):
            try:
                model.load_state_dict(torch.load(user_file))
            except Exception as e:
                st.error(
                    f"Sorry, there was a problem processing your model file.\n {e}"
                )

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


if __name__ == "__main__":
    main()
