import streamlit as st
import torch
import clip
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import os

# -----------------------------------
# PAGE CONFIG ONE PAGE LAYOUT
# -----------------------------------
st.set_page_config(layout="wide")
st.title("CLIP Image–Text Similarity Demo")

# -----------------------------------
# LOAD CLIP MODEL
# -----------------------------------
@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess, device

model, preprocess, device = load_model()

# -----------------------------------
# DEFAULT IMAGE GRID
# -----------------------------------
IMAGE_FOLDER = "images"

image_files = [
    f for f in os.listdir(IMAGE_FOLDER)
    if f.endswith((".jpg", ".png", ".jpeg"))
]

st.subheader("Select a Default Image")

# Session state for persistence
if "selected_image_path" not in st.session_state:
    st.session_state.selected_image_path = None

# Responsive horizontal grid
num_cols = 5
rows = [
    image_files[i:i + num_cols]
    for i in range(0, len(image_files), num_cols)
]

for row in rows:

    cols = st.columns(len(row))

    for col, img_file in zip(cols, row):

        img_path = os.path.join(IMAGE_FOLDER, img_file)
        img = Image.open(img_path)

        col.image(img, use_container_width=True)

        if col.button("Select", key=img_file):
            st.session_state.selected_image_path = img_path

# -----------------------------------
# UPLOAD OPTION
# -----------------------------------
uploaded_file = st.file_uploader(
    "Or Upload Your Own Image",
    type=["jpg", "jpeg", "png"]
)

# Decide image source
image = None

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

elif st.session_state.selected_image_path is not None:
    image = Image.open(
        st.session_state.selected_image_path
    ).convert("RGB")

# -----------------------------------
# LABEL SECTION
# -----------------------------------
st.subheader("Text Labels")

default_labels = [
    "a photo of a dog",
    "a forest",
    "a city",
    "a mountain",
    "a river",
    "a desert",
    "a highway",
    "a village"
]

selected_labels = st.multiselect(
    "Select Labels",
    default_labels
)

# ADD NEW LABEL DYNAMICALLY
new_label = st.text_input("Add New Label")

if new_label:
    if new_label not in selected_labels:
        selected_labels.append(new_label)

# -----------------------------------
# MAIN EXECUTION
# -----------------------------------
if image is not None and len(selected_labels) > 0:

    # Resize image
    image = image.resize((252, 252))

    # -----------------------------------
    # ROW 1 → IMAGE + BAR GRAPH
    # -----------------------------------
    col1, col2 = st.columns(2)

    # IMAGE
    with col1:
        st.image(image, caption="Input Image", width=300)

    # -----------------------------------
    # ENCODE IMAGE + TEXT
    # -----------------------------------
    image_input = preprocess(image).unsqueeze(0).to(device)
    text_inputs = clip.tokenize(selected_labels).to(device)

    with torch.no_grad():

        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_inputs)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        similarity = (
            image_features @ text_features.T
        ).cpu().numpy()

    scores = similarity[0]

    # -----------------------------------
    # BAR GRAPH for similarity scores
    # -----------------------------------
    with col2:

        fig_bar, ax_bar = plt.subplots(figsize=(5, 4))

        ax_bar.barh(selected_labels, scores)
        ax_bar.invert_yaxis()
        ax_bar.set_title("Similarity Scores")

        st.pyplot(fig_bar)
        plt.close(fig_bar)

    # -----------------------------------
    # ROW 2 → HEATMAP + TSNE
    # -----------------------------------
    col3, col4 = st.columns(2)

    # ---------- HEATMAP ----------
    with col3:

        fig1, ax1 = plt.subplots(figsize=(5, 4))

        sns.heatmap(
            similarity,
            annot=True,
            xticklabels=selected_labels,
            yticklabels=["Image"],
            cmap="viridis",
            ax=ax1
        )

        ax1.set_title("Similarity Heatmap")

        st.pyplot(fig1)
        plt.close(fig1)

    # -----------------------------------
    # TSNE PERPLEXITY FIXED
    # -----------------------------------
    with col4:

        combined = torch.cat(
            [image_features, text_features]
        ).cpu().numpy()

        labels_plot = ["Image"] + selected_labels
        n_samples = combined.shape[0]

        if n_samples < 3:

            st.info(
                "t-SNE needs at least 2 text labels."
            )

        else:

            perplexity = min(3, n_samples - 1)

            tsne = TSNE(
                n_components=2,
                perplexity=perplexity,
                random_state=42
            )

            reduced = tsne.fit_transform(combined)

            fig2, ax2 = plt.subplots(figsize=(5, 4))

            for i, label in enumerate(labels_plot):
                x, y = reduced[i]
                ax2.scatter(x, y)
                ax2.text(x + 1, y + 1, label, fontsize=9)

            ax2.set_title("Joint Embedding t-SNE")

            st.pyplot(fig2)
            plt.close(fig2)

else:

    st.info(
        "Select/upload an image and choose labels."
    )
