import os
import pickle
import numpy as np
import streamlit as st
import tensorflow as tf

from tensorflow.keras.preprocessing.sequence import pad_sequences

# --------------------------------------------------
# Configuration
# --------------------------------------------------
MAX_LENGTH = 100
LABELS = ["World", "Sports", "Business", "Sci/Tech"]

EXAMPLES = {
    "World": (
        "World",
        "The government announced new international peace "
        "negotiations to resolve the ongoing conflict."
    ),
    "Sports": (
        "Sports",
        "India defeated Australia in the final to win "
        "the cricket championship."
    ),
    "Business": (
        "Business",
        "Global stock markets rose after major companies "
        "reported stronger quarterly profits."
    ),
    "Sci/Tech": (
        "Sci/Tech",
        "Researchers developed a new artificial intelligence "
        "system to improve computer performance."
    ),
}

st.set_page_config(
    page_title="NewsSense AI | News Classifier",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------
# Custom styling
# --------------------------------------------------
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(145deg, #0b1020 0%, #101827 100%);
    }

    .block-container {
        max-width: 1180px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    [data-testid="stSidebar"] {
        background-color: #111827;
        border-right: 1px solid #293449;
    }

    .hero {
        padding: 30px;
        border-radius: 20px;
        border: 1px solid #334155;
        background: linear-gradient(120deg, #172554, #172033 65%, #132c35);
        margin-bottom: 25px;
    }

    .eyebrow {
        color: #67e8f9;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
    }

    .hero h1 {
        color: #f8fafc;
        font-size: clamp(2rem, 5vw, 3.1rem);
        margin: 12px 0;
        line-height: 1.15;
    }

    .hero p {
        color: #cbd5e1;
        font-size: 1.05rem;
        line-height: 1.7;
        max-width: 760px;
        margin-bottom: 0;
    }

    .section-label {
        color: #94a3b8;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 1.5px;
        text-transform: uppercase;
    }

    .result-category {
        color: #f8fafc;
        font-size: 2rem;
        font-weight: 750;
        margin: 6px 0;
    }

    .muted {
        color: #94a3b8;
        font-size: 0.9rem;
    }

    div[data-testid="stMetric"] {
        background: #151f31;
        border: 1px solid #334155;
        padding: 18px;
        border-radius: 14px;
    }

    div[data-testid="stMetricLabel"] {
        color: #cbd5e1;
    }

    div[data-testid="stMetricValue"] {
        color: #f8fafc;
    }

    div.stButton > button {
        border-radius: 10px;
        font-weight: 600;
        min-height: 44px;
    }

    hr {
        border-color: #293449;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# Load the final model and matching tokenizer
# --------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model(
        "models/news_cnn_v3.keras"
    )

    with open("models/tokenizer_v3.pkl", "rb") as file:
        tokenizer = pickle.load(file)

    return model, tokenizer


# --------------------------------------------------
# Sidebar
# --------------------------------------------------
with st.sidebar:
    st.markdown("## 📰 NewsSense AI")
    st.caption("NLP + Deep Learning")
    st.divider()

    st.markdown("### About this model")
    st.write(
        "A Convolutional Neural Network (CNN) that learns "
        "patterns in news text and predicts its category."
    )

    st.markdown("### Categories")
    st.write("🌍 World")
    st.write("🏆 Sports")
    st.write("📈 Business")
    st.write("🔬 Sci/Tech")

    st.divider()

    st.metric("CNN v3 test accuracy", "91.78%")
    st.caption(
        "Reported on 7,600 AG News test articles. "
        "This is an evaluation result, not a guarantee "
        "of individual prediction accuracy."
    )

    st.divider()
    st.caption("Dataset: AG News")
    st.caption("Architecture: Parallel-filter 1D CNN")


# --------------------------------------------------
# Main page
# --------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">AI-powered text intelligence</div>
        <h1>NewsSense AI</h1>
        <p>
            Turn a news headline or article into an instant
            category prediction. Explore how a deep learning
            model interprets news across four major topics.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Load model with a friendly error message
try:
    model, tokenizer = load_artifacts()
except Exception as error:
    st.error(
        "The model could not be loaded. Check that "
        "`models/news_cnn_v3.keras` and "
        "`models/tokenizer_v3.pkl` exist."
    )
    st.exception(error)
    st.stop()

# --------------------------------------------------
# Input panel
# --------------------------------------------------
left, right = st.columns([1.5, 1], gap="large")

with left:
    st.markdown(
        '<div class="section-label">01 / Input article</div>',
        unsafe_allow_html=True,
    )
    st.subheader("What is the news about?")
    st.write(
        "Paste a headline or a short article, or start "
        "with one of the sample headlines."
    )

    selected_example = st.selectbox(
        "Try a sample",
        options=["Choose an example..."] + list(EXAMPLES.keys()),
    )

    if (
        selected_example != "Choose an example..."
        and st.session_state.get("last_example") != selected_example
    ):
        st.session_state["news_input"] = EXAMPLES[selected_example][1]
        st.session_state["last_example"] = selected_example

    if selected_example == "Choose an example...":
        st.session_state["last_example"] = "Choose an example..."

    with st.form("classification_form"):
        news_text = st.text_area(
            "News headline or article",
            key="news_input",
            height=190,
            placeholder=(
                "Example: Researchers announce a major "
                "breakthrough in renewable energy..."
            ),
            help="English-language text works best with this model.",
        )

        submitted = st.form_submit_button(
            "✨  Classify article",
            type="primary",
            use_container_width=True,
        )

with right:
    st.markdown(
        '<div class="section-label">02 / How it works</div>',
        unsafe_allow_html=True,
    )
    st.subheader("From text to prediction")

    with st.container(border=True):
        st.markdown("**01 · Tokenization**")
        st.caption(
            "Text is converted into sequences of vocabulary IDs."
        )
        st.divider()

        st.markdown("**02 · Feature learning**")
        st.caption(
            "Embedding and parallel convolution filters "
            "learn useful local text patterns."
        )
        st.divider()

        st.markdown("**03 · Classification**")
        st.caption(
            "A softmax output estimates the probability "
            "of each news category."
        )

# --------------------------------------------------
# Prediction
# --------------------------------------------------
if submitted:
    if not news_text or not news_text.strip():
        st.warning("Please enter a news headline or article first.")
    else:
        with st.spinner("Analyzing the article..."):
            sequences = tokenizer.texts_to_sequences([news_text])

            padded_text = pad_sequences(
                sequences,
                maxlen=MAX_LENGTH,
                padding="post",
                truncating="post",
            )

            probabilities = model.predict(
                padded_text,
                verbose=0,
            )[0]

            predicted_index = int(np.argmax(probabilities))
            predicted_label = LABELS[predicted_index]
            confidence = float(probabilities[predicted_index])

        st.divider()

        st.markdown(
            '<div class="section-label">03 / Prediction result</div>',
            unsafe_allow_html=True,
        )
        st.subheader("Analysis complete")

        result_col, confidence_col = st.columns([1.5, 1])

        with result_col:
            with st.container(border=True):
                st.caption("PREDICTED CATEGORY")
                st.markdown(
                    f'<div class="result-category">'
                    f'{predicted_label}</div>',
                    unsafe_allow_html=True,
                )
                st.write(
                    "The model considers this the most likely "
                    "category among the four available classes."
                )

        with confidence_col:
            st.metric(
                "Model confidence",
                f"{confidence:.2%}",
            )
            st.caption(
                "Confidence is the model's estimated probability, "
                "not a guarantee that the prediction is correct."
            )

        st.subheader("Category probability distribution")
        st.caption(
            "Compare the model's estimated probabilities "
            "across all four categories."
        )

        probability_columns = st.columns(2, gap="large")

        for i, label in enumerate(LABELS):
            probability = float(probabilities[i])
            bar_width = max(0.0, min(probability * 100, 100.0))

            with probability_columns[i % 2]:
                st.write(f"**{label}** — {probability:.2%}")

                st.html(
                    f"""
                    <div style="
                        width: 100%;
                        height: 10px;
                        background: #273449;
                        border-radius: 999px;
                        overflow: hidden;
                        margin-bottom: 22px;
                    ">
                        <div style="
                            width: {bar_width:.4f}%;
                            height: 10px;
                            background: #22d3ee;
                            border-radius: 999px;
                        "></div>
                    </div>
                    """
                )

        st.info(
            "This is a four-category news classifier. "
            "It may misclassify ambiguous, very short, "
            "or out-of-domain text."
        )


# --------------------------------------------------
# Model evaluation dashboard
# --------------------------------------------------
st.divider()

st.markdown(
    '<div class="section-label">04 / Model evaluation</div>',
    unsafe_allow_html=True,
)
st.header("How well does the model perform?")
st.write(
    "Evaluation on the AG News test set, with a comparison "
    "against a traditional machine learning baseline."
)

# Performance metrics
metric1, metric2, metric3 = st.columns(3)

with metric1:
    st.metric("CNN v3 accuracy", "91.78%")

with metric2:
    st.metric("Logistic Regression", "92.00%")

with metric3:
    st.metric("Test articles", "7,600")

st.caption(
    "Both models were evaluated on the same 7,600 test articles "
    "and trained using all 120,000 training articles."
)

# Model comparison chart
st.subheader("CNN vs traditional machine learning")

comparison_chart = "results/full_data_model_comparison.png"

if os.path.exists(comparison_chart):
    st.image(
        comparison_chart,
        caption="Accuracy, macro precision, recall and F1-score",
        use_container_width=True,
    )
else:
    st.info("Run compare_full_data.py to generate the comparison chart.")

# Confusion matrix
st.subheader("CNN v3 confusion matrix")

confusion_chart = "results/confusion_matrix.png"

if os.path.exists(confusion_chart):
    st.image(
        confusion_chart,
        caption="Predicted versus actual categories on the test set",
        use_container_width=True,
    )
else:
    st.info("Run evaluate.py to generate the confusion matrix.")

# Category-level F1 comparison
st.subheader("Category-wise performance")

category_chart = "results/full_data_category_f1.png"

if os.path.exists(category_chart):
    st.image(
        category_chart,
        caption="F1-score comparison for World, Sports, Business and Sci/Tech",
        use_container_width=True,
    )
else:
    st.info("Run compare_full_data.py to generate category-wise results.")

# Architecture explanation
st.subheader("Inside the CNN")

arch1, arch2, arch3 = st.columns(3)

with arch1:
    with st.container(border=True):
        st.markdown("**01 · Embedding**")
        st.write(
            "Converts token IDs into trainable dense word vectors."
        )

with arch2:
    with st.container(border=True):
        st.markdown("**02 · Parallel CNN filters**")
        st.write(
            "Kernel sizes 3, 5 and 7 learn different local text patterns."
        )

with arch3:
    with st.container(border=True):
        st.markdown("**03 · Classification**")
        st.write(
            "Dense layers and softmax produce probabilities for four categories."
        )

st.caption(
    "Note: accuracy and confidence are different. Test accuracy measures "
    "performance across a dataset; prediction confidence is the model's "
    "estimated probability for one input."
)


# --------------------------------------------------
# Footer
# --------------------------------------------------
st.divider()
st.markdown(
    """
    <div style="text-align:center; color:#94a3b8;
                font-size:0.85rem; padding:8px;">
        NewsSense AI · Academic NLP & Deep Learning Project
        <br>
        Powered by TensorFlow, Keras and Streamlit
        <br>
        Developed by Ayush Yadav
    </div>
    """,
    unsafe_allow_html=True,
)
