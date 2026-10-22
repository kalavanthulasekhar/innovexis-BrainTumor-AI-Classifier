
from pathlib import Path
import json
import os
import shutil
import tempfile
from urllib.request import Request, urlopen

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, UnidentifiedImageError




APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent

MODEL_FILENAME = "mobilenetv2_finetuned_best.keras"
MODEL_PATH = Path(os.environ.get("MODEL_PATH", ROOT / "models" / MODEL_FILENAME))
MODEL_URL = os.environ.get("MODEL_URL")
MODEL_TOKEN = os.environ.get("MODEL_TOKEN")
MODEL_CACHE_PATH = Path(tempfile.gettempdir()) / "brain-tumor-ai" / MODEL_FILENAME
REPORT_PATH = ROOT / "reports" / "final_test_results.json"
CM_PATH = ROOT / "reports" / "figures" / "final_test_confusion_matrix.png"

IMG_SIZE = (160, 160)

CLASSES = ["glioma", "meningioma", "no_tumor", "pituitary"]

LABELS = {
    "glioma": "Glioma",
    "meningioma": "Meningioma",
    "no_tumor": "No Tumor",
    "pituitary": "Pituitary",
}

ICONS = {
    "glioma": "G",
    "meningioma": "M",
    "no_tumor": "N",
    "pituitary": "P",
}


st.set_page_config(
    page_title="BrainTumor AI",
    page_icon="BT",
    layout="wide",
    initial_sidebar_state="expanded",
)




st.markdown(
    """
    <style>
    :root {
        --bg: #07111f;
        --surface: #0d1a2b;
        --surface-raised: #122238;
        --border: rgba(151, 184, 219, .14);
        --text: #eef5ff;
        --muted: #91a7c2;
        --teal: #50e0c2;
        --blue: #83b9ff;
    }

    .stApp {
        color: var(--text);
        background:
            radial-gradient(ellipse at 70% -15%, rgba(17, 104, 122, .20), transparent 38%),
            radial-gradient(ellipse at 0% 45%, rgba(25, 68, 112, .10), transparent 28%),
            var(--bg);
    }
    [data-testid="stHeader"] { background: rgba(7, 17, 31, .72); }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0c1929 0%, #091321 100%);
        border-right: 1px solid var(--border);
    }
    [data-testid="stSidebar"] > div:first-child { padding-top: 1.2rem; }
    [data-testid="stSidebar"] h3 { margin-top: .2rem; }
    .sidebar-brand { display: flex; align-items: center; gap: .75rem; margin: .25rem 0 1.2rem; }
    .sidebar-mark { display: grid; place-items: center; width: 42px; height: 42px; border: 1px solid rgba(80, 224, 194, .35); border-radius: 13px; color: var(--teal); background: linear-gradient(145deg, rgba(34, 141, 143, .26), rgba(28, 71, 112, .28)); font-size: .86rem; font-weight: 900; letter-spacing: -.05em; }
    .sidebar-title { color: #eef5ff; font-size: 1rem; font-weight: 820; letter-spacing: -.02em; }
    .sidebar-caption { margin-top: .12rem; color: #8298b1; font-size: .65rem; letter-spacing: .09em; text-transform: uppercase; }
    [data-testid="stSidebar"] [data-testid="stRadio"] label { border-radius: 10px; }
    .block-container { max-width: 1500px; padding: 1.8rem 2.2rem 3.5rem; }

    h1, h2, h3, h4 { color: var(--text); letter-spacing: -.035em; }
    p, label { color: #c4d2e3; }
    [data-testid="stCaptionContainer"] { color: var(--muted); }

    .eyebrow {
        color: var(--teal); text-transform: uppercase; letter-spacing: 2px;
        font-size: .68rem; font-weight: 800;
    }
    .hero {
        position: relative; overflow: hidden; isolation: isolate;
        padding: clamp(1.6rem, 4vw, 3.1rem); margin-bottom: 1.35rem;
        border: 1px solid rgba(102, 205, 210, .22); border-radius: 26px;
        background: linear-gradient(112deg, rgba(16, 34, 54, .98) 5%, rgba(12, 45, 59, .92) 62%, rgba(16, 61, 72, .86));
        box-shadow: 0 22px 60px rgba(0, 0, 0, .20);
    }
    .hero:after {
        content: ""; position: absolute; z-index: -1; width: 360px; height: 360px;
        right: -85px; top: -160px; border: 1px solid rgba(105, 230, 217, .18);
        border-radius: 50%; box-shadow: 0 0 0 35px rgba(105, 230, 217, .035), 0 0 0 75px rgba(105, 230, 217, .025);
    }
    .hero h1 { color: #f5f9ff; max-width: 760px; font-size: clamp(2.25rem, 4.4vw, 3.6rem); line-height: 1.04; letter-spacing: -.055em; margin: .65rem 0 .8rem; }
    .hero p { max-width: 680px; color: #b8c9dc; font-size: 1rem; line-height: 1.75; }
    .status-pill {
        display: inline-flex; align-items: center; border: 1px solid rgba(80, 224, 194, .3);
        border-radius: 999px; padding: .48rem .78rem; color: #8af0da;
        background: rgba(10, 93, 79, .25); font-size: .68rem; font-weight: 800; letter-spacing: .08em;
    }

    .section-head { color: var(--text); font-size: 1.12rem; font-weight: 780; margin: 1.3rem 0 .7rem; }
    .section-sub { color: var(--muted); font-size: .88rem; line-height: 1.6; margin: -.25rem 0 1rem; }
    .metric {
        position: relative; overflow: hidden; min-height: 132px; padding: 1.15rem 1.2rem;
        background: linear-gradient(145deg, rgba(19, 37, 59, .96), rgba(12, 27, 44, .96));
        border: 1px solid var(--border); border-radius: 18px;
        box-shadow: 0 12px 30px rgba(0, 0, 0, .12); transition: transform .18s ease, border-color .18s ease;
    }
    .metric:hover { transform: translateY(-3px); border-color: rgba(80, 224, 194, .35); }
    .metric-top { display: flex; justify-content: space-between; align-items: center; color: #a9bad0; font-size: .76rem; }
    .metric-icon {
        display: inline-grid; place-items: center; width: 29px; height: 29px;
        border-radius: 9px; color: var(--teal); background: rgba(80, 224, 194, .10);
        font-size: .7rem; font-weight: 850;
    }
    .metric-value { margin-top: .65rem; color: #f4f8ff; font-size: 1.8rem; font-weight: 820; letter-spacing: -.04em; }
    .metric-note { margin-top: .16rem; color: #8098b5; font-size: .7rem; }

    .panel, .prediction {
        padding: 1.25rem; margin-bottom: .8rem; border: 1px solid var(--border);
        border-radius: 19px; background: linear-gradient(145deg, rgba(17, 34, 54, .96), rgba(10, 24, 40, .96));
    }
    .prediction { padding: 1.65rem; border-color: rgba(80, 224, 194, .25); background: radial-gradient(ellipse at 100% 0%, rgba(42, 142, 145, .16), transparent 45%), linear-gradient(145deg, #10263a, #0c1b2c); }
    .prediction-label { color: #9bb2cb; text-transform: uppercase; letter-spacing: 1.35px; font-size: .68rem; font-weight: 750; }
    .prediction-name { color: var(--teal); font-size: clamp(1.8rem, 3vw, 2.55rem); font-weight: 850; margin: .35rem 0; letter-spacing: -.04em; }
    .prediction-score { color: #f5f8ff; font-size: 2.35rem; font-weight: 850; margin-top: .35rem; letter-spacing: -.05em; }
    .muted { color: var(--muted); font-size: .84rem; line-height: 1.7; }
    .prob-row { display: flex; align-items: center; justify-content: space-between; gap: 12px; color: #d5e1ef; font-size: .84rem; margin: 1rem 0 .4rem; }
    .prob-track { width: 100%; height: 7px; border-radius: 99px; background: #1c3047; overflow: hidden; }
    .prob-fill { height: 100%; border-radius: 99px; background: linear-gradient(90deg, #22a997, #67ead0); box-shadow: 0 0 14px rgba(80, 224, 194, .28); }

    .class-tile { padding: 1rem; min-height: 112px; border: 1px solid var(--border); border-radius: 16px; background: rgba(16, 32, 51, .86); }
    .class-symbol { display: inline-grid; place-items: center; width: 29px; height: 29px; border-radius: 9px; color: var(--teal); background: rgba(80, 224, 194, .10); font-size: .72rem; font-weight: 850; }
    .class-name { color: #edf4ff; font-size: .9rem; font-weight: 760; margin-top: .55rem; }
    .class-description { color: var(--muted); font-size: .73rem; margin-top: .2rem; }
    .notice { padding: 1rem 1.15rem; margin-top: 1.1rem; border: 1px solid rgba(217, 168, 91, .3); border-radius: 14px; background: rgba(80, 55, 20, .22); color: #e8d4a6; font-size: .82rem; line-height: 1.65; }
    .footer { border-top: 1px solid var(--border); margin-top: 2.5rem; padding-top: 1.1rem; color: #8296af; font-size: .74rem; text-align: center; }

    div[data-testid="stFileUploader"] { padding: .8rem; border: 1px dashed rgba(112, 159, 193, .42); border-radius: 16px; background: rgba(13, 29, 47, .72); }
    div[data-testid="stFileUploader"] section { border: 0; }
    div.stButton > button { min-height: 2.65rem; border: 1px solid #31516a; border-radius: 11px; color: #e8f2ff; font-weight: 720; transition: all .16s ease; }
    div.stButton > button:hover { border-color: rgba(80, 224, 194, .65); color: white; transform: translateY(-1px); }
    div.stButton > button[kind="primary"] { color: white; background: linear-gradient(100deg, #137f83, #159d93); border: 1px solid #50cfc1; }
    [data-testid="stMetric"] { padding: .9rem; border: 1px solid var(--border); border-radius: 14px; background: rgba(17, 34, 54, .85); }
    [data-testid="stExpander"] { border: 1px solid var(--border); border-radius: 14px; background: rgba(13, 28, 46, .65); }
    [data-testid="stTabs"] button { color: #8ea5c0; font-weight: 700; }
    [data-testid="stTabs"] button[aria-selected="true"] { color: var(--teal); }
    [data-testid="stDataFrame"] { border: 1px solid var(--border); border-radius: 14px; overflow: hidden; }
    [data-testid="stAlert"] { border-radius: 13px; }
    .launch-panel { padding: 1.55rem 1.7rem; border: 1px solid rgba(80, 224, 194, .22); border-radius: 19px; background: radial-gradient(ellipse at 95% 0%, rgba(48, 160, 159, .17), transparent 42%), linear-gradient(135deg, #111e31, #102a36); }
    .launch-kicker { color: #70e6d2; font-size: .68rem; font-weight: 800; letter-spacing: 1.7px; }
    .launch-title { margin: .45rem 0 .35rem; color: #f3f7ff; font-size: 1.35rem; font-weight: 800; }
    .launch-copy { max-width: 680px; margin: 0; color: #a9bdd4; font-size: .88rem; line-height: 1.6; }
    div.st-key-launch_mri_analysis button { min-height: 3.1rem; margin-top: .7rem; border-radius: 12px; box-shadow: 0 8px 22px rgba(20, 126, 128, .22); }

    @media (max-width: 700px) {
        .block-container { padding: 1rem .8rem 2rem; }
        .hero { padding: 1.4rem; border-radius: 20px; }
        .hero h1 { font-size: 2.2rem; }
        .metric { min-height: 112px; padding: .9rem; }
        .metric-value { font-size: 1.45rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)




@st.cache_resource
def get_model():
    model_path = MODEL_PATH

    if not model_path.is_file():
        if not MODEL_URL:
            raise FileNotFoundError(
                f"Model missing: {MODEL_PATH}. Set MODEL_URL to a trusted model file URL."
            )

        MODEL_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        if not MODEL_CACHE_PATH.is_file():
            headers = {}
            if MODEL_TOKEN:
                headers["Authorization"] = f"Bearer {MODEL_TOKEN}"

            request = Request(MODEL_URL, headers=headers)
            download_path = MODEL_CACHE_PATH.with_name(
                MODEL_CACHE_PATH.name + ".download"
            )
            try:
                with urlopen(request, timeout=120) as response:
                    with download_path.open("wb") as downloaded_model:
                        shutil.copyfileobj(response, downloaded_model)
                if download_path.stat().st_size == 0:
                    raise ValueError("The downloaded model file is empty.")
                download_path.replace(MODEL_CACHE_PATH)
            except Exception:
                download_path.unlink(missing_ok=True)
                raise

        model_path = MODEL_CACHE_PATH

    return tf.keras.models.load_model(model_path)


@st.cache_data
def get_report():
    if not REPORT_PATH.exists():
        return {}
    try:
        with open(REPORT_PATH, encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError):
        return {}


report = get_report()




# Initialize navigation state before rendering the radio widget.
if "active_page" not in st.session_state:
    st.session_state.active_page = "Overview"

def navigate_to(page):
    st.session_state.active_page = page

with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-mark">BT</div>
            <div>
                <div class="sidebar-title">BrainTumor AI</div>
                <div class="sidebar-caption">MRI intelligence platform</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    page = st.radio(
        "Workspace",
        ["Overview", "MRI Analysis", "Model Insights", "About"],
        key="active_page",
        label_visibility="visible",
    )

    st.divider()
    st.markdown("### Model Details")
    st.markdown("**MobileNetV2 - Fine-tuned**")
    st.caption("Input: 160 x 160 RGB")
    st.caption("Classes: 4")



def metric_card(title, value, note, symbol):
    st.markdown(
        f"""
        <div class="metric">
            <div class="metric-top">
                <span>{title}</span>
                <span class="metric-icon">{symbol}</span>
            </div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_class_tiles():
    cols = st.columns(4)
    descriptions = {
        "glioma": "Dataset class",
        "meningioma": "Dataset class",
        "no_tumor": "Dataset class",
        "pituitary": "Dataset class",
    
    }

    for col, name in zip(cols, CLASSES):
        with col:
            st.markdown(
                f"""
                <div class="class-tile">
                    <div class="class-symbol">{ICONS[name]}</div>
                    <div class="class-name">{LABELS[name]}</div>
                    <div class="class-description">
                        {descriptions[name]}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def run_prediction(uploaded_file, model):
    image = Image.open(uploaded_file).convert("RGB")
    resized = image.resize(IMG_SIZE)
    # Match training: model accepts RGB float pixels in the 0-255 range;
    # the model itself contains MobileNetV2 rescaling.
    array = np.asarray(resized, dtype=np.float32)[None, ...]

    probabilities = model.predict(array, verbose=0)[0]

    if (
        probabilities.shape != (len(CLASSES),)
        or not np.isfinite(probabilities).all()
    ):
        raise ValueError("Model returned invalid probabilities.")

    predicted_index = int(np.argmax(probabilities))

    return image, probabilities, CLASSES[predicted_index]


def render_prediction_workspace(model):
    st.markdown('<div class="section-head">MRI Analysis Workspace</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">Upload a single MRI image to inspect '
        'the model output and probability distribution.</div>',
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader(
        "Upload MRI image",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        key="mri_file",
        help="Supported formats: JPG, PNG, BMP and WEBP.",
    )

    if uploaded is None:
        st.markdown(
            """
            <div class="panel">
                <div class="eyebrow">READY FOR ANALYSIS</div>
                <h3 style="color:#edf4ff">Your MRI workspace is ready</h3>
                <p class="muted">
                    Choose an image above to display the original scan,
                    predicted dataset class and model probability scores.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    try:
        image, probs, predicted = run_prediction(uploaded, model)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        st.error(f"Unable to process the uploaded image: {exc}")
        return
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        return

    st.markdown("---")

    left, right = st.columns([1.05, 1], gap="large")

    with left:
        st.markdown('<div class="section-head">Image Preview</div>',
                    unsafe_allow_html=True)
        st.image(image, use_container_width=True)

        with st.expander("Image details"):
            st.write(f"**File:** {uploaded.name}")
            st.write(f"**Dimensions:** {image.width} x {image.height}")
            st.write("**Color format:** RGB")
            st.write(f"**Model input:** {IMG_SIZE[0]} x {IMG_SIZE[1]}")

    with right:
        st.markdown('<div class="section-head">Analysis Result</div>',
                    unsafe_allow_html=True)

        confidence = float(probs[CLASSES.index(predicted)])

        st.markdown(
            f"""
            <div class="prediction">
                <div class="prediction-label">Predicted dataset class</div>
                <div class="prediction-name">{LABELS[predicted]}</div>
                <div class="prediction-label">Top-class probability</div>
                <div class="prediction-score">{confidence:.2%}</div>
                <p class="muted">
                    A model probability is not a medical diagnosis or
                    a calibrated measure of certainty.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("#### Class probability distribution")

        for idx in np.argsort(probs)[::-1]:
            name = CLASSES[int(idx)]
            value = float(probs[idx])
            percent = max(0.0, min(100.0, value * 100))

            st.markdown(
                f"""
                <div class="prob-row">
                    <span>{LABELS[name]}</span>
                    <strong>{percent:.2f}%</strong>
                </div>
                <div class="prob-track">
                    <div class="prob-fill" style="width:{percent}%"></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown(
        """
        <div class="notice">
            <strong>Research use only.</strong> This application has not
            been clinically validated. Do not use its output to diagnose
            or rule out disease, or to make treatment decisions. Please
            consult a qualified medical professional.
        </div>
        """,
        unsafe_allow_html=True,
    )




if st.session_state.active_page == "Overview":

    st.markdown(
        """
        <div class="hero">
            <div class="eyebrow">BRAIN MRI | AI ANALYTICS</div>
            <h1>Clarity for every scan.</h1>
            <p>
                Explore the model's evaluation, understand the dataset, and
                inspect MRI classification outputs in one focused workspace.
            </p>
            <span class="status-pill">TRANSFER LEARNING | MOBILENETV2</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Load recorded evaluation results
    accuracy = report.get("test_accuracy")
    macro_f1 = report.get("test_macro_f1")
    macro_recall = report.get("test_macro_recall")
    test_images = report.get("test_images", 246)

    # Model status
    try:
        model = get_model()
        model_status = "Online"
    except Exception:
        model = None
        model_status = "Unavailable"

    # KPI cards
    st.markdown(
        '<div class="section-head">Performance at a glance</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4, gap="medium")

    with c1:
        metric_card(
            "Model status",
            model_status,
            "Saved checkpoint",
            "OK",
        )

    with c2:
        metric_card(
            "Test accuracy",
            f"{accuracy:.2%}" if isinstance(accuracy, (int, float)) else "N/A",
            "Held-out test set",
            "A",
        )

    with c3:
        metric_card(
            "Macro F1-score",
            f"{macro_f1:.2%}" if isinstance(macro_f1, (int, float)) else "N/A",
            "Four-class evaluation",
            "F1",
        )

    with c4:
        metric_card(
            "Test images",
            f"{test_images:,}" if isinstance(test_images, int) else "N/A",
            "Evaluation samples",
            "#",
        )

    st.markdown("---")

    # Interactive tabs
    overview_tab, dataset_tab, model_tab = st.tabs(
        ["Overview", "Dataset Explorer", "Model Details"]
    )

    with overview_tab:
        left, right = st.columns([1.15, 1], gap="large")

        with left:
            st.markdown(
                '<div class="section-head">Evaluation metrics</div>',
                unsafe_allow_html=True,
            )

            if all(
                isinstance(x, (int, float))
                for x in [accuracy, macro_f1, macro_recall]
            ):
                st.bar_chart(
                    {
                        "Score": {
                            "Accuracy": accuracy,
                            "Macro F1": macro_f1,
                            "Macro Recall": macro_recall,
                        }
                    },
                    y="Score",
                    height=300,
                )
                st.caption(
                    "Metrics are from the previously recorded final "
                    "test evaluation."
                )
            else:
                st.info("Evaluation metrics are not available.")

        with right:
            st.markdown(
                '<div class="section-head">Classification classes</div>',
                unsafe_allow_html=True,
            )

            class_data = {
                "Class": [
                    "Glioma",
                    "Meningioma",
                    "No Tumor",
                    "Pituitary",
                ],
                "Training images": [564, 358, 335, 438],
                "Validation images": [161, 124, 99, 118],
                "Test images": [80, 63, 49, 54],
            }

            import pandas as pd

            class_df = pd.DataFrame(class_data)
            st.dataframe(
                class_df,
                hide_index=True,
                use_container_width=True,
            )

        st.caption("Dataset totals: 1,695 train | 502 validation | 246 test")

    with dataset_tab:
        st.markdown(
            '<div class="section-head">Explore dataset composition</div>',
            unsafe_allow_html=True,
        )

        import pandas as pd

        dataset_df = pd.DataFrame({
            "Class": ["Glioma", "Meningioma", "No Tumor", "Pituitary"],
            "Train": [564, 358, 335, 438],
            "Validation": [161, 124, 99, 118],
            "Test": [80, 63, 49, 54],
        })

        split = st.selectbox(
            "Choose dataset split",
            ["Train", "Validation", "Test", "All splits"],
            key="dataset_split_explorer",
        )

        if split == "All splits":
            chart_df = dataset_df.set_index("Class")[[
                "Train", "Validation", "Test"
            ]]
            st.bar_chart(chart_df, height=360)
        else:
            chart_df = dataset_df[["Class", split]].set_index("Class")
            st.bar_chart(chart_df, height=360)

        st.markdown("#### Class distribution table")
        st.dataframe(dataset_df, hide_index=True, use_container_width=True)

        totals = {
            "Train": 1695,
            "Validation": 502,
            "Test": 246,
        }

        m1, m2, m3 = st.columns(3)

        with m1:
            st.metric("Training", f"{totals['Train']:,} images")
        with m2:
            st.metric("Validation", f"{totals['Validation']:,} images")
        with m3:
            st.metric("Testing", f"{totals['Test']:,} images")

    with model_tab:
        st.markdown(
            '<div class="section-head">Model configuration</div>',
            unsafe_allow_html=True,
        )

        config_left, config_right = st.columns(2)

        with config_left:
            st.markdown("#### Architecture")
            st.write("MobileNetV2")
            st.write("Fine-tuned transfer-learning model")
            st.write("Framework: TensorFlow / Keras")

        with config_right:
            st.markdown("#### Inference settings")
            st.write("Input: 160 x 160 pixels")
            st.write("Color format: RGB")
            st.write("Output classes: 4")
            st.write("Execution: Local model checkpoint")

        st.info(
            "The model status only confirms that the checkpoint loads. "
            "It does not establish clinical readiness."
        )

    # MRI launch call-to-action
    st.markdown("---")

    st.markdown(
        """
        <div class="launch-panel">
            <div class="launch-kicker">MRI ANALYSIS WORKSPACE</div>
            <div class="launch-title">Ready to analyze an MRI?</div>
            <p class="launch-copy">
                Upload a scan to explore the model's predicted class and
                four-class probability distribution.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.button(
        "Open MRI analysis",
        key="launch_mri_analysis",
        type="primary",
        use_container_width=True,
        on_click=navigate_to,
        args=("MRI Analysis",),
    )



elif st.session_state.active_page == "MRI Analysis":
    st.markdown(
        """
        <div class="eyebrow">IMAGE WORKSPACE</div>
        <h1 style="color:#f3f7ff;margin:.35rem 0">MRI Analysis</h1>
        <p class="muted">
            Upload an image to run inference with the saved model.
        </p>
        """,
        unsafe_allow_html=True,
    )

    try:
        model = get_model()
        render_prediction_workspace(model)
    except Exception as exc:
        st.error(f"Unable to initialize the model: {exc}")




elif st.session_state.active_page == "Model Insights":
    st.markdown(
        """
        <div class="eyebrow">MODEL PERFORMANCE</div>
        <h1 style="color:#f3f7ff;margin:.35rem 0">Model Insights</h1>
        <p class="muted">
            Previously recorded evaluation metrics for the selected model.
        </p>
        """,
        unsafe_allow_html=True,
    )

    if report:
        a, b, c = st.columns(3)

        with a:
            metric_card(
                "Test accuracy",
                f"{report['test_accuracy']:.2%}",
                "Held-out test set",
                "A",
            )
        with b:
            metric_card(
                "Macro F1-score",
                f"{report['test_macro_f1']:.2%}",
                "Mean class F1",
                "F1",
            )
        with c:
            metric_card(
                "Macro recall",
                f"{report['test_macro_recall']:.2%}",
                "Mean class recall",
                "OK",
            )

        st.markdown('<div class="section-head">Evaluation details</div>',
                    unsafe_allow_html=True)
        st.write(f"**Model:** {report.get('model', 'Not recorded')}")
        st.write(f"**Test images:** {report.get('test_images', 'N/A')}")
        st.write("**Class order:** " + ", ".join(CLASSES))

        st.markdown('<div class="section-head">How to read these scores</div>',
                    unsafe_allow_html=True)
        score_cols = st.columns(3, gap="medium")
        score_explanations = [
            (
                "Accuracy",
                "Share of test images assigned to the correct dataset class.",
            ),
            (
                "Macro F1",
                "Average F1 across classes, giving each class equal weight.",
            ),
            (
                "Macro recall",
                "Average fraction of examples found for each class.",
            ),
        ]
        for col, (title, copy) in zip(score_cols, score_explanations):
            with col:
                st.markdown(
                    f'<div class="panel"><div class="eyebrow">{title}</div>'
                    f'<p class="muted">{copy}</p></div>',
                    unsafe_allow_html=True,
                )

        st.markdown(
            '<div class="muted">These metrics summarize one held-out '
            'evaluation set. They do not guarantee performance on scans '
            'from other sources or settings.</div>',
            unsafe_allow_html=True,
        )

        if CM_PATH.exists():
            st.markdown('<div class="section-head">Test confusion matrix</div>',
                        unsafe_allow_html=True)
            st.markdown(
                '<p class="muted">The matrix compares actual labels with '
                'predicted labels. Values on the diagonal are correct '
                'classifications; off-diagonal values show where the model '
                'confused two classes.</p>',
                unsafe_allow_html=True,
            )
            st.image(
                str(CM_PATH),
                caption="Saved final test-set confusion matrix",
                use_container_width=True,
            )
        else:
            st.info(
                "Confusion matrix image not found. Run the evaluation "
                "notebook to generate it."
            )
    else:
        st.warning(
            "Evaluation report not found. Expected file: "
            "reports/final_test_results.json"
        )




elif st.session_state.active_page == "About":
    st.markdown(
        """
        <div class="eyebrow">PROJECT INFORMATION</div>
        <h1 style="color:#f3f7ff;margin:.35rem 0">About BrainTumor AI</h1>
        <p class="muted">
            An educational image-classification project built with
            TensorFlow, MobileNetV2 and Streamlit.
        </p>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-head">Technical overview</div>',
                unsafe_allow_html=True)

    details = [
        ("Architecture", "MobileNetV2 transfer learning"),
        ("Input image", "160 x 160 pixels, RGB"),
        ("Framework", "TensorFlow / Keras"),
        ("Interface", "Streamlit"),
        ("Output", "Four dataset classes"),
    ]

    for title, value in details:
        st.markdown(
            f"""
            <div class="panel" style="padding:.85rem 1rem">
                <span style="color:#9cafc6">{title}</span>
                <span style="float:right;color:#edf4ff;font-weight:700">
                    {value}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-head">How the project works</div>',
                unsafe_allow_html=True)
    workflow = [
        ("01", "Prepare", "MRI images are organized into train, validation, and test splits."),
        ("02", "Train", "A MobileNetV2 backbone is fine-tuned to predict the four dataset labels."),
        ("03", "Explore", "Upload an image to view the model's class scores and inspect recorded evaluation results."),
    ]
    workflow_cols = st.columns(3, gap="medium")
    for col, (step, title, copy) in zip(workflow_cols, workflow):
        with col:
            st.markdown(
                f'<div class="panel"><div class="class-symbol">{step}</div>'
                f'<div class="class-name">{title}</div>'
                f'<div class="class-description">{copy}</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown('<div class="section-head">Dataset labels</div>',
                unsafe_allow_html=True)
    label_cols = st.columns(4, gap="small")
    for col, class_name in zip(label_cols, CLASSES):
        with col:
            st.markdown(
                f'<div class="class-tile"><div class="class-symbol">'
                f'{ICONS[class_name]}</div><div class="class-name">'
                f'{LABELS[class_name]}</div><div class="class-description">'
                'Model output category</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown('<div class="section-head">What this prototype is for</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="panel"><p class="muted">This project is designed '
        'for learning and experimentation with image classification, '
        'transfer learning, and model evaluation. It provides a simple way '
        'to explore how a saved model responds to an image and how its '
        'test-set metrics are presented.</p></div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="notice">
            <strong>Medical disclaimer:</strong> This project is an
            experimental classifier and is not a validated medical
            device. Its predictions must not replace professional
            interpretation of medical images.
        </div>
        """,
        unsafe_allow_html=True,
    )




st.markdown(
    """
    <div class="footer">
        BrainTumor AI | Research & Education | TensorFlow + Streamlit<br>
        Experimental predictions only - not for clinical decision-making.
    </div>
    """,
    unsafe_allow_html=True,
)









