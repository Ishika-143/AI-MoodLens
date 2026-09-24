import streamlit as st
import cv2
import numpy as np
from PIL import Image
import onnxruntime as ort
from pathlib import Path
import time


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI MoodLens",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# PREMIUM UI
# ============================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: "DM Sans", sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 10% 5%, rgba(139,92,246,0.15), transparent 25%),
        radial-gradient(circle at 90% 10%, rgba(59,130,246,0.12), transparent 25%),
        radial-gradient(circle at 50% 100%, rgba(124,58,237,0.10), transparent 35%),
        #080711;
    color: #f5f3ff;
}

.block-container {
    max-width: 1250px;
    padding-top: 3rem;
    padding-bottom: 4rem;
}

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    background: transparent !important;
}

/* Main title */

h1 {
    font-family: "Space Grotesk", sans-serif !important;
    font-size: 4.2rem !important;
    font-weight: 700 !important;
    letter-spacing: -2px;
    background: linear-gradient(
        90deg,
        #ffffff,
        #c4b5fd,
        #93c5fd
    );
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.3rem !important;
}

h2 {
    font-family: "Space Grotesk", sans-serif !important;
    color: #ede9fe !important;
}

h3 {
    font-family: "Space Grotesk", sans-serif !important;
    color: #ddd6fe !important;
}

p {
    color: #aaa4c5;
}

hr {
    border: none !important;
    height: 1px !important;
    background: linear-gradient(
        90deg,
        transparent,
        rgba(139,92,246,0.45),
        transparent
    ) !important;
    margin: 2rem 0 !important;
}

/* Buttons */

.stButton > button {
    width: 100%;
    min-height: 50px;
    border-radius: 15px;
    border: 1px solid rgba(167,139,250,0.3);
    background: linear-gradient(
        135deg,
        #7c3aed,
        #2563eb
    );
    color: white;
    font-weight: 700;
    font-size: 15px;
    box-shadow: 0 10px 35px rgba(124,58,237,0.25);
    transition: all 0.25s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 15px 45px rgba(124,58,237,0.38);
}

/* Camera / uploader */

[data-testid="stFileUploader"] {
    background: rgba(20,18,34,0.75);
    border: 1px solid rgba(139,92,246,0.20);
    border-radius: 22px;
    padding: 12px;
}

[data-testid="stFileUploaderDropzone"] {
    background: rgba(10,9,20,0.7) !important;
    border: 1px dashed rgba(167,139,250,0.35) !important;
    border-radius: 16px !important;
}

/* Containers */

[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(17,15,29,0.72);
    border: 1px solid rgba(139,92,246,0.15);
    border-radius: 22px;
    box-shadow:
        inset 0 1px 0 rgba(255,255,255,0.025),
        0 20px 60px rgba(0,0,0,0.22);
}

/* Metrics */

[data-testid="stMetric"] {
    background: rgba(16,14,29,0.78);
    border: 1px solid rgba(139,92,246,0.15);
    padding: 18px;
    border-radius: 18px;
}

[data-testid="stMetricLabel"] {
    color: #a8a0c0 !important;
}

[data-testid="stMetricValue"] {
    color: #f5f3ff !important;
    font-family: "Space Grotesk", sans-serif;
}

/* Progress */

.stProgress > div > div > div > div {
    background: linear-gradient(
        90deg,
        #8b5cf6,
        #3b82f6
    );
    border-radius: 20px;
}

.stProgress > div > div {
    background: rgba(255,255,255,0.07);
    border-radius: 20px;
}

/* Alerts */

[data-testid="stAlert"] {
    border-radius: 16px !important;
    border: 1px solid rgba(139,92,246,0.18) !important;
    background: rgba(17,15,29,0.75) !important;
}

/* Images */

[data-testid="stImage"] img {
    border-radius: 20px;
    border: 1px solid rgba(139,92,246,0.18);
    box-shadow: 0 20px 60px rgba(0,0,0,0.3);
}

/* Tabs */

button[data-baseweb="tab"] {
    color: #9f97b7 !important;
    font-weight: 600;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: #c4b5fd !important;
}

[data-baseweb="tab-highlight"] {
    background: #8b5cf6 !important;
}

@media (max-width: 768px) {
    h1 {
        font-size: 2.8rem !important;
    }

    .block-container {
        padding-top: 1.5rem;
    }
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "facial_expression_recognition_mobilefacenet_2022july.onnx"
)


# ============================================================
# EMOTIONS
# ============================================================

EMOTIONS = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "sad",
    "surprise",
    "neutral"
]


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():

    # --------------------------------------------------------
    # Face detector
    # --------------------------------------------------------

    cascade_file = (
        Path(cv2.__file__).resolve().parent
        / "data"
        / "haarcascade_frontalface_default.xml"
    )

    if not cascade_file.exists():

        # fallback location used by OpenCV installations
        cascade_file = (
            Path(cv2.data.haarcascades)
            / "haarcascade_frontalface_default.xml"
        )

    if not cascade_file.exists():
        raise FileNotFoundError(
            "Haar Cascade file was not found. "
            "Please check OpenCV installation."
        )

    face_detector = cv2.CascadeClassifier(
        str(cascade_file)
    )

    if face_detector.empty():
        raise RuntimeError(
            "OpenCV could not load the Haar Cascade file."
        )

    # --------------------------------------------------------
    # Emotion model
    # --------------------------------------------------------

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Emotion model not found:\n{MODEL_PATH}"
        )

    expression_model = ort.InferenceSession(
        str(MODEL_PATH),
        providers=["CPUExecutionProvider"]
    )

    return face_detector, expression_model


# ============================================================
# LOAD MODEL SAFELY
# ============================================================

try:

    face_detector, expression_model = load_models()

except Exception as e:

    st.error("⚠️ AI model setup failed.")

    st.code(str(e))

    st.stop()


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_face(face):

    face = cv2.cvtColor(
        face,
        cv2.COLOR_BGR2GRAY
    )

    face = cv2.resize(
        face,
        (112, 112)
    )

    face = face.astype(
        np.float32
    ) / 255.0

    face = (face - 0.5) / 0.5

    face = np.expand_dims(
        face,
        axis=0
    )

    face = np.expand_dims(
        face,
        axis=0
    )

    return face.astype(np.float32)


# ============================================================
# EMOTION PREDICTION
# ============================================================

def predict_emotion(face):

    input_name = expression_model.get_inputs()[0].name
    output_name = expression_model.get_outputs()[0].name

    input_tensor = preprocess_face(face)

    outputs = expression_model.run(
        [output_name],
        {
            input_name: input_tensor
        }
    )

    raw_output = np.asarray(
        outputs[0]
    ).squeeze()

    # Softmax
    exp_values = np.exp(
        raw_output - np.max(raw_output)
    )

    probabilities = (
        exp_values /
        np.sum(exp_values)
    )

    emotion_index = int(
        np.argmax(probabilities)
    )

    emotion = EMOTIONS[
        emotion_index
    ]

    confidence = float(
        probabilities[emotion_index]
    )

    all_predictions = [
        (EMOTIONS[i], float(probabilities[i]))
        for i in range(
            len(EMOTIONS)
        )
    ]

    all_predictions.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return (
        emotion,
        confidence,
        all_predictions
    )


# ============================================================
# EMOTION DISPLAY
# ============================================================

EMOTION_INFO = {

    "happy": {
        "emoji": "😊",
        "title": "Happy",
        "description": "The facial features show patterns associated with a positive emotional expression."
    },

    "sad": {
        "emoji": "😔",
        "title": "Sad",
        "description": "The facial features show patterns associated with a subdued emotional expression."
    },

    "angry": {
        "emoji": "😠",
        "title": "Angry",
        "description": "The facial features show patterns associated with an intense emotional expression."
    },

    "fear": {
        "emoji": "😨",
        "title": "Fear",
        "description": "The facial features show patterns associated with an alert or fearful expression."
    },

    "surprise": {
        "emoji": "😮",
        "title": "Surprised",
        "description": "The facial features show patterns associated with a surprised expression."
    },

    "disgust": {
        "emoji": "🤢",
        "title": "Disgust",
        "description": "The facial features show patterns associated with a disgusted expression."
    },

    "neutral": {
        "emoji": "😐",
        "title": "Neutral",
        "description": "The facial features show patterns associated with a neutral expression."
    }
}


# ============================================================
# HERO
# ============================================================

st.markdown(
    "### AI-POWERED COMPUTER VISION",
)

st.title("AI MoodLens")

st.subheader(
    "See the emotion behind the expression."
)

st.write(
    "MoodLens uses computer vision and machine learning "
    "to analyze facial expressions and estimate the "
    "most likely emotional state."
)

st.write("")

metric1, metric2, metric3 = st.columns(3)

with metric1:
    st.metric(
        "🧠 Technology",
        "Computer Vision"
    )

with metric2:
    st.metric(
        "⚡ Processing",
        "ONNX + CPU"
    )

with metric3:
    st.metric(
        "🎭 Emotions",
        "7 Classes"
    )

st.divider()


# ============================================================
# INPUT MODE
# ============================================================

st.subheader("🎥 Start Emotion Analysis")

mode = st.radio(
    "Choose analysis method",
    [
        "📷 Camera",
        "🖼️ Upload Image"
    ],
    horizontal=True
)


# ============================================================
# GET IMAGE
# ============================================================

image = None


if mode == "📷 Camera":

    camera_image = st.camera_input(
        "Take a picture"
    )

    if camera_image is not None:

        image = Image.open(
            camera_image
        ).convert("RGB")


else:

    uploaded_file = st.file_uploader(
        "Upload a face image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ]
    )

    if uploaded_file is not None:

        image = Image.open(
            uploaded_file
        ).convert("RGB")


# ============================================================
# ANALYSIS
# ============================================================

if image is not None:

    st.write("")

    preview_col, control_col = st.columns(
        [1.25, 0.9],
        gap="large"
    )

    with preview_col:

        st.markdown("#### 📸 Input Image")

        st.image(
            image,
            use_container_width=True
        )

    with control_col:

        st.markdown("#### 🧠 AI Ready")

        st.write(
            "MoodLens will locate a face, analyze "
            "facial features and classify the "
            "most likely emotional expression."
        )

        st.write("")

        analyze_button = st.button(
            "✨ Analyze Emotion",
            use_container_width=True
        )


    # ========================================================
    # RUN ANALYSIS
    # ========================================================

    if analyze_button:

        st.divider()

        st.subheader(
            "🔍 AI Analysis"
        )

        progress = st.progress(0)

        status = st.empty()

        analysis_steps = [
            ("📷 Processing image...", 20),
            ("👤 Detecting facial features...", 40),
            ("🧬 Extracting expression patterns...", 60),
            ("🧠 Running emotion model...", 80),
            ("✨ Preparing result...", 100)
        ]

        for message, value in analysis_steps:

            status.info(message)

            progress.progress(
                value
            )

            time.sleep(0.35)


        # ----------------------------------------------------
        # Convert PIL -> OpenCV
        # ----------------------------------------------------

        image_array = np.asarray(
            image
        )

        image_bgr = cv2.cvtColor(
            image_array,
            cv2.COLOR_RGB2BGR
        )

        gray = cv2.cvtColor(
            image_bgr,
            cv2.COLOR_BGR2GRAY
        )


        # ----------------------------------------------------
        # Face detection
        # ----------------------------------------------------

        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60)
        )


        if len(faces) == 0:

            status.error(
                "No clear face detected."
            )

            st.warning(
                "Please try another image with a clear, "
                "front-facing face and better lighting."
            )

            st.stop()


        # ----------------------------------------------------
        # Select largest face
        # ----------------------------------------------------

        largest_face = max(
            faces,
            key=lambda box: box[2] * box[3]
        )

        x, y, w, h = largest_face

        face_crop = image_bgr[
            y:y+h,
            x:x+w
        ]


        # ----------------------------------------------------
        # Predict
        # ----------------------------------------------------

        emotion, confidence, predictions = (
            predict_emotion(
                face_crop
            )
        )


        # ----------------------------------------------------
        # Draw face box
        # ----------------------------------------------------

        annotated = image_bgr.copy()

        cv2.rectangle(
            annotated,
            (x, y),
            (x+w, y+h),
            (160, 100, 240),
            3
        )

        annotated_rgb = cv2.cvtColor(
            annotated,
            cv2.COLOR_BGR2RGB
        )

        result_image = Image.fromarray(
            annotated_rgb
        )


        status.success(
            "✓ Analysis complete"
        )

        time.sleep(0.3)


        # ====================================================
        # RESULT
        # ====================================================

        st.divider()

        st.subheader(
            "🎭 Emotion Detected"
        )

        info = EMOTION_INFO.get(
            emotion,
            EMOTION_INFO["neutral"]
        )


        result1, result2, result3 = st.columns(
            3,
            gap="medium"
        )

        with result1:

            st.metric(
                "Detected Emotion",
                f"{info['emoji']} {info['title']}"
            )

        with result2:

            st.metric(
                "Model Confidence",
                f"{confidence * 100:.1f}%"
            )

        with result3:

            st.metric(
                "Faces Detected",
                str(len(faces))
            )


        st.write("")

        st.success(
            f"{info['emoji']} {info['title']}: "
            f"{info['description']}"
        )


        # ====================================================
        # RESULT IMAGE + CONFIDENCE
        # ====================================================

        result_image_col, confidence_col = st.columns(
            [1.15, 1],
            gap="large"
        )

        with result_image_col:

            st.markdown(
                "#### 👤 Face Analysis"
            )

            st.image(
                result_image,
                use_container_width=True
            )

        with confidence_col:

            st.markdown(
                "#### 🎯 Detection Confidence"
            )

            st.write(
                "The confidence value represents the "
                "model's relative probability for its "
                "top predicted emotion."
            )

            st.progress(
                confidence
            )

            st.metric(
                "Top Prediction",
                f"{confidence * 100:.1f}%"
            )


        # ====================================================
        # TOP PREDICTIONS
        # ====================================================

        st.divider()

        st.subheader(
            "📊 Emotion Probability Breakdown"
        )

        for rank, (label, probability) in enumerate(
            predictions,
            start=1
        ):

            emotion_info = EMOTION_INFO.get(
                label,
                EMOTION_INFO["neutral"]
            )

            percentage = (
                probability * 100
            )

            st.write(
                f"**{rank}. "
                f"{emotion_info['emoji']} "
                f"{emotion_info['title']}** — "
                f"{percentage:.1f}%"
            )

            st.progress(
                probability
            )


        # ====================================================
        # INSIGHT
        # ====================================================

        st.divider()

        st.subheader(
            "💡 AI Insight"
        )

        insight_col1, insight_col2 = st.columns(
            2,
            gap="large"
        )

        with insight_col1:

            st.info(
                f"The strongest detected expression is "
                f"**{info['title']}**, with a model confidence "
                f"of **{confidence * 100:.1f}%**."
            )

        with insight_col2:

            st.warning(
                "Facial-expression classification is an "
                "AI estimate based on visible facial patterns. "
                "It does not determine a person's actual feelings "
                "or mental state."
            )


        # ====================================================
        # HOW IT WORKS
        # ====================================================

        st.divider()

        st.subheader(
            "⚙️ How MoodLens Works"
        )

        step1, step2, step3, step4 = st.columns(4)

        with step1:

            st.markdown("### 01")
            st.write("Image Input")
            st.caption(
                "Camera or uploaded image"
            )

        with step2:

            st.markdown("### 02")
            st.write("Face Detection")
            st.caption(
                "Computer vision locates the face"
            )

        with step3:

            st.markdown("### 03")
            st.write("ML Classification")
            st.caption(
                "ONNX model analyzes expression patterns"
            )

        with step4:

            st.markdown("### 04")
            st.write("Emotion Result")
            st.caption(
                "Top emotion and probabilities"
            )


# ============================================================
# EMPTY STATE
# ============================================================

else:

    st.write("")

    left, right = st.columns(
        2,
        gap="large"
    )

    with left:

        st.markdown(
            "### 🔬 What MoodLens analyzes"
        )

        st.write(
            "MoodLens is designed to recognize seven "
            "facial-expression categories:"
        )

        st.write(
            "😊 Happy  •  😔 Sad  •  😠 Angry  •  "
            "😨 Fear"
        )

        st.write(
            "😮 Surprise  •  🤢 Disgust  •  😐 Neutral"
        )

    with right:

        st.markdown(
            "### 📷 For better results"
        )

        st.write(
            "• Keep your face clearly visible"
        )

        st.write(
            "• Use good lighting"
        )

        st.write(
            "• Face the camera directly"
        )

        st.write(
            "• Avoid heavily blurred images"
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "AI MoodLens • Computer Vision + Machine Learning • "
    "Educational Project"
)

st.caption(
    "⚠️ MoodLens provides an AI-based facial-expression "
    "classification and should not be treated as a definitive "
    "assessment of a person's emotions."
)