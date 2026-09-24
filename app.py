import streamlit as st
import cv2
import numpy as np
from pathlib import Path
import time

# =========================================================
# AI MOODLENS
# Python + OpenCV + Machine Learning + ONNX
# =========================================================

st.set_page_config(
    page_title="AI MoodLens",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).parent

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "facial_expression_recognition_mobilefacenet_2022july.onnx"
)

# =========================================================
# EMOTIONS
# =========================================================

EMOTIONS = [
    "angry",
    "disgust",
    "fearful",
    "happy",
    "neutral",
    "sad",
    "surprised"
]

EMOTION_EMOJI = {
    "angry": "😠",
    "disgust": "🤢",
    "fearful": "😨",
    "happy": "😊",
    "neutral": "😐",
    "sad": "😔",
    "surprised": "😮"
}

# =========================================================
# SESSION HISTORY
# =========================================================

if "emotion_history" not in st.session_state:
    st.session_state.emotion_history = []

if "confidence_history" not in st.session_state:
    st.session_state.confidence_history = []

if "scan_count" not in st.session_state:
    st.session_state.scan_count = 0

if "last_emotion" not in st.session_state:
    st.session_state.last_emotion = "None"

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    .stApp {
        background:
            radial-gradient(
                circle at 10% 10%,
                rgba(124, 58, 237, 0.15),
                transparent 25%
            ),
            radial-gradient(
                circle at 90% 10%,
                rgba(14, 165, 233, 0.12),
                transparent 25%
            ),
            #080912;
    }

    .main .block-container {
        max-width: 1450px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    h1 {
        font-size: 3.8rem !important;
        font-weight: 800 !important;
        letter-spacing: -2px;
        text-align: center;
    }

    h2, h3 {
        font-weight: 750 !important;
    }

    [data-testid="stMetric"] {
        background: rgba(255,255,255,0.035);
        border: 1px solid rgba(255,255,255,0.08);
        padding: 18px;
        border-radius: 16px;
    }

    hr {
        border-color: rgba(255,255,255,0.08);
    }

    </style>
    """,
    unsafe_allow_html=True
)

# =========================================================
# LOAD MODELS
# =========================================================

@st.cache_resource
def load_models():

    cascade_file = (
        cv2.data.haarcascades
        + "haarcascade_frontalface_default.xml"
    )

    face_detector = cv2.CascadeClassifier(
        cascade_file
    )

    if face_detector.empty():
        return None, None

    if not MODEL_PATH.exists():
        return face_detector, None

    expression_model = cv2.dnn.readNet(
        str(MODEL_PATH)
    )

    return face_detector, expression_model


face_detector, expression_model = load_models()

# =========================================================
# MODEL CHECK
# =========================================================

if face_detector is None:
    st.error("OpenCV face detector could not be loaded.")
    st.stop()

if expression_model is None:
    st.error("Facial expression ONNX model was not found.")
    st.info(
        "Required file:\n\n"
        "models/facial_expression_recognition_mobilefacenet_2022july.onnx"
    )
    st.stop()

# =========================================================
# HERO
# =========================================================

st.write("")

st.caption(
    "✦ COMPUTER VISION  •  MACHINE LEARNING  •  ONNX"
)

st.title("🧠 AI MoodLens")

st.markdown(
    """
    <div style="text-align:center;">
    <p style="color:#9295a8; font-size:1.05rem;">
    Understand visible facial expressions using
    AI-powered computer vision.
    </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.write("")

status1, status2, status3 = st.columns(3)

with status1:
    st.success("● Camera System Ready")

with status2:
    st.info("● AI Model Loaded")

with status3:
    st.success("● Detection Online")

st.divider()

# =========================================================
# MAIN WORKSPACE
# =========================================================

left, right = st.columns(
    [1.2, 0.95],
    gap="large"
)

# =========================================================
# CAMERA
# =========================================================

with left:

    st.subheader("📷 Live Expression Scanner")

    st.caption(
        "Capture a clear image of your face to begin AI analysis."
    )

    camera_image = st.camera_input(
        "Take a picture",
        label_visibility="collapsed"
    )

    if camera_image is None:

        st.info(
            "📸 Camera is ready. "
            "Take a picture to analyze your expression."
        )

# =========================================================
# AI ANALYSIS
# =========================================================

with right:

    st.subheader("🤖 AI Analysis")

    st.caption(
        "Facial expression classification powered by machine learning."
    )

    # -----------------------------------------------------
    # WAITING
    # -----------------------------------------------------

    if camera_image is None:

        st.metric(
            "System Status",
            "READY"
        )

        st.info(
            "Waiting for camera input..."
        )

        st.write(
            "The system will detect the largest visible "
            "face and classify its expression."
        )

    # -----------------------------------------------------
    # IMAGE RECEIVED
    # -----------------------------------------------------

    else:

        image_bytes = camera_image.getvalue()

        image_array = np.frombuffer(
            image_bytes,
            dtype=np.uint8
        )

        frame = cv2.imdecode(
            image_array,
            cv2.IMREAD_COLOR
        )

        if frame is None:

            st.error(
                "Unable to read the captured image."
            )

            st.stop()

        # -------------------------------------------------
        # FACE DETECTION
        # -------------------------------------------------

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80)
        )

        # -------------------------------------------------
        # NO FACE
        # -------------------------------------------------

        if len(faces) == 0:

            st.warning(
                "🔍 No face detected."
            )

            st.write(
                "Please face the camera directly, "
                "make sure your face is well lit, "
                "and try again."
            )

        # -------------------------------------------------
        # FACE FOUND
        # -------------------------------------------------

        else:

            x, y, w, h = max(
                faces,
                key=lambda box: box[2] * box[3]
            )

            face = frame[
                y:y + h,
                x:x + w
            ]

            if face.size == 0:

                st.error(
                    "Could not process the detected face."
                )

                st.stop()

            # -------------------------------------------------
            # DETECTION IMAGE
            # -------------------------------------------------

            display_frame = frame.copy()

            cv2.rectangle(
                display_frame,
                (x, y),
                (x + w, y + h),
                (145, 92, 246),
                3
            )

            cv2.putText(
                display_frame,
                "FACE DETECTED",
                (x, max(30, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (145, 92, 246),
                2
            )

            display_rgb = cv2.cvtColor(
                display_frame,
                cv2.COLOR_BGR2RGB
            )

            st.image(
                display_rgb,
                caption="Computer Vision Face Detection",
                use_container_width=True
            )

            # -------------------------------------------------
            # PREPROCESS
            # -------------------------------------------------

            face_rgb = cv2.cvtColor(
                face,
                cv2.COLOR_BGR2RGB
            )

            face_resized = cv2.resize(
                face_rgb,
                (112, 112)
            )

            face_float = (
                face_resized.astype(
                    np.float32
                ) / 255.0
            )

            face_float = (
                face_float - 0.5
            ) / 0.5

            blob = cv2.dnn.blobFromImage(
                face_float
            )

            # -------------------------------------------------
            # AI INFERENCE
            # -------------------------------------------------

            with st.spinner(
                "🧠 AI is analyzing facial features..."
            ):

                time.sleep(0.5)

                expression_model.setInput(
                    blob,
                    "data"
                )

                output = expression_model.forward()

            # -------------------------------------------------
            # MODEL OUTPUT
            # -------------------------------------------------

            scores = np.asarray(
                output
            ).flatten()

            if len(scores) < 7:

                st.error(
                    "The ML model returned an unexpected output."
                )

                st.stop()

            # -------------------------------------------------
            # SOFTMAX
            # -------------------------------------------------

            exp_scores = np.exp(
                scores - np.max(scores)
            )

            probabilities = (
                exp_scores
                / np.sum(exp_scores)
            )

            # -------------------------------------------------
            # RESULT
            # -------------------------------------------------

            emotion_index = int(
                np.argmax(probabilities)
            )

            emotion = EMOTIONS[
                emotion_index
            ]

            confidence = (
                float(
                    probabilities[
                        emotion_index
                    ]
                ) * 100
            )

            # -------------------------------------------------
            # SAVE HISTORY
            # -------------------------------------------------

            st.session_state.emotion_history.append(
                emotion
            )

            st.session_state.confidence_history.append(
                confidence
            )

            st.session_state.scan_count += 1

            st.session_state.last_emotion = emotion

            # -------------------------------------------------
            # RESULT DISPLAY
            # -------------------------------------------------

            st.divider()

            st.success(
                f"{EMOTION_EMOJI[emotion]} "
                f"Detected Expression: "
                f"{emotion.upper()}"
            )

            result1, result2 = st.columns(2)

            with result1:

                st.metric(
                    "Expression",
                    emotion.title()
                )

            with result2:

                st.metric(
                    "Confidence",
                    f"{confidence:.1f}%"
                )

            st.progress(
                min(
                    confidence / 100,
                    1.0
                )
            )

            # -------------------------------------------------
            # TOP 3
            # -------------------------------------------------

            st.subheader(
                "📊 Top Expression Signals"
            )

            top_indices = np.argsort(
                probabilities
            )[::-1][:3]

            for rank, index in enumerate(
                top_indices,
                start=1
            ):

                name = EMOTIONS[
                    index
                ]

                value = (
                    float(
                        probabilities[index]
                    ) * 100
                )

                st.write(
                    f"**#{rank}** "
                    f"{EMOTION_EMOJI[name]} "
                    f"**{name.title()}** — "
                    f"{value:.1f}%"
                )

# =========================================================
# SESSION ANALYTICS
# =========================================================

st.divider()

st.subheader(
    "⚡ Session Analytics"
)

analytics1, analytics2, analytics3 = st.columns(3)

with analytics1:

    st.metric(
        "Total Scans",
        st.session_state.scan_count
    )

with analytics2:

    st.metric(
        "Last Expression",
        (
            f"{EMOTION_EMOJI.get(st.session_state.last_emotion, '')} "
            f"{st.session_state.last_emotion.title()}"
            if st.session_state.last_emotion != "None"
            else "None"
        )
    )

with analytics3:

    if st.session_state.confidence_history:

        average_confidence = np.mean(
            st.session_state.confidence_history
        )

        st.metric(
            "Average Confidence",
            f"{average_confidence:.1f}%"
        )

    else:

        st.metric(
            "Average Confidence",
            "0%"
        )

# =========================================================
# EMOTION HISTORY
# =========================================================

if st.session_state.emotion_history:

    st.divider()

    st.subheader(
        "📈 Emotion History"
    )

    history_col1, history_col2 = st.columns(
        [1.3, 1]
    )

    # -----------------------------------------------------
    # HISTORY CHART
    # -----------------------------------------------------

    with history_col1:

        emotion_numbers = [
            EMOTIONS.index(emotion) + 1
            for emotion in st.session_state.emotion_history
        ]

        chart_data = {
            "Scan": list(
                range(
                    1,
                    len(emotion_numbers) + 1
                )
            ),
            "Expression": emotion_numbers
        }

        st.line_chart(
            chart_data,
            x="Scan",
            y="Expression"
        )

        st.caption(
            "Expression index: 1 Angry • 2 Disgust • "
            "3 Fearful • 4 Happy • 5 Neutral • "
            "6 Sad • 7 Surprised"
        )

    # -----------------------------------------------------
    # RECENT SCANS
    # -----------------------------------------------------

    with history_col2:

        st.write("**Recent Scans**")

        recent = list(
            zip(
                st.session_state.emotion_history,
                st.session_state.confidence_history
            )
        )[-5:]

        for number, (name, value) in enumerate(
            reversed(recent),
            start=1
        ):

            st.write(
                f"**Scan {len(recent) - number + 1}:** "
                f"{EMOTION_EMOJI[name]} "
                f"{name.title()} — "
                f"{value:.1f}%"
            )

# =========================================================
# ALL 7 EMOTIONS
# =========================================================

if camera_image is not None and len(faces) > 0:

    st.divider()

    st.subheader(
        "🎭 Complete Expression Probability"
    )

    emotion_chart_data = {}

    for index, name in enumerate(EMOTIONS):

        emotion_chart_data[
            name.title()
        ] = [
            float(
                probabilities[index]
            ) * 100
        ]

    st.bar_chart(
        emotion_chart_data
    )

    st.caption(
        "Probability distribution returned by the ML model."
    )

# =========================================================
# SYSTEM OVERVIEW
# =========================================================

st.divider()

st.subheader(
    "🔬 System Overview"
)

metric1, metric2, metric3, metric4 = st.columns(4)

with metric1:
    st.metric(
        "Expression Classes",
        "7"
    )

with metric2:
    st.metric(
        "ML Model",
        "ONNX"
    )

with metric3:
    st.metric(
        "Vision Engine",
        "OpenCV"
    )

with metric4:
    st.metric(
        "Input",
        "Camera"
    )

# =========================================================
# HOW IT WORKS
# =========================================================

st.divider()

st.subheader(
    "🧩 How AI MoodLens Works"
)

step1, step2, step3, step4 = st.columns(4)

with step1:

    st.markdown("### 01")
    st.write("📷 **Capture**")
    st.caption(
        "Camera captures a facial image."
    )

with step2:

    st.markdown("### 02")
    st.write("🔍 **Detect**")
    st.caption(
        "OpenCV detects the visible face."
    )

with step3:

    st.markdown("### 03")
    st.write("🧠 **Analyze**")
    st.caption(
        "The ONNX model analyzes facial features."
    )

with step4:

    st.markdown("### 04")
    st.write("📊 **Classify**")
    st.caption(
        "The model returns expression probabilities."
    )

# =========================================================
# EXPRESSION LIBRARY
# =========================================================

st.divider()

st.subheader(
    "🎭 Expression Intelligence"
)

st.write(
    "AI MoodLens recognizes seven visible facial-expression categories:"
)

exp1, exp2 = st.columns(2)

with exp1:

    st.write("😠 **Angry**")
    st.write("🤢 **Disgust**")
    st.write("😨 **Fearful**")
    st.write("😊 **Happy**")

with exp2:

    st.write("😐 **Neutral**")
    st.write("😔 **Sad**")
    st.write("😮 **Surprised**")

# =========================================================
# TECHNOLOGY STACK
# =========================================================

st.divider()

st.subheader(
    "🛠 Technology Stack"
)

tech1, tech2, tech3, tech4 = st.columns(4)

with tech1:
    st.info("🐍 Python")

with tech2:
    st.info("👁 OpenCV")

with tech3:
    st.info("🧠 Machine Learning")

with tech4:
    st.info("⚙ ONNX")

# =========================================================
# RESET SESSION
# =========================================================

st.divider()

if st.button(
    "🔄 Reset Session Analytics"
):

    st.session_state.emotion_history = []
    st.session_state.confidence_history = []
    st.session_state.scan_count = 0
    st.session_state.last_emotion = "None"

    st.rerun()

# =========================================================
# DISCLAIMER
# =========================================================

st.divider()

st.caption(
    "⚠️ AI MoodLens classifies visible facial expressions "
    "from an image. Facial-expression classification should "
    "not be interpreted as a definitive measurement of a "
    "person's actual internal emotional state."
)

st.caption(
    "AI MoodLens • Python + Computer Vision + Machine Learning"
)