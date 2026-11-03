import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import certifi
os.environ["SSL_CERT_FILE"] = certifi.where()

import io
import uuid

import numpy as np
import pandas as pd
import streamlit as st
import keras

from PIL import Image
from huggingface_hub import hf_hub_download
from keras.applications.efficientnet import preprocess_input
from supabase import create_client


# ==================== Configuration ====================

MODEL_ID = "Aldahmashi/DR-EfficientNetB0"
BUCKET = "fundus-images"

LABELS = [
    "No DR",
    "Mild",
    "Moderate",
    "Severe",
    "Proliferative DR",
]

st.set_page_config(
    page_title="RetinaView",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==================== Supabase ====================

@st.cache_resource
def get_supabase():
    supabase_url = st.secrets["SUPABASE_URL"].strip().rstrip("/")
    supabase_key = st.secrets["SUPABASE_SERVICE_ROLE_KEY"].strip()
    return create_client(supabase_url, supabase_key)


def get_patients():
    return (
        get_supabase()
        .table("patients")
        .select("id, name, created_at")
        .order("name")
        .execute()
        .data
    )


def get_or_create_patient(name):
    db = get_supabase()

    response = (
        db.table("patients")
        .select("id, name")
        .eq("name", name)
        .limit(1)
        .execute()
    )

    if response.data:
        return response.data[0]

    response = (
        db.table("patients")
        .insert({"name": name})
        .execute()
    )

    if not response.data:
        raise RuntimeError("Could not create patient record.")

    return response.data[0]


def get_tests(patient_id):
    return (
        get_supabase()
        .table("tests")
        .select(
            "id, patient_id, tested_at, "
            "predicted_class, scores, image_path"
        )
        .eq("patient_id", patient_id)
        .order("tested_at", desc=True)
        .execute()
        .data
    )


def save_test(patient_id, image_bytes, result):
    db = get_supabase()
    image_path = f"{patient_id}/{uuid.uuid4().hex}.jpg"

    db.storage.from_(BUCKET).upload(
        image_path,
        image_bytes,
        {
            "content-type": "image/jpeg",
            "upsert": "false",
        },
    )

    try:
        response = (
            db.table("tests")
            .insert({
                "patient_id": patient_id,
                "predicted_class": result["predicted_class"],
                "scores": result["scores"],
                "image_path": image_path,
            })
            .execute()
        )

        if not response.data:
            raise RuntimeError("Test record was not saved.")

    except Exception:
        try:
            db.storage.from_(BUCKET).remove([image_path])
        except Exception:
            pass
        raise


def load_private_image(image_path):
    return (
        get_supabase()
        .storage.from_(BUCKET)
        .download(image_path)
    )


# ==================== EfficientNetB0 ====================

@st.cache_resource
def load_efficientnet():
    path = hf_hub_download(
        repo_id=MODEL_ID,
        filename="final_model.keras",
    )
    return keras.saving.load_model(path)


def predict_efficientnet(image):
    model = load_efficientnet()

    resized = image.convert("RGB").resize((224, 224))
    array = np.asarray(resized, dtype=np.float32)
    array = np.expand_dims(array, axis=0)
    array = preprocess_input(array)

    scores = np.asarray(
        model.predict(array, verbose=0)[0]
    ).reshape(-1)

    if len(scores) != len(LABELS):
        raise ValueError(
            f"Expected {len(LABELS)} scores, got {len(scores)}."
        )

    if not np.all(np.isfinite(scores)):
        raise ValueError("Model returned invalid scores.")

    best = int(np.argmax(scores))

    return {
        "predicted_class": LABELS[best],
        "scores": {
            label: float(score)
            for label, score in zip(LABELS, scores)
        },
    }


# ==================== Display Helpers ====================

def show_scores(scores):
    st.caption("Model output scores")

    score_df = pd.DataFrame({
        "Class": LABELS,
        "Score (%)": [
            float(scores[label]) * 100
            for label in LABELS
        ],
    }).set_index("Class")

    st.bar_chart(score_df)


def show_history(tests):
    if not tests:
        st.info("No previous tests recorded.")
        return

    chronological = list(reversed(tests))
    rows = []

    for number, test in enumerate(chronological, start=1):
        predicted = test["predicted_class"]

        rows.append({
            "Test": number,
            "Date": test["tested_at"],
            "Result": predicted,
            "Severity level": (
                LABELS.index(predicted)
                if predicted in LABELS
                else None
            ),
        })

    history = pd.DataFrame(rows)

    st.subheader("Test history")
    st.dataframe(
        history[["Date", "Result"]],
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Classification trend")
    st.line_chart(
        history.set_index("Test")[["Severity level"]]
    )

    st.caption(
        "Levels: 0 = No DR, 1 = Mild, 2 = Moderate, "
        "3 = Severe, 4 = Proliferative DR. "
        "These are ordered model classes, not a validated "
        "measure of disease progression."
    )


def show_test_result(test):
    st.metric(
        "Predicted class",
        test["predicted_class"],
    )

    st.caption(f"Test date: {test['tested_at']}")

    severity = LABELS.index(test["predicted_class"])

    if severity >= 2:
        st.warning(
            "Moderate-or-higher model classification. "
            "Clinical review is appropriate."
        )
    else:
        st.info(
            "No DR/Mild model classification. "
            "This does not rule out eye disease."
        )

    show_scores(test["scores"])


# ==================== Session State ====================

if "role" not in st.session_state:
    st.session_state.role = None

if "patient_id" not in st.session_state:
    st.session_state.patient_id = None

if "patient_name" not in st.session_state:
    st.session_state.patient_name = ""


# ==================== Clinical Modern Theme ====================

st.markdown("""
<style>
/* Main page */
.stApp,
[data-testid="stAppViewContainer"] {
    background: #F4F7FA;
    color: #26384A;
}

[data-testid="stHeader"] {
    background: rgba(244, 247, 250, 0.92);
}

.block-container {
    max-width: 1440px;
    padding: 2rem 2.4rem 3rem;
}

/* Typography */
html, body, [class*="css"] {
    font-family: Arial, sans-serif;
}

h1, h2, h3, h4 {
    color: #17324D !important;
    font-weight: 650 !important;
    letter-spacing: -0.02em;
}

h1 {
    font-size: 2.15rem !important;
}

h2 {
    font-size: 1.55rem !important;
}

h3 {
    font-size: 1.12rem !important;
}

p, label, li {
    color: #34485B;
}

[data-testid="stCaptionContainer"] {
    color: #718096 !important;
}

/* Sidebar */
[data-testid="stSidebar"],
[data-testid="stSidebar"] > div {
    background: #17324D !important;
}

[data-testid="stSidebar"] {
    border-right: 1px solid #294760;
}

[data-testid="stSidebar"] * {
    color: #F2F6FA;
}

[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
    color: #C2D1DF !important;
}

[data-testid="stSidebar"] hr {
    border-color: #3A536B;
}

[data-testid="stSidebar"] div.stButton > button {
    background: #24435F !important;
    color: #FFFFFF !important;
    border: 1px solid #49637B !important;
}

[data-testid="stSidebar"] div.stButton > button:hover {
    background: #315875 !important;
    border-color: #72C6C5 !important;
}

/* Buttons */
div.stButton > button,
[data-testid="stDownloadButton"] button {
    min-height: 2.7rem;
    border-radius: 10px !important;
    font-weight: 600 !important;
    border: 1px solid #D3DEE8 !important;
    background: #FFFFFF !important;
    color: #17324D !important;
    transition: 0.2s ease;
}

div.stButton > button:hover,
[data-testid="stDownloadButton"] button:hover {
    border-color: #167D83 !important;
    background: #EDF8F8 !important;
}

div.stButton > button[kind="primary"] {
    background: #167D83 !important;
    border-color: #167D83 !important;
    color: #FFFFFF !important;
}

div.stButton > button[kind="primary"]:hover {
    background: #11666C !important;
    border-color: #11666C !important;
}

/* Content cards */
.iot-card {
    background: #FFFFFF;
    border: 1px solid #E0E7EF;
    border-radius: 16px;
    padding: 22px;
    margin-bottom: 14px;
    min-height: 130px;
    box-shadow: 0 3px 12px rgba(23, 50, 77, 0.04);
}

.iot-card h3 {
    color: #17324D !important;
    margin: 0 0 12px 0;
    font-size: 1.08rem !important;
}

.iot-card p {
    color: #64748B !important;
    font-size: 0.94rem;
    line-height: 1.6;
}

.line-icon {
    display: inline-flex;
    width: 32px;
    height: 32px;
    align-items: center;
    justify-content: center;
    border-radius: 10px;
    background: #E8F5F5;
    color: #167D83 !important;
    font-size: 1.15rem;
    margin-right: 8px;
    vertical-align: middle;
}

/* Result severity cards */
.iot-card.green {
    background: #F0F8F2;
    border-color: #B8DDC2;
}

.iot-card.yellow {
    background: #FFF9EC;
    border-color: #EAD9A6;
}

.iot-card.orange {
    background: #FFF5EC;
    border-color: #EBC6A5;
}

.iot-card.red {
    background: #FFF1F1;
    border-color: #E8B8BC;
}

.iot-card.neutral {
    background: #FFFFFF;
    border-color: #DCE5ED;
}

/* Status pills */
.status-pill {
    display: inline-block;
    padding: 6px 11px;
    border-radius: 8px;
    font-size: 0.8rem;
    font-weight: 650;
    margin: 5px 0;
}

.status-pill.green {
    background: #DDF1E2;
    color: #23663B !important;
}

.status-pill.yellow {
    background: #FFF0C7;
    color: #795B10 !important;
}

.status-pill.orange {
    background: #FFE5D0;
    color: #8C4A1C !important;
}

.status-pill.red {
    background: #FADCDD;
    color: #922F38 !important;
}

.status-pill.neutral {
    background: #E8EEF4;
    color: #526477 !important;
}

/* Metrics */
[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #E0E7EF;
    border-radius: 14px;
    padding: 18px 20px;
    box-shadow: 0 3px 12px rgba(23, 50, 77, 0.035);
}

[data-testid="stMetricLabel"] {
    color: #64748B !important;
}

[data-testid="stMetricValue"] {
    color: #17324D !important;
    font-weight: 700;
}

/* Inputs and upload areas */
[data-testid="stTextInput"] input,
[data-testid="stSelectbox"] div[data-baseweb="select"],
[data-testid="stFileUploader"] section {
    border-radius: 10px;
}

[data-testid="stFileUploader"] section {
    background: #FFFFFF;
    border: 1px dashed #B8C9D8;
}

div[data-testid="stTabs"] button {
    font-weight: 600;
}

[data-testid="stAlert"] {
    border-radius: 12px;
}

hr {
    border-color: #E0E7EF;
}

/* Tables */
[data-testid="stDataFrame"] {
    border: 1px solid #E0E7EF;
    border-radius: 12px;
    overflow: hidden;
}

/* Responsive spacing */
@media (max-width: 768px) {
    .block-container {
        padding: 1.2rem 1rem 2rem;
    }

    h1 {
        font-size: 1.8rem !important;
    }
}
</style>
""", unsafe_allow_html=True)


# ==================== Main Page ====================

st.title("RetinaView")
st.caption("IoT-Based Retinal Imaging and Diabetic Retinopathy Analysis")

st.warning(
    "Research prototype only. Model outputs are not a medical "
    "diagnosis and must not be used alone for clinical decisions."
)


# ==================== Role Selection ====================

if st.session_state.role is None:
    st.subheader("Welcome to RetinaView")
    st.write("Choose how you want to access the prototype.")

    patient_col, doctor_col = st.columns(2, gap="large")

    with patient_col:
        st.markdown(
            '<div class="iot-card">'
            '<h3><span class="line-icon">○</span> Patient Dashboard</h3>'
            '<p>Upload or capture a retinal image, view the latest result, '
            'and review previous examinations.</p></div>',
            unsafe_allow_html=True,
        )

        if st.button(
            "Continue as Patient",
            use_container_width=True,
            type="primary",
        ):
            st.session_state.role = "patient"
            st.rerun()

    with doctor_col:
        st.markdown(
            '<div class="iot-card">'
            '<h3><span class="line-icon">＋</span> Doctor Dashboard</h3>'
            '<p>Review patient results, compare examination history, '
            'and see higher-severity results first.</p></div>',
            unsafe_allow_html=True,
        )

        if st.button(
            "Continue as Doctor",
            use_container_width=True,
            type="primary",
        ):
            st.session_state.role = "doctor"
            st.rerun()

    st.stop()


# ==================== Sidebar ====================

with st.sidebar:
    st.markdown("# RetinaView")
    st.caption("RETINAL MONITORING SYSTEM")
    st.divider()

    st.markdown("**NAVIGATION**")
    st.write(f"Current view: {st.session_state.role.title()}")

    if st.session_state.role == "patient":
        st.markdown("Patient workspace")
        st.caption("Examination · Results · History")
    else:
        st.markdown("Doctor workspace")
        st.caption("Patients · Results · History")

    st.divider()

    if st.button("Back to role selection", use_container_width=True):
        st.session_state.role = None
        st.rerun()


# ==================== Result Styling Helpers ====================

def result_style(predicted_class):
    """Map model classes to display colors and a sortable severity level."""
    levels = {
        "No DR": 0,
        "Mild": 1,
        "Moderate": 2,
        "Severe": 3,
        "Proliferative DR": 4,
    }

    level = levels.get(predicted_class, -1)

    if level >= 3:
        return "red", "High priority", level
    if level == 2:
        return "orange", "Medium priority", level
    if level == 1:
        return "yellow", "Follow-up", level
    if level == 0:
        return "green", "Lower priority", level

    return "neutral", "No test yet", level


def show_priority_card(patient, latest_test):
    if latest_test:
        predicted = latest_test["predicted_class"]
        color, priority, _ = result_style(predicted)
        date_text = latest_test.get("tested_at", "Date unavailable")
        result_text = predicted
    else:
        color, priority = "neutral", "No test yet"
        date_text = "No examinations recorded"
        result_text = "Awaiting first test"

    st.markdown(
        f'<div class="iot-card {color}">'
        f'<h3>{patient["name"]}</h3>'
        f'<p><b>Latest result:</b> {result_text}</p>'
        f'<span class="status-pill {color}">{priority}</span>'
        f'<p><small>{date_text}</small></p></div>',
        unsafe_allow_html=True,
    )


# ==================== Patient Interface ====================

if st.session_state.role == "patient":
    st.header("Patient Dashboard")
    st.caption("Monitor your retinal health and examination history.")

    name = st.text_input(
        "Patient name",
        value=st.session_state.patient_name,
        placeholder="Enter patient name",
    ).strip()

    if not name:
        st.info("Enter your name to continue.")
        st.stop()

    try:
        patient = get_or_create_patient(name)
        st.session_state.patient_id = patient["id"]
        st.session_state.patient_name = patient["name"]
    except Exception as exc:
        st.error(f"Could not access patient record: {exc}")
        st.stop()

    st.caption(f"Patient ID: {patient['id']}")

    try:
        tests = get_tests(st.session_state.patient_id)
    except Exception as exc:
        st.error(f"Could not load test history: {exc}")
        st.stop()

    # Examination and latest result
    new_col, result_col = st.columns(2, gap="large")

    with new_col:
        st.markdown(
            '<div class="iot-card">'
            '<h3><span class="line-icon">▣</span> New Examination</h3>'
            '<p>Upload a retinal image or use your device camera.</p></div>',
            unsafe_allow_html=True,
        )

        source = st.radio(
            "Image source",
            ["Upload from device", "Use camera"],
            horizontal=True,
        )

        if source == "Upload from device":
            captured = st.file_uploader(
                "Upload retinal fundus image",
                type=["jpg", "jpeg", "png", "bmp"],
            )
        else:
            captured = st.camera_input("Capture an image")

        if captured is not None:
            try:
                original = Image.open(
                    io.BytesIO(captured.getvalue())
                ).convert("RGB")

                st.image(
                    original,
                    caption="Selected retinal image",
                    use_container_width=True,
                )

                if st.button(
                    "Analyze image",
                    type="primary",
                    use_container_width=True,
                ):
                    with st.spinner("Analyzing retinal image..."):
                        result = predict_efficientnet(original)

                        output = io.BytesIO()
                        original.save(output, format="JPEG")

                        save_test(
                            st.session_state.patient_id,
                            output.getvalue(),
                            result,
                        )

                    st.success("Analysis completed and saved.")
                    st.rerun()

            except Exception as exc:
                st.error(f"Analysis or saving failed: {exc}")

    with result_col:
        st.markdown(
            '<div class="iot-card">'
            '<h3><span class="line-icon">◉</span> Latest Result</h3>'
            '<p>Most recent model classification.</p></div>',
            unsafe_allow_html=True,
        )

        if tests:
            latest = tests[0]
            color, priority, _ = result_style(latest["predicted_class"])

            st.markdown(
                f'<div class="iot-card {color}">'
                f'<h2>{latest["predicted_class"]}</h2>'
                f'<span class="status-pill {color}">{priority}</span>'
                f'<p>Test date: {latest["tested_at"]}</p></div>',
                unsafe_allow_html=True,
            )

            show_test_result(latest)
        else:
            st.info("Your first test result will appear here.")

    # Examination history
    st.divider()
    st.header("Progress Tracking & Examination History")

    if tests:
        show_history(tests)

        with st.expander("View previous test images and model scores"):
            for test in tests:
                st.markdown(
                    f'**{test["tested_at"]} — {test["predicted_class"]}**'
                )

                st.json(test["scores"])

                if test.get("image_path"):
                    try:
                        image_data = load_private_image(test["image_path"])
                        st.image(image_data, width=350)
                    except Exception:
                        st.error("Could not load this image.")
    else:
        st.info(
            "Your examination history and progress chart will appear "
            "here after your first test."
        )


# ==================== Doctor Interface ====================

elif st.session_state.role == "doctor":
    st.header("Doctor Dashboard")
    st.caption(
        "Patient overview — higher-severity model classifications appear first."
    )

    try:
        patients = get_patients()

        if not patients:
            st.info(
                "No patient records yet. Create a patient record from "
                "the patient interface first."
            )
            st.stop()

        patient_tests = {}
        sortable_patients = []

        for patient in patients:
            tests = get_tests(patient["id"])
            patient_tests[patient["id"]] = tests

            if tests:
                color, priority, level = result_style(
                    tests[0]["predicted_class"]
                )
            else:
                color, priority, level = "neutral", "No test yet", -1

            sortable_patients.append(
                (level, patient["name"].lower(), patient)
            )

        # Sort by latest classification severity, then patient name
        sortable_patients.sort(
            key=lambda item: (-item[0], item[1])
        )

        total_patients = len(patients)
        high_priority_count = sum(
            1
            for level, _, _ in sortable_patients
            if level >= 3
        )
        patients_with_tests = sum(
            bool(patient_tests[p["id"]])
            for p in patients
        )

        metric1, metric2, metric3 = st.columns(3)

        metric1.metric("Total patients", total_patients)
        metric2.metric("High priority", high_priority_count)
        metric3.metric("Patients with tests", patients_with_tests)

        st.divider()

        add_col, list_col = st.columns([0.85, 1.7], gap="large")

        with add_col:
            st.markdown(
                '<div class="iot-card">'
                '<h3><span class="line-icon">＋</span> Add Patient</h3>'
                '<p>Patient records are created when a patient enters '
                'their name in the patient interface.</p></div>',
                unsafe_allow_html=True,
            )

            st.info(
                "For this prototype, patient registration starts "
                "from the Patient Dashboard."
            )

        with list_col:
            st.subheader("Patients List")
            st.caption(
                "Priority order: high severity, then moderate, mild, and no DR."
            )

            for level, _, patient in sortable_patients:
                tests = patient_tests[patient["id"]]
                latest = tests[0] if tests else None

                show_priority_card(patient, latest)

                latest_col, history_col = st.columns(2)

                with latest_col:
                    if st.button(
                        "Latest Test",
                        key=f"latest_{patient['id']}",
                        use_container_width=True,
                    ):
                        st.session_state["doctor_selected_patient"] = patient["id"]
                        st.session_state["doctor_view"] = "latest"

                with history_col:
                    if st.button(
                        "Tracking & History",
                        key=f"history_{patient['id']}",
                        use_container_width=True,
                    ):
                        st.session_state["doctor_selected_patient"] = patient["id"]
                        st.session_state["doctor_view"] = "history"

        selected_id = st.session_state.get("doctor_selected_patient")
        selected_view = st.session_state.get("doctor_view", "latest")

        selected_patient = next(
            (p for p in patients if p["id"] == selected_id),
            None,
        )

        if selected_patient:
            selected_tests = patient_tests[selected_patient["id"]]

            st.divider()
            st.subheader(
                f'{selected_patient["name"]} — {selected_view.title()}'
            )
            st.caption(f'Patient ID: {selected_patient["id"]}')

            if not selected_tests:
                st.info("No tests recorded for this patient.")

            elif selected_view == "latest":
                show_test_result(selected_tests[0])

                if selected_tests[0].get("image_path"):
                    with st.expander("View latest retinal image"):
                        try:
                            image_data = load_private_image(
                                selected_tests[0]["image_path"]
                            )
                            st.image(image_data, width=450)
                        except Exception:
                            st.error("Could not load the image.")

            else:
                show_history(selected_tests)

                with st.expander("View all test records and images"):
                    for test in selected_tests:
                        st.markdown(
                            f'**{test["tested_at"]} — '
                            f'{test["predicted_class"]}**'
                        )

                        st.json(test["scores"])

                        if test.get("image_path"):
                            try:
                                image_data = load_private_image(
                                    test["image_path"]
                                )
                                st.image(image_data, width=350)
                            except Exception:
                                st.error("Could not load the image.")

                export_df = pd.DataFrame([
                    {
                        "Test ID": test["id"],
                        "Date": test["tested_at"],
                        "Classification": test["predicted_class"],
                        **{
                            f"Score - {label}": test["scores"][label]
                            for label in LABELS
                        },
                    }
                    for test in selected_tests
                ])

                st.download_button(
                    "Export history as CSV",
                    data=export_df.to_csv(index=False).encode("utf-8-sig"),
                    file_name="retinaview_patient_history.csv",
                    mime="text/csv",
                )

    except Exception as exc:
        st.error(f"Could not load doctor dashboard: {exc}")