import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="DiabetesIQ · Risk Predictor",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# LOAD MODEL FILES
# ─────────────────────────────────────────────
@st.cache_resource
def load_models():
    model = joblib.load(r"F:\Gits\Daibetes ML pr\DT.pkl")
    scaler = joblib.load(r"F:\Gits\Daibetes ML pr\scaler.pkl")
    expected_columns = joblib.load(r"F:\Gits\Daibetes ML pr\columns.pkl")
    return model, scaler, expected_columns

model, scaler, expected_columns = load_models()

# ─────────────────────────────────────────────
# GLOBAL CSS — dark medical theme + bg image
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Root palette ── */
:root {
    --red:        #e83a3a;
    --red-glow:   #ff5a5a;
    --red-dim:    rgba(232,58,58,0.15);
    --teal:       #0ecece;
    --teal-dim:   rgba(14,206,206,0.12);
    --bg:         #080c12;
    --surface:    rgba(255,255,255,0.04);
    --border:     rgba(255,255,255,0.08);
    --text:       #e8eaf0;
    --muted:      rgba(232,234,240,0.45);
}

/* ── Full-page background ── */
.stApp {
    background:
        linear-gradient(135deg, rgba(8,12,18,0.88) 0%, rgba(15,5,5,0.92) 100%),
        url('https://images.unsplash.com/photo-1631549916768-4119b2e5f926?w=1600&q=80')
        center/cover no-repeat fixed;
    font-family: 'DM Sans', sans-serif;
    color: var(--text);
}

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem 4rem; max-width: 1100px; margin: auto; }

/* ── Hero banner ── */
.hero {
    text-align: center;
    padding: 3.5rem 2rem 2rem;
    position: relative;
}
.hero::before {
    content: '';
    position: absolute;
    top: 50%; left: 50%;
    transform: translate(-50%, -50%);
    width: 500px; height: 200px;
    background: radial-gradient(ellipse, rgba(232,58,58,0.18) 0%, transparent 70%);
    pointer-events: none;
}
.hero-badge {
    display: inline-block;
    background: var(--red-dim);
    border: 1px solid var(--red);
    color: var(--red-glow);
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 3px;
    text-transform: uppercase;
    padding: 0.35rem 1.1rem;
    border-radius: 999px;
    margin-bottom: 1.2rem;
}
.hero h1 {
    font-family: 'Syne', sans-serif;
    font-size: clamp(2.4rem, 5vw, 3.6rem);
    font-weight: 800;
    line-height: 1.1;
    margin: 0 0 0.8rem;
    background: linear-gradient(135deg, #ffffff 30%, var(--red-glow) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero p {
    color: var(--muted);
    font-size: 1.05rem;
    font-weight: 300;
    max-width: 560px;
    margin: 0 auto 2rem;
    line-height: 1.7;
}

/* ── Section headers ── */
.section-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--teal);
    margin-bottom: 0.4rem;
}
.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.4rem;
    font-weight: 700;
    color: #fff;
    margin-bottom: 1.5rem;
    padding-bottom: 0.7rem;
    border-bottom: 1px solid var(--border);
}

/* ── Glass cards ── */
.glass-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.8rem 2rem;
    margin-bottom: 1.5rem;
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
    position: relative;
    overflow: hidden;
}
.glass-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.12), transparent);
}

/* ── Info cards row ── */
.info-row {
    display: flex;
    gap: 1rem;
    margin-bottom: 2rem;
    flex-wrap: wrap;
}
.info-card {
    flex: 1;
    min-width: 160px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    text-align: center;
    backdrop-filter: blur(6px);
}
.info-card .ic-icon { font-size: 1.8rem; margin-bottom: 0.4rem; }
.info-card .ic-label {
    font-size: 0.7rem;
    font-weight: 500;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.25rem;
}
.info-card .ic-value {
    font-family: 'Syne', sans-serif;
    font-size: 1.2rem;
    font-weight: 700;
    color: #fff;
}

/* ── Streamlit widget overrides ── */
.stSlider > div > div { accent-color: var(--red); }

div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 10px !important;
    color: #fff !important;
}
div[data-baseweb="select"] * { color: #e8eaf0 !important; }

label[data-testid="stWidgetLabel"] > div > p {
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    color: rgba(232,234,240,0.7) !important;
    text-transform: uppercase;
    letter-spacing: 0.8px;
}

/* ── Predict button ── */
div[data-testid="stButton"] > button {
    width: 100%;
    background: linear-gradient(135deg, #c0392b, var(--red)) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.9rem 2rem !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    cursor: pointer !important;
    transition: all 0.25s ease !important;
    box-shadow: 0 4px 24px rgba(232,58,58,0.35) !important;
}
div[data-testid="stButton"] > button:hover {
    background: linear-gradient(135deg, var(--red), #ff6b6b) !important;
    box-shadow: 0 6px 32px rgba(232,58,58,0.55) !important;
    transform: translateY(-1px) !important;
}

/* ── Result banner ── */
.result-banner {
    border-radius: 16px;
    padding: 2rem 2.5rem;
    text-align: center;
    position: relative;
    overflow: hidden;
    backdrop-filter: blur(10px);
    margin-top: 1.5rem;
    animation: fadeSlideUp 0.5s ease both;
}
@keyframes fadeSlideUp {
    from { opacity: 0; transform: translateY(20px); }
    to   { opacity: 1; transform: translateY(0); }
}
.result-banner.positive {
    background: linear-gradient(135deg, rgba(200,30,30,0.25), rgba(232,58,58,0.12));
    border: 1px solid rgba(232,58,58,0.4);
}
.result-banner.negative {
    background: linear-gradient(135deg, rgba(14,206,206,0.15), rgba(0,180,120,0.1));
    border: 1px solid rgba(14,206,206,0.35);
}
.result-banner .result-icon { font-size: 3rem; margin-bottom: 0.6rem; }
.result-banner .result-headline {
    font-family: 'Syne', sans-serif;
    font-size: 1.7rem;
    font-weight: 800;
    margin-bottom: 0.4rem;
}
.result-banner.positive .result-headline { color: var(--red-glow); }
.result-banner.negative  .result-headline { color: var(--teal); }
.result-banner .result-sub {
    color: var(--muted);
    font-size: 0.95rem;
    margin-bottom: 1.2rem;
}

/* ── Risk meter ── */
.risk-meter-wrap { margin: 1.2rem 0 0.5rem; }
.risk-meter-track {
    background: rgba(255,255,255,0.08);
    border-radius: 999px;
    height: 14px;
    width: 100%;
    overflow: hidden;
    position: relative;
}
.risk-meter-fill {
    height: 100%;
    border-radius: 999px;
    transition: width 0.8s cubic-bezier(.22,.61,.36,1);
    position: relative;
}
.risk-meter-fill.high  {
    background: linear-gradient(90deg, #c0392b, var(--red-glow));
    box-shadow: 0 0 12px rgba(232,58,58,0.6);
}
.risk-meter-fill.low   {
    background: linear-gradient(90deg, #0a9e9e, var(--teal));
    box-shadow: 0 0 12px rgba(14,206,206,0.5);
}
.risk-meter-fill.medium {
    background: linear-gradient(90deg, #e67e22, #f1c40f);
    box-shadow: 0 0 12px rgba(241,196,15,0.5);
}
.risk-labels {
    display: flex;
    justify-content: space-between;
    font-size: 0.68rem;
    color: var(--muted);
    margin-top: 0.4rem;
}
.risk-pct {
    font-family: 'Syne', sans-serif;
    font-size: 3rem;
    font-weight: 800;
    margin: 0.8rem 0 0.2rem;
}
.result-banner.positive .risk-pct { color: var(--red-glow); }
.result-banner.negative  .risk-pct { color: var(--teal); }
.risk-sentence {
    font-size: 0.95rem;
    color: var(--muted);
    line-height: 1.6;
    max-width: 480px;
    margin: 0 auto;
}

/* ── Symptom chips ── */
.chip-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin-top: 0.6rem;
}
.chip {
    font-size: 0.72rem;
    font-weight: 500;
    padding: 0.3rem 0.8rem;
    border-radius: 999px;
    border: 1px solid var(--border);
    color: var(--muted);
    background: rgba(255,255,255,0.03);
    letter-spacing: 0.5px;
}
.chip.active {
    background: var(--red-dim);
    border-color: var(--red);
    color: var(--red-glow);
}

/* ── Disclaimer ── */
.disclaimer {
    background: rgba(255,200,0,0.06);
    border: 1px solid rgba(255,200,0,0.2);
    border-radius: 10px;
    padding: 1rem 1.4rem;
    font-size: 0.8rem;
    color: rgba(255,220,100,0.75);
    line-height: 1.6;
    margin-top: 2rem;
}

/* ── Divider ── */
hr.styled {
    border: none;
    border-top: 1px solid var(--border);
    margin: 2rem 0;
}

/* ── Tooltip badges ── */
.tip {
    font-size: 0.68rem;
    color: var(--muted);
    background: rgba(255,255,255,0.04);
    border-radius: 6px;
    padding: 0.2rem 0.5rem;
    margin-left: 0.4rem;
    vertical-align: middle;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HERO SECTION
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">🩸 AI-Powered Medical Screening</div>
    <h1>Diabetes Risk<br>Predictor</h1>
    <p>
        Answer a few clinical questions and our machine-learning model will
        estimate your likelihood of having diabetes — in seconds.
    </p>
</div>
""", unsafe_allow_html=True)

# Stats row
st.markdown("""
<div class="info-row">
    <div class="info-card">
        <div class="ic-icon">🧠</div>
        <div class="ic-label">Model</div>
        <div class="ic-value">Decision Tree</div>
    </div>
    <div class="info-card">
        <div class="ic-icon">📊</div>
        <div class="ic-label">Features</div>
        <div class="ic-value">16 Clinical</div>
    </div>
    <div class="info-card">
        <div class="ic-icon">⚡</div>
        <div class="ic-label">Result</div>
        <div class="ic-value">Instant</div>
    </div>
    <div class="info-card">
        <div class="ic-icon">🔬</div>
        <div class="ic-label">Type</div>
        <div class="ic-value">Screening</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SECTION 1 — DEMOGRAPHICS
# ─────────────────────────────────────────────
st.markdown("""
<div class="glass-card">
    <div class="section-label">Section 01</div>
    <div class="section-title">👤 Patient Demographics</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    age = st.slider("Age (years)", 18, 100, 40)
with col2:
    sex = st.selectbox("Biological Sex", ["Male (M)", "Female (F)"])

st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SECTION 2 — PRIMARY SYMPTOMS
# ─────────────────────────────────────────────
st.markdown("""
<div class="glass-card">
    <div class="section-label">Section 02</div>
    <div class="section-title">💧 Primary Metabolic Symptoms</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    polyuria      = st.selectbox("Polyuria", ["No", "Yes"],
                                  help="Excessive urination (>2.5L/day)")
with col2:
    polydipsia    = st.selectbox("Polydipsia", ["No", "Yes"],
                                  help="Abnormal thirst / excessive fluid intake")
with col3:
    polyphagia    = st.selectbox("Polyphagia", ["No", "Yes"],
                                  help="Excessive hunger even after eating")

col4, col5, col6 = st.columns(3)
with col4:
    sudden_wl     = st.selectbox("Sudden Weight Loss", ["No", "Yes"])
with col5:
    weakness      = st.selectbox("Weakness / Fatigue", ["No", "Yes"])
with col6:
    obesity       = st.selectbox("Obesity", ["No", "Yes"])

st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SECTION 3 — SECONDARY SYMPTOMS
# ─────────────────────────────────────────────
st.markdown("""
<div class="glass-card">
    <div class="section-label">Section 03</div>
    <div class="section-title">🔍 Secondary & Neurological Symptoms</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    visual_blur   = st.selectbox("Visual Blurring", ["No", "Yes"])
with col2:
    irritability  = st.selectbox("Irritability", ["No", "Yes"])
with col3:
    partial_par   = st.selectbox("Partial Paresis", ["No", "Yes"],
                                  help="Partial loss of voluntary movement")

col4, col5, col6 = st.columns(3)
with col4:
    muscle_stiff  = st.selectbox("Muscle Stiffness", ["No", "Yes"])
with col5:
    alopecia      = st.selectbox("Alopecia (Hair Loss)", ["No", "Yes"])
with col6:
    itching       = st.selectbox("Itching", ["No", "Yes"])

st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SECTION 4 — SKIN & HEALING
# ─────────────────────────────────────────────
st.markdown("""
<div class="glass-card">
    <div class="section-label">Section 04</div>
    <div class="section-title">🩹 Skin, Healing & Infections</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    delayed_heal  = st.selectbox("Delayed Healing", ["No", "Yes"],
                                  help="Wounds take unusually long to heal")
with col2:
    genital_thrush = st.selectbox("Genital Thrush", ["No", "Yes"],
                                   help="Recurrent yeast/fungal infections")

st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SYMPTOM SUMMARY CHIPS
# ─────────────────────────────────────────────
symptom_map = {
    "Polyuria": polyuria,
    "Polydipsia": polydipsia,
    "Polyphagia": polyphagia,
    "Weight Loss": sudden_wl,
    "Weakness": weakness,
    "Obesity": obesity,
    "Visual Blur": visual_blur,
    "Irritability": irritability,
    "Partial Paresis": partial_par,
    "Muscle Stiffness": muscle_stiff,
    "Alopecia": alopecia,
    "Itching": itching,
    "Delayed Healing": delayed_heal,
    "Genital Thrush": genital_thrush,
}

chips_html = '<div class="chip-grid">'
active_count = 0
for name, val in symptom_map.items():
    is_active = val == "Yes"
    if is_active:
        active_count += 1
    chips_html += f'<span class="chip {"active" if is_active else ""}">'
    chips_html += f'{"✓ " if is_active else ""}{name}</span>'
chips_html += "</div>"

st.markdown(f"""
<div style="margin-bottom:1.5rem; padding: 1rem 1.4rem;
            background: rgba(255,255,255,0.025); border-radius:12px;
            border: 1px solid var(--border);">
    <div style="font-size:0.75rem; color:var(--muted); margin-bottom:0.6rem;
                font-weight:500; letter-spacing:1px; text-transform:uppercase;">
        Active Symptoms — {active_count} / {len(symptom_map)} flagged
    </div>
    {chips_html}
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PREDICT BUTTON
# ─────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
predict_btn = st.button("🔬  Analyze My Risk", use_container_width=True)

# ─────────────────────────────────────────────
# PREDICTION LOGIC
# ─────────────────────────────────────────────
if predict_btn:
    sex_code = 1 if "M" in sex else 0

    raw_input = {
        'Age': age,
        'Gender': sex_code,
        'Polyuria':             1 if polyuria       == "Yes" else 0,
        'Polydipsia':           1 if polydipsia      == "Yes" else 0,
        'sudden weight loss':   1 if sudden_wl       == "Yes" else 0,
        'weakness':             1 if weakness        == "Yes" else 0,
        'Polyphagia':           1 if polyphagia      == "Yes" else 0,
        'Genital thrush':       1 if genital_thrush  == "Yes" else 0,
        'visual blurring':      1 if visual_blur     == "Yes" else 0,
        'Itching':              1 if itching         == "Yes" else 0,
        'Irritability':         1 if irritability    == "Yes" else 0,
        'delayed healing':      1 if delayed_heal    == "Yes" else 0,
        'partial paresis':      1 if partial_par     == "Yes" else 0,
        'muscle stiffness':     1 if muscle_stiff    == "Yes" else 0,
        'Alopecia':             1 if alopecia        == "Yes" else 0,
        'Obesity':              1 if obesity         == "Yes" else 0,
    }

    input_df = pd.DataFrame([raw_input])
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[expected_columns]

    scaled_input = scaler.transform(input_df)
    prediction   = model.predict(scaled_input)[0]
    probability  = model.predict_proba(scaled_input)[0][1]
    pct          = probability * 100

    # Classify risk tier
    if pct >= 65:
        tier = "high"; tier_label = "High Risk"
    elif pct >= 35:
        tier = "medium"; tier_label = "Moderate Risk"
    else:
        tier = "low"; tier_label = "Low Risk"

    if prediction == 1:
        sentence = (
            f"Based on {active_count} reported symptoms and your demographic profile, "
            f"the model estimates a <strong>{pct:.1f}%</strong> probability of diabetes. "
            f"Please consult a healthcare professional for a clinical diagnosis."
        )
        st.markdown(f"""
        <div class="result-banner positive">
            <div class="result-icon">⚠️</div>
            <div class="result-headline">Diabetes Risk Detected</div>
            <div class="result-sub">Your symptom profile suggests elevated risk</div>
            <div class="risk-pct">{pct:.1f}%</div>
            <div class="risk-meter-wrap">
                <div class="risk-meter-track">
                    <div class="risk-meter-fill {tier}" style="width:{pct:.1f}%"></div>
                </div>
                <div class="risk-labels"><span>0%</span><span>Low</span><span>Moderate</span><span>High</span><span>100%</span></div>
            </div>
            <br>
            <div class="risk-sentence">{sentence}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        sentence = (
            f"Based on {active_count} reported symptoms and your demographic profile, "
            f"the model estimates only a <strong>{pct:.1f}%</strong> probability of diabetes. "
            f"Keep up regular health check-ups and a balanced lifestyle."
        )
        st.markdown(f"""
        <div class="result-banner negative">
            <div class="result-icon">✅</div>
            <div class="result-headline">Low Diabetes Risk</div>
            <div class="result-sub">Your current symptom profile looks healthy</div>
            <div class="risk-pct">{pct:.1f}%</div>
            <div class="risk-meter-wrap">
                <div class="risk-meter-track">
                    <div class="risk-meter-fill {tier}" style="width:{pct:.1f}%"></div>
                </div>
                <div class="risk-labels"><span>0%</span><span>Low</span><span>Moderate</span><span>High</span><span>100%</span></div>
            </div>
            <br>
            <div class="risk-sentence">{sentence}</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Breakdown columns ──
    st.markdown("<hr class='styled'>", unsafe_allow_html=True)
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("""
        <div class="glass-card" style="height:100%">
            <div class="section-label">Result Breakdown</div>
            <div class="section-title" style="font-size:1.1rem">🧬 Key Risk Factors</div>
        """, unsafe_allow_html=True)

        key_symptoms = [k for k, v in symptom_map.items() if v == "Yes"]
        if key_symptoms:
            for s in key_symptoms:
                st.markdown(f"- 🔴 **{s}** — flagged positive", unsafe_allow_html=False)
        else:
            st.markdown("_No significant symptoms reported._")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_b:
        st.markdown("""
        <div class="glass-card" style="height:100%">
            <div class="section-label">Next Steps</div>
            <div class="section-title" style="font-size:1.1rem">📋 Recommendations</div>
        """, unsafe_allow_html=True)

        if prediction == 1:
            st.markdown("""
- 🏥 **See a doctor** for fasting blood glucose test (FBG)
- 📝 Request an **HbA1c** (glycated haemoglobin) test
- 🥗 Adopt a **low-glycaemic diet** immediately
- 🏃 Aim for **150 min/week** of moderate exercise
- ⚖️ Monitor **BMI and waist circumference** regularly
            """)
        else:
            st.markdown("""
- ✅ Continue a **balanced, low-sugar diet**
- 🏃 Maintain **regular physical activity**
- 💧 Stay **well-hydrated** daily
- 📅 Schedule **annual health screenings**
- 🔍 Re-test if new symptoms appear
            """)
        st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# DISCLAIMER
# ─────────────────────────────────────────────
st.markdown("""
<div class="disclaimer">
    ⚠️ <strong>Medical Disclaimer:</strong> This tool is for <em>informational and educational purposes only</em>.
    It does not constitute medical advice, diagnosis, or treatment. Always consult a qualified healthcare
    professional before making any health-related decisions. Model accuracy depends on training data quality
    and may not generalise to all populations.
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center; margin-top:2.5rem; color: rgba(232,234,240,0.25);
            font-size:0.75rem; letter-spacing:1px;">
    DiabetesIQ · Built by Aditya · Powered by Streamlit + scikit-learn
</div>
""", unsafe_allow_html=True)