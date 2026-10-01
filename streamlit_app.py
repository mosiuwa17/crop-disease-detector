import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import csv
import time
from datetime import datetime

# ========== PAGE CONFIG ==========
st.set_page_config(
    page_title="Crop Disease AI Detector | Tsana Africa",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== CUSTOM CSS ==========
st.markdown("""<style>
.block-container {
    padding-top: 0rem;
    padding-bottom: 2rem;
    padding-left: 2rem;
    padding-right: 2rem;
    max-width: 1400px;
}
header {visibility: hidden;}

html, body, [class*="css"] {
    font-family: 'Inter', 'Segoe UI', sans-serif;
}

.hero {
    background: linear-gradient(135deg, #1B4332 0%, #2D6A4F 50%, #40916C 100%);
    padding: 80px 40px;
    border-radius: 24px;
    color: white;
    text-align: center;
    margin-bottom: 40px;
    position: relative;
    overflow: hidden;
    box-shadow: 0 20px 60px rgba(27, 67, 50, 0.3);
}
.hero-logo {
    font-size: 80px;
    margin-bottom: 10px;
    filter: drop-shadow(0 4px 20px rgba(212, 163, 115, 0.5));
}
.hero-title {
    font-size: 56px;
    font-weight: 800;
    margin: 10px 0;
    letter-spacing: -1.5px;
}
.hero-subtitle {
    font-size: 20px;
    opacity: 0.9;
    margin-bottom: 30px;
    font-weight: 300;
}
.hero-badge {
    display: inline-block;
    background: rgba(212, 163, 115, 0.25);
    border: 1px solid rgba(212, 163, 115, 0.5);
    color: #D4A373;
    padding: 8px 20px;
    border-radius: 50px;
    font-size: 14px;
    font-weight: 600;
    margin: 0 5px;
}

.feature-card {
    background: white;
    padding: 30px 25px;
    border-radius: 20px;
    border: 1px solid #E8E5DE;
    text-align: center;
    transition: all 0.3s ease;
    height: 100%;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.04);
}
.feature-card:hover {
    transform: translateY(-8px);
    box-shadow: 0 20px 40px rgba(27, 67, 50, 0.15);
    border-color: #52B788;
}
.feature-icon { font-size: 44px; margin-bottom: 15px; }
.feature-title { font-size: 18px; font-weight: 700; color: #1B4332; margin-bottom: 10px; }
.feature-desc { font-size: 14px; color: #636E72; line-height: 1.6; }

.stat-card {
    background: linear-gradient(135deg, #FAF9F6, #F0F4F0);
    padding: 25px;
    border-radius: 18px;
    border-left: 5px solid #52B788;
}
.stat-number { font-size: 42px; font-weight: 800; color: #1B4332; line-height: 1; margin-bottom: 8px; }
.stat-label { font-size: 14px; color: #636E72; text-transform: uppercase; letter-spacing: 1px; font-weight: 600; }

.result-card {
    background: white;
    padding: 35px;
    border-radius: 20px;
    border: 1px solid #E8E5DE;
    box-shadow: 0 10px 40px rgba(0, 0, 0, 0.06);
}
.disease-name {
    font-size: 38px;
    font-weight: 800;
    color: #1B4332;
    margin: 10px 0;
    letter-spacing: -1px;
}
.confidence-badge {
    display: inline-block;
    padding: 8px 20px;
    border-radius: 50px;
    font-weight: 700;
    font-size: 18px;
    margin: 5px 0;
}
.conf-high { background: #D8F3DC; color: #1B4332; }
.conf-mod { background: #FFF3CD; color: #8B6914; }
.conf-low { background: #F8D7DA; color: #842029; }

.severity-container { margin: 20px 0; }
.severity-label-row {
    display: flex;
    justify-content: space-between;
    font-size: 12px;
    color: #636E72;
    margin-bottom: 6px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.severity-bar-bg {
    background: #F0F4F0;
    border-radius: 12px;
    height: 18px;
    overflow: hidden;
}
.severity-bar-fill {
    height: 100%;
    border-radius: 12px;
    transition: width 0.6s ease;
}

.section-header {
    font-size: 32px;
    font-weight: 800;
    color: #1B4332;
    margin: 40px 0 20px 0;
    letter-spacing: -0.5px;
}
.section-subtitle { font-size: 16px; color: #636E72; margin-bottom: 30px; }

.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #E8E5DE, transparent);
    margin: 50px 0;
}

.stButton > button {
    background: #1B4332;
    color: white;
    border: none;
    border-radius: 12px;
    padding: 12px 30px;
    font-weight: 600;
    font-size: 15px;
    transition: all 0.3s;
    box-shadow: 0 4px 15px rgba(27, 67, 50, 0.2);
}
.stButton > button:hover {
    background: #2D6A4F;
    box-shadow: 0 8px 25px rgba(27, 67, 50, 0.3);
    transform: translateY(-2px);
}

[data-testid="stFileUploader"] {
    background: white;
    border-radius: 20px;
    padding: 20px;
    border: 2px dashed #52B788;
}
[data-testid="stFileUploader"]:hover {
    border-color: #1B4332;
    background: #FAF9F6;
}

[data-testid="stSidebar"] {
    background: #1B4332 !important;
    border-right: none !important;
}
[data-testid="stSidebar"] * { color: #FFFFFF !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] h4,
[data-testid="stSidebar"] h5 {
    color: #D4A373 !important;
    font-weight: 700 !important;
}
[data-testid="stSidebar"] .stCaption,
[data-testid="stSidebar"] small { color: #B7DCC7 !important; }
[data-testid="stSidebar"] p { color: #FFFFFF !important; }
[data-testid="stSidebar"] label { color: #FFFFFF !important; font-weight: 600 !important; }
[data-testid="stSidebar"] hr {
    border-color: rgba(255, 255, 255, 0.15) !important;
    margin: 15px 0 !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background: #FFFFFF !important;
    color: #1B4332 !important;
    border-radius: 10px !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] svg { color: #1B4332 !important; }
[data-testid="stSidebar"] [data-baseweb="popover"] ul { background: #FFFFFF !important; }
[data-testid="stSidebar"] [data-baseweb="popover"] li { color: #1B4332 !important; }
[data-testid="stSidebar"] [data-testid="stMetric"] {
    background: rgba(255, 255, 255, 0.08) !important;
    padding: 12px 15px !important;
    border-radius: 12px !important;
    border-left: 3px solid #D4A373 !important;
}
[data-testid="stSidebar"] [data-testid="stMetricValue"] {
    color: #FFFFFF !important;
    font-weight: 800 !important;
}
[data-testid="stSidebar"] [data-testid="stMetricLabel"] {
    color: #B7DCC7 !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    font-size: 11px !important;
    letter-spacing: 1px !important;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 10px;
    background: #FAF9F6;
    padding: 8px;
    border-radius: 15px;
    border: 1px solid #E8E5DE;
}
.stTabs [data-baseweb="tab"] {
    height: 50px;
    padding: 0 25px;
    background: transparent;
    border-radius: 10px;
    font-weight: 600;
    color: #636E72;
    font-size: 15px;
}
.stTabs [aria-selected="true"] {
    background: #1B4332 !important;
    color: #FFFFFF !important;
}
.stTabs [aria-selected="true"] * { color: #FFFFFF !important; }

.streamlit-expanderHeader {
    font-weight: 600;
    color: #1B4332;
    background: #FFFFFF;
    border-radius: 12px;
}

.encyclopedia-card {
    background: white;
    padding: 25px;
    border-radius: 16px;
    border-left: 5px solid #52B788;
    margin-bottom: 15px;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.04);
}
.encyclopedia-row { display: flex; margin: 8px 0; font-size: 14px; }
.encyclopedia-key {
    width: 140px;
    font-weight: 700;
    color: #636E72;
    text-transform: uppercase;
    font-size: 11px;
    letter-spacing: 0.5px;
    padding-top: 3px;
}
.encyclopedia-val { flex: 1; color: #2D3436; line-height: 1.6; }
</style>""", unsafe_allow_html=True)

# ========== CONFIGURATION ==========
MODEL_PATH = 'crop_disease_model.h5'
HISTORY_FILE = 'diagnosis_history_web.csv'

# ========== LOAD MODEL ==========
@st.cache_resource(show_spinner=False)
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

with st.spinner("🌿 Loading AI model..."):
    model = load_model()

# ========== CLASS NAMES ==========
CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight',
    'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight',
    'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

# ========== MULTI-LANGUAGE NAMES ==========
LANG_NAMES = {
    'en': {
        'Corn_(maize)___Northern_Leaf_Blight': 'Northern Leaf Blight',
        'Corn_(maize)___Common_rust_': 'Common Rust',
        'Corn_(maize)___healthy': 'Healthy Maize',
        'Tomato___Early_blight': 'Early Blight',
        'Tomato___Late_blight': 'Late Blight',
        'Tomato___healthy': 'Healthy Tomato',
        'Potato___Early_blight': 'Early Blight',
        'Potato___Late_blight': 'Late Blight',
        'Potato___healthy': 'Healthy Potato',
        'Apple___Apple_scab': 'Apple Scab',
        'Apple___healthy': 'Healthy Apple',
        'Grape___Black_rot': 'Black Rot',
        'Grape___healthy': 'Healthy Grape',
    },
    'st': {
        'Corn_(maize)___Northern_Leaf_Blight': 'Lebala la poone',
        'Corn_(maize)___Common_rust_': 'Mafome a poone',
        'Corn_(maize)___healthy': 'Poone e phetse hantle',
        'Tomato___Early_blight': 'Bolwetse ba tamati',
        'Tomato___Late_blight': 'Bolwetse bo boholo ba tamati',
        'Tomato___healthy': 'Tamati e phetse hantle',
        'Potato___Early_blight': 'Bolwetse ba litapole',
        'Potato___Late_blight': 'Bolwetse bo boholo ba litapole',
        'Potato___healthy': 'Litapole li phetse hantle',
        'Apple___Apple_scab': 'Bolwetse ba apole',
        'Apple___healthy': 'Apole e phetse hantle',
        'Grape___Black_rot': 'Bolwetse ba morara',
        'Grape___healthy': 'Morara o phetse hantle',
    },
    'fr': {
        'Corn_(maize)___Northern_Leaf_Blight': 'Brûlure nordique',
        'Corn_(maize)___Common_rust_': 'Rouille commune',
        'Corn_(maize)___healthy': 'Maïs sain',
        'Tomato___Early_blight': 'Alternariose',
        'Tomato___Late_blight': 'Mildiou',
        'Tomato___healthy': 'Tomate saine',
        'Potato___Early_blight': 'Alternariose',
        'Potato___Late_blight': 'Mildiou',
        'Potato___healthy': 'Pomme de terre saine',
        'Apple___Apple_scab': 'Tavelure du pommier',
        'Apple___healthy': 'Pomme saine',
        'Grape___Black_rot': 'Black rot',
        'Grape___healthy': 'Raisin sain',
    },
    'zu': {
        'Corn_(maize)___Northern_Leaf_Blight': 'Isifo samakhasi',
        'Corn_(maize)___Common_rust_': 'Ukugqwala',
        'Corn_(maize)___healthy': 'Umbila onempilo',
        'Tomato___Early_blight': 'Isifo samakhasi',
        'Tomato___Late_blight': 'Isifo esibi',
        'Tomato___healthy': 'Utamatisi onempilo',
        'Potato___Early_blight': 'Isifo samakhasi',
        'Potato___Late_blight': 'Isifo esibi',
        'Potato___healthy': 'Izambane elinempilo',
        'Apple___Apple_scab': 'Isifo sehlamvu',
        'Apple___healthy': 'I-apula elinempilo',
        'Grape___Black_rot': 'Isifo esimnyama',
        'Grape___healthy': 'Umvini onempilo',
    },
    'xh': {
        'Corn_(maize)___Northern_Leaf_Blight': 'Isifo samagqabi',
        'Corn_(maize)___Common_rust_': 'Umhlwa',
        'Corn_(maize)___healthy': 'Umbona ophilileyo',
        'Tomato___Early_blight': 'Isifo samagqabi',
        'Tomato___Late_blight': 'Isifo esibi',
        'Tomato___healthy': 'Utamatisi ophilileyo',
        'Potato___Early_blight': 'Isifo samagqabi',
        'Potato___Late_blight': 'Isifo esibi',
        'Potato___healthy': 'Iitapile eziphilileyo',
        'Apple___Apple_scab': 'Isifo samagqabi',
        'Apple___healthy': 'Iapile ephilileyo',
        'Grape___Black_rot': 'Isifo esimnyama',
        'Grape___healthy': 'Umdiliya ophilileyo',
    },
    'af': {
        'Corn_(maize)___Northern_Leaf_Blight': 'Noordelike blaarskroei',
        'Corn_(maize)___Common_rust_': 'Gewone roes',
        'Corn_(maize)___healthy': 'Gesonde mielies',
        'Tomato___Early_blight': 'Vroeë roes',
        'Tomato___Late_blight': 'Laatroes',
        'Tomato___healthy': 'Gesonde tamatie',
        'Potato___Early_blight': 'Vroeë roes',
        'Potato___Late_blight': 'Laatroes',
        'Potato___healthy': 'Gesonde aartappel',
        'Apple___Apple_scab': 'Appelskurft',
        'Apple___healthy': 'Gesonde appel',
        'Grape___Black_rot': 'Swartvrot',
        'Grape___healthy': 'Gesonde druif',
    },
}

# ========== TREATMENT DATABASE ==========
TREATMENTS_EN = {
    'Corn_(maize)___Northern_Leaf_Blight': 'Remove affected leaves. Apply fungicide (chlorothalonil). Improve air circulation. Plant resistant varieties next season.',
    'Corn_(maize)___Common_rust_': 'Apply fungicide. Plant resistant varieties. Avoid overhead watering.',
    'Corn_(maize)___healthy': 'No disease detected. Continue regular care and monitoring.',
    'Tomato___Early_blight': 'Remove affected leaves. Apply copper-based fungicide. Avoid overhead watering. Practice crop rotation.',
    'Tomato___Late_blight': 'Remove and destroy affected plants. Apply fungicide immediately. Improve drainage.',
    'Tomato___healthy': 'No disease detected. Continue regular care and monitoring.',
    'Potato___Early_blight': 'Remove affected leaves. Apply fungicide. Practice crop rotation.',
    'Potato___Late_blight': 'Remove and destroy affected plants. Apply fungicide immediately.',
    'Potato___healthy': 'No disease detected. Continue regular care and monitoring.',
    'Apple___Apple_scab': 'Apply fungicide. Remove fallen leaves. Prune for air circulation.',
    'Apple___healthy': 'No disease detected. Continue regular care and monitoring.',
    'Grape___Black_rot': 'Remove affected fruit. Apply fungicide. Prune for air circulation.',
    'Grape___healthy': 'No disease detected. Continue regular care and monitoring.'
}

TREATMENTS_ST = {
    'Corn_(maize)___Northern_Leaf_Blight': 'Tlosa makhasi a lutseng a le joalo. Sebelisa moriana o bolaeang likokoana-hloko. Hloekisa moea. Jala mefuta e mamellang.',
    'Corn_(maize)___Common_rust_': 'Sebelisa moriana o bolaeang likokoana-hloko. Jala mefuta e mamellang. Qoba ho nosetsa ka holimo.',
    'Corn_(maize)___healthy': 'Ha ho lefu le fumanoeng. Tsoela pele ho hlokomela lijalo.',
    'Tomato___Early_blight': 'Tlosa makhasi a lutseng a le joalo. Sebelisa moriana o nang le koporo. Qoba ho nosetsa ka holimo. Fetola lijalo.',
    'Tomato___Late_blight': 'Tlosa le ho chesa limela tse lutseng li le joalo. Sebelisa moriana hang-hang. Ntlafatsa drainage.',
    'Tomato___healthy': 'Ha ho lefu le fumanoeng. Tsoela pele ho hlokomela lijalo.',
    'Potato___Early_blight': 'Tlosa makhasi a lutseng a le joalo. Sebelisa moriana. Fetola lijalo.',
    'Potato___Late_blight': 'Tlosa le ho chesa limela tse lutseng li le joalo. Sebelisa moriana hang-hang.',
    'Potato___healthy': 'Ha ho lefu le fumanoeng. Tsoela pele ho hlokomela lijalo.',
    'Apple___Apple_scab': 'Sebelisa moriana. Tlosa makhasi a oeleng. Fokotsa makala.',
    'Apple___healthy': 'Ha ho lefu le fumanoeng. Tsoela pele ho hlokomela lijalo.',
    'Grape___Black_rot': 'Tlosa litholoana tse lutseng li le joalo. Sebelisa moriana. Fokotsa makala.',
    'Grape___healthy': 'Ha ho lefu le fumanoeng. Tsoela pele ho hlokomela lijalo.'
}

TREATMENTS_FR = {
    'Corn_(maize)___Northern_Leaf_Blight': 'Retirez les feuilles affectées. Appliquez un fongicide. Améliorez la circulation de l\'air.',
    'Corn_(maize)___Common_rust_': 'Appliquez un fongicide. Plantez des variétés résistantes.',
    'Corn_(maize)___healthy': 'Aucune maladie détectée. Continuez les soins réguliers.',
    'Tomato___Early_blight': 'Retirez les feuilles affectées. Appliquez un fongicide à base de cuivre.',
    'Tomato___Late_blight': 'Retirez et détruisez les plantes affectées. Appliquez un fongicide immédiatement.',
    'Tomato___healthy': 'Aucune maladie détectée. Continuez les soins réguliers.',
    'Potato___Early_blight': 'Retirez les feuilles affectées. Appliquez un fongicide. Pratiquez la rotation des cultures.',
    'Potato___Late_blight': 'Retirez et détruisez les plantes affectées. Appliquez un fongicide immédiatement.',
    'Potato___healthy': 'Aucune maladie détectée. Continuez les soins réguliers.',
    'Apple___Apple_scab': 'Appliquez un fongicide. Retirez les feuilles tombées. Taillez pour la circulation de l\'air.',
    'Apple___healthy': 'Aucune maladie détectée. Continuez les soins réguliers.',
    'Grape___Black_rot': 'Retirez les fruits affectés. Appliquez un fongicide.',
    'Grape___healthy': 'Aucune maladie détectée. Continuez les soins réguliers.'
}

TREATMENTS_ZU = {
    'Corn_(maize)___Northern_Leaf_Blight': 'Susa amaqabunga athintekile. Sebenzisa isibulala-zifo. Thuthukisa umoya.',
    'Corn_(maize)___Common_rust_': 'Sebenzisa isibulala-zifo. TsHala izinhlobo eziqinile.',
    'Corn_(maize)___healthy': 'Azikho izifo ezitholiwe. Qhubeka nokunakekela.',
    'Tomato___Early_blight': 'Susa amaqabunga athintekile. Sebenzisa isibulala-zifo sethusi.',
    'Tomato___Late_blight': 'Susa futhi ushise izitshalo ezithintekile. Sebenzisa isibulala-zifo ngokushesha.',
    'Tomato___healthy': 'Azikho izifo ezitholiwe. Qhubeka nokunakekela.',
    'Potato___Early_blight': 'Susa amaqabunga athintekile. Sebenzisa isibulala-zifo.',
    'Potato___Late_blight': 'Susa futhi ushise izitshalo ezithintekile. Sebenzisa isibulala-zifo ngokushesha.',
    'Potato___healthy': 'Azikho izifo ezitholiwe. Qhubeka nokunakekela.',
    'Apple___Apple_scab': 'Sebenzisa isibulala-zifo. Susa amaqabunga awileyo.',
    'Apple___healthy': 'Azikho izifo ezitholiwe. Qhubeka nokunakekela.',
    'Grape___Black_rot': 'Susa izithelo ezithintekile. Sebenzisa isibulala-zifo.',
    'Grape___healthy': 'Azikho izifo ezitholiwe. Qhubeka nokunakekela.'
}

TREATMENTS_XH = {
    'Corn_(maize)___Northern_Leaf_Blight': 'Susa amagqabi achaphazelekileyo. Sebenzisa isibulali-zifo. Phucula umoya.',
    'Corn_(maize)___Common_rust_': 'Sebenzisa isibulali-zifo. Tyala iindidi ezomeleleyo.',
    'Corn_(maize)___healthy': 'Akukho zifo zifunyenweyo. Qhubeka nokunyamekela.',
    'Tomato___Early_blight': 'Susa amagqabi achaphazelekileyo. Sebenzisa isibulali-zifo sobhedu.',
    'Tomato___Late_blight': 'Susa kwaye utshise izityalo ezichaphazelekileyo. Sebenzisa isibulali-zifo ngokukhawuleza.',
    'Tomato___healthy': 'Akukho zifo zifunyenweyo. Qhubeka nokunyamekela.',
    'Potato___Early_blight': 'Susa amagqabi achaphazelekileyo. Sebenzisa isibulali-zifo.',
    'Potato___Late_blight': 'Susa kwaye utshise izityalo ezichaphazelekileyo. Sebenzisa isibulali-zifo ngokukhawuleza.',
    'Potato___healthy': 'Akukho zifo zifunyenweyo. Qhubeka nokunyamekela.',
    'Apple___Apple_scab': 'Sebenzisa isibulali-zifo. Susa amagqabi awileyo.',
    'Apple___healthy': 'Akukho zifo zifunyenweyo. Qhubeka nokunyamekela.',
    'Grape___Black_rot': 'Susa iziqhamo ezichaphazelekayo. Sebenzisa isibulali-zifo.',
    'Grape___healthy': 'Akukho zifo zifunyenweyo. Qhubeka nokunyamekela.'
}

TREATMENTS_AF = {
    'Corn_(maize)___Northern_Leaf_Blight': 'Verwyder aangetaste blare. Pas swamdoder toe. Verbeter lugvloei. Plant weerstandbiedende variëteite.',
    'Corn_(maize)___Common_rust_': 'Pas swamdoder toe. Plant weerstandbiedende variëteite.',
    'Corn_(maize)___healthy': 'Geen siekte opgespoor nie. Hou aan met gereelde sorg.',
    'Tomato___Early_blight': 'Verwyder aangetaste blare. Pas koper-gebaseerde swamdoder toe.',
    'Tomato___Late_blight': 'Verwyder en vernietig aangetaste plante. Pas swamdoder onmiddellik toe.',
    'Tomato___healthy': 'Geen siekte opgespoor nie. Hou aan met gereelde sorg.',
    'Potato___Early_blight': 'Verwyder aangetaste blare. Pas swamdoder toe. Beoefen wisselbou.',
    'Potato___Late_blight': 'Verwyder en vernietig aangetaste plante. Pas swamdoder onmiddellik toe.',
    'Potato___healthy': 'Geen siekte opgespoor nie. Hou aan met gereelde sorg.',
    'Apple___Apple_scab': 'Pas swamdoder toe. Verwyder gevalle blare. Snoei vir lugvloei.',
    'Apple___healthy': 'Geen siekte opgespoor nie. Hou aan met gereelde sorg.',
    'Grape___Black_rot': 'Verwyder aangetaste vrugte. Pas swamdoder toe.',
    'Grape___healthy': 'Geen siekte opgespoor nie. Hou aan met gereelde sorg.'
}

TREATMENT_MAP = {
    'en': TREATMENTS_EN, 'st': TREATMENTS_ST, 'fr': TREATMENTS_FR,
    'zu': TREATMENTS_ZU, 'xh': TREATMENTS_XH, 'af': TREATMENTS_AF,
}

FALLBACK = {
    'en': 'Consult your local agricultural extension officer for treatment advice.',
    'st': 'Buisana le ofisiri ea temo ea sebaka sa heno bakeng sa thuso.',
    'fr': 'Consultez votre agent de vulgarisation agricole local.',
    'zu': 'Xhumana nomeluleki wakho wezolimo wendawo.',
    'xh': 'Qhagamshelana neGosa lakho lezolimo lendawo.',
    'af': 'Raadpleeg u plaaslike landbou-uitbreidingsbeampte.',
}

def get_treatment(disease, language='en'):
    return TREATMENT_MAP.get(language, TREATMENTS_EN).get(disease, FALLBACK.get(language, FALLBACK['en']))

def get_display_name(disease, language='en'):
    if language in LANG_NAMES and disease in LANG_NAMES[language]:
        return LANG_NAMES[language][disease]
    return disease.replace('___', ' • ').replace('_', ' ')

# ========== DISEASE ENCYCLOPEDIA ==========
ENCYCLOPEDIA = {
    'Corn_(maize)___Northern_Leaf_Blight': {
        'name': 'Northern Leaf Blight', 'crop': 'Maize (Corn)',
        'pathogen': 'Fungus (Exserohilum turcicum)',
        'symptoms': 'Long, elliptical gray-green or tan lesions on leaves, starting from lower leaves.',
        'spread': 'Wind-borne spores; favored by cool, wet conditions.',
        'treatment': 'Apply fungicide (chlorothalonil, propiconazole). Remove affected leaves. Plant resistant hybrids.',
        'severity': 65, 'severity_label': 'Moderate-High'
    },
    'Corn_(maize)___Common_rust_': {
        'name': 'Common Rust', 'crop': 'Maize (Corn)',
        'pathogen': 'Fungus (Puccinia sorghi)',
        'symptoms': 'Small, circular to elongate brown pustules on both leaf surfaces.',
        'spread': 'Wind-borne spores from southern regions.',
        'treatment': 'Apply fungicide early. Plant resistant varieties. Avoid overhead watering.',
        'severity': 55, 'severity_label': 'Moderate'
    },
    'Corn_(maize)___healthy': {
        'name': 'Healthy Maize', 'crop': 'Maize (Corn)',
        'pathogen': 'None',
        'symptoms': 'Vibrant green leaves with no spots, lesions, or discoloration.',
        'spread': 'N/A',
        'treatment': 'Continue regular care: adequate water, balanced fertilizer, periodic monitoring.',
        'severity': 0, 'severity_label': 'None'
    },
    'Tomato___Early_blight': {
        'name': 'Early Blight', 'crop': 'Tomato',
        'pathogen': 'Fungus (Alternaria solani)',
        'symptoms': 'Dark brown concentric ring spots on lower leaves; yellowing around lesions.',
        'spread': 'Soil-borne; splash dispersal by rain or irrigation.',
        'treatment': 'Copper-based fungicide. Remove affected leaves. Stake plants for airflow. Rotate crops.',
        'severity': 60, 'severity_label': 'Moderate-High'
    },
    'Tomato___Late_blight': {
        'name': 'Late Blight', 'crop': 'Tomato',
        'pathogen': 'Water mold (Phytophthora infestans)',
        'symptoms': 'Greasy gray-green patches that turn brown; white mold on leaf undersides.',
        'spread': 'Wind and water; extremely rapid in cool, wet weather.',
        'treatment': 'Remove and destroy infected plants. Apply fungicide immediately. Improve drainage.',
        'severity': 95, 'severity_label': 'Severe'
    },
    'Tomato___healthy': {
        'name': 'Healthy Tomato', 'crop': 'Tomato',
        'pathogen': 'None',
        'symptoms': 'Deep green leaves, no spots, sturdy stems, regular flowering.',
        'spread': 'N/A',
        'treatment': 'Continue regular care: water at base, mulch, monitor weekly.',
        'severity': 0, 'severity_label': 'None'
    },
    'Potato___Early_blight': {
        'name': 'Early Blight', 'crop': 'Potato',
        'pathogen': 'Fungus (Alternaria solani)',
        'symptoms': 'Dark brown spots with concentric rings on older leaves.',
        'spread': 'Soil and crop debris; splash dispersal.',
        'treatment': 'Apply fungicide. Remove affected leaves. Rotate crops. Avoid overhead watering.',
        'severity': 55, 'severity_label': 'Moderate'
    },
    'Potato___Late_blight': {
        'name': 'Late Blight', 'crop': 'Potato',
        'pathogen': 'Water mold (Phytophthora infestans)',
        'symptoms': 'Water-soaked lesions on leaves, white mold on undersides, tuber rot.',
        'spread': 'Wind and water; can destroy a field in days.',
        'treatment': 'Remove and destroy affected plants. Apply fungicide immediately. Improve drainage.',
        'severity': 95, 'severity_label': 'Severe'
    },
    'Potato___healthy': {
        'name': 'Healthy Potato', 'crop': 'Potato',
        'pathogen': 'None',
        'symptoms': 'Uniform green leaves, sturdy stems, no lesions.',
        'spread': 'N/A',
        'treatment': 'Continue regular care: hilling, watering, monitoring.',
        'severity': 0, 'severity_label': 'None'
    },
    'Apple___Apple_scab': {
        'name': 'Apple Scab', 'crop': 'Apple',
        'pathogen': 'Fungus (Venturia inaequalis)',
        'symptoms': 'Olive-green to black velvety spots on leaves and fruit.',
        'spread': 'Overwinters in fallen leaves; spring rains release spores.',
        'treatment': 'Apply fungicide (myclobutanil, captan). Remove fallen leaves. Prune for airflow.',
        'severity': 60, 'severity_label': 'Moderate-High'
    },
    'Apple___healthy': {
        'name': 'Healthy Apple', 'crop': 'Apple',
        'pathogen': 'None',
        'symptoms': 'Clean leaves, no spots, healthy fruit development.',
        'spread': 'N/A',
        'treatment': 'Continue regular care: pruning, monitoring, balanced fertilizer.',
        'severity': 0, 'severity_label': 'None'
    },
    'Grape___Black_rot': {
        'name': 'Black Rot', 'crop': 'Grape',
        'pathogen': 'Fungus (Guignardia bidwellii)',
        'symptoms': 'Brown circular leaf spots with dark borders; shriveled black berries.',
        'spread': 'Warm, humid conditions; overwinters in mummified fruit.',
        'treatment': 'Remove affected fruit and canes. Apply fungicide. Prune for airflow.',
        'severity': 75, 'severity_label': 'High'
    },
    'Grape___healthy': {
        'name': 'Healthy Grape', 'crop': 'Grape',
        'pathogen': 'None',
        'symptoms': 'Vigorous vines, clean leaves, healthy fruit clusters.',
        'spread': 'N/A',
        'treatment': 'Continue regular care: canopy management, monitoring.',
        'severity': 0, 'severity_label': 'None'
    },
}

def get_encyclopedia(disease):
    if disease in ENCYCLOPEDIA:
        return ENCYCLOPEDIA[disease]
    return {
        'name': disease.replace('___', ' • ').replace('_', ' '),
        'crop': disease.split('___')[0].replace('_', ' ') if '___' in disease else 'Unknown',
        'pathogen': 'See agricultural extension officer for details.',
        'symptoms': 'Visible lesions, discoloration, or spots on leaves.',
        'spread': 'Varies by pathogen.',
        'treatment': get_treatment(disease, 'en'),
        'severity': 50, 'severity_label': 'Moderate'
    }

def get_severity(disease):
    d = disease.lower()
    if 'healthy' in d:
        return ('healthy', '#2D6A4F', '#D8F3DC')
    if any(x in d for x in ['late_blight', 'yellow_leaf_curl', 'mosaic_virus', 'greening']):
        return ('severe', '#BC4749', '#F8D7DA')
    if any(x for x in ['early_blight', 'black_rot', 'rust', 'blight'] if x in d):
        return ('moderate', '#E9A319', '#FFF3CD')
    return ('mild', '#D4A373', '#FDEBD0')

# ========== PREDICTION (FIXED: NO /255 SCALING) ==========
def predict_disease(image):
    img = image.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, 0)
    predictions = model.predict(img_array, verbose=0)
    top_3_idx = np.argsort(predictions[0])[-3:][::-1]
    return [{'disease': CLASS_NAMES[i], 'confidence': round(float(100 * predictions[0][i]), 2)} for i in top_3_idx]

def save_to_history(image_name, disease, confidence):
    file_exists = os.path.isfile(HISTORY_FILE)
    with open(HISTORY_FILE, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Date', 'Image', 'Disease', 'Confidence'])
        writer.writerow([datetime.now().strftime('%Y-%m-%d %H:%M'), image_name, disease, f'{confidence:.1f}%'])

def get_stats():
    if not os.path.isfile(HISTORY_FILE):
        return {'total': 0, 'top_disease': '—'}
    diseases = {}
    total = 0
    with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            diseases[row['Disease']] = diseases.get(row['Disease'], 0) + 1
            total += 1
    top = max(diseases.items(), key=lambda x: x[1])[0] if diseases else '—'
    return {'total': total, 'top_disease': top}

# ========== SIDEBAR ==========
with st.sidebar:
    st.markdown("### 🌿 Tsana Africa")
    st.caption("AI for Agriculture")
    st.markdown("---")

    language_label = st.selectbox(
        "🌍 Language",
        ["English", "Sesotho", "Français", "isiZulu", "isiXhosa", "Afrikaans"]
    )
    lang_map = {
        "English": "en", "Sesotho": "st", "Français": "fr",
        "isiZulu": "zu", "isiXhosa": "xh", "Afrikaans": "af"
    }
    current_language = lang_map[language_label]

    st.markdown("---")
    st.markdown("### 📊 Live Stats")
    stats = get_stats()
    st.metric("Total Diagnoses", stats['total'])
    if stats['top_disease'] != '—':
        top_display = get_display_name(stats['top_disease'], current_language)
        st.metric("Top Disease", top_display[:20])

    st.markdown("---")
    st.markdown("### 🌍 Offline-First")
    st.caption("This AI runs on your device. Perfect for rural farmers without internet.")
    st.markdown("---")
    st.markdown("**Built by Motlatsi Mosiuoa**")
    st.caption("Lesotho • 2026")

# ========== HERO SECTION ==========
st.markdown("""<div class="hero">
<div class="hero-logo">🌿</div>
<div class="hero-title">Crop Disease AI Detector</div>
<div class="hero-subtitle">Put an agricultural expert in every farmer's pocket</div>
<span class="hero-badge">🤖 AI-Powered</span>
<span class="hero-badge">📶 Works Offline</span>
<span class="hero-badge">🌍 6 Languages</span>
</div>""", unsafe_allow_html=True)

# ========== MAIN TABS ==========
tab1, tab2, tab3, tab4 = st.tabs(["🔬 Diagnose", "📚 Encyclopedia", "📊 How It Works", "🌾 Why It Matters"])

# ========== TAB 1: DIAGNOSE ==========
with tab1:
    uploaded_file = st.file_uploader(
        "📷 Upload a crop leaf image (JPG, PNG)",
        type=['jpg', 'jpeg', 'png', 'bmp'],
        label_visibility="visible"
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

        col1, col2 = st.columns([1, 1], gap="large")

        with col1:
            st.markdown("#### 📸 Your Image")
            st.image(image, use_container_width=True)

        with col2:
            st.markdown("#### 🔬 Diagnosis")

            progress_bar = st.progress(0)
            status_text = st.empty()

            steps = [
                ("Loading image...", 15),
                ("Preparing pixels...", 35),
                ("Running neural network...", 65),
                ("Analyzing patterns...", 85),
                ("Finalizing diagnosis...", 100),
            ]
            for msg, pct in steps:
                status_text.caption(f"🌿 {msg}")
                progress_bar.progress(pct)
                time.sleep(0.15)

            results = predict_disease(image)
            status_text.empty()
            progress_bar.empty()

            disease = results[0]['disease']
            confidence = results[0]['confidence']
            severity_key, severity_color, severity_bg = get_severity(disease)
            display_name = get_display_name(disease, current_language)
            treatment = get_treatment(disease, current_language)
            enc = get_encyclopedia(disease)

            if confidence >= 85:
                conf_class = "conf-high"
                conf_text = "High Confidence"
            elif confidence >= 60:
                conf_class = "conf-mod"
                conf_text = "Moderate Confidence"
            else:
                conf_class = "conf-low"
                conf_text = "Low Confidence"

            severity_labels = {
                'healthy': 'Healthy 🌱',
                'mild': 'Mild 🟡',
                'moderate': 'Moderate 🟠',
                'severe': 'Severe 🔴'
            }

            result_html = f"""<div class="result-card">
<div style="font-size: 14px; color: #636E72; text-transform: uppercase; letter-spacing: 1px; font-weight: 600;">Detected Disease</div>
<div class="disease-name">🌿 {display_name}</div>
<div>
<span class="confidence-badge {conf_class}">📊 {confidence:.1f}%</span>
<span style="background: {severity_bg}; color: {severity_color}; padding: 6px 16px; border-radius: 50px; font-weight: 600; font-size: 14px; margin-left: 10px;">{severity_labels[severity_key]}</span>
</div>
<div style="margin-top: 8px; font-size: 13px; color: #636E72; font-style: italic;">{conf_text}</div>
<div class="severity-container">
<div class="severity-label-row">
<span>Severity Level</span>
<span>{enc['severity']}% — {enc['severity_label']}</span>
</div>
<div class="severity-bar-bg">
<div class="severity-bar-fill" style="width: {enc['severity']}%; background: linear-gradient(90deg, {severity_color}, #E9A319);"></div>
</div>
</div>
</div>"""

            st.markdown(result_html, unsafe_allow_html=True)

            if confidence < 50:
                st.error("⚠ **Low confidence.** Please retake the photo with better lighting and a clear view of the leaf.")

            st.markdown("#### 💊 Treatment Recommendation")
            st.info(treatment)

        st.markdown("---")
        st.markdown("#### 🔍 Other Possibilities")
        cols = st.columns(2)
        for i, r in enumerate(results[1:]):
            name = r['disease'].replace('___', ' • ').replace('_', ' ')
            with cols[i]:
                st.markdown(f"""<div style="background: white; padding: 15px; border-radius: 12px; border: 1px solid #E8E5DE;">
<div style="font-weight: 600; color: #1B4332;">🌱 {name}</div>
<div style="color: #636E72; font-size: 14px;">{r['confidence']:.1f}% confidence</div>
</div>""", unsafe_allow_html=True)

        save_to_history(uploaded_file.name, disease, confidence)

        st.markdown("---")
        st.info(f"📚 **Want to learn more about {display_name}?** Open the **Encyclopedia** tab above.")

    else:
        st.markdown("""<div style="text-align: center; padding: 40px; color: #636E72;">
<div style="font-size: 60px; opacity: 0.4;">📷</div>
<h3 style="color: #1B4332;">Ready to diagnose</h3>
<p>Upload a clear photo of a crop leaf to get started</p>
</div>""", unsafe_allow_html=True)

        st.markdown("<div class='section-header'>Why farmers love it</div>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("""<div class="feature-card">
<div class="feature-icon">⚡</div>
<div class="feature-title">Instant Results</div>
<div class="feature-desc">Get a diagnosis in seconds — no waiting, no travel to extension offices.</div>
</div>""", unsafe_allow_html=True)
        with c2:
            st.markdown("""<div class="feature-card">
<div class="feature-icon">📶</div>
<div class="feature-title">Works Offline</div>
<div class="feature-desc">The AI runs on your device. No internet needed after loading.</div>
</div>""", unsafe_allow_html=True)
        with c3:
            st.markdown("""<div class="feature-card">
<div class="feature-icon">🌍</div>
<div class="feature-title">6 Languages</div>
<div class="feature-desc">Available in English, Sesotho, French, isiZulu, isiXhosa, and Afrikaans.</div>
</div>""", unsafe_allow_html=True)

# ========== TAB 2: ENCYCLOPEDIA ==========
with tab2:
    st.markdown("<div class='section-header'>Crop Disease Encyclopedia</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-subtitle'>Detailed information about each disease — symptoms, causes, and treatment</div>", unsafe_allow_html=True)

    crops = sorted(set([enc['crop'] for enc in ENCYCLOPEDIA.values()]))
    selected_crop = st.selectbox("🌾 Filter by crop", ["All"] + crops)

    filtered_entries = {
        k: v for k, v in ENCYCLOPEDIA.items()
        if selected_crop == "All" or v['crop'] == selected_crop
    }

    st.markdown(f"**Showing {len(filtered_entries)} disease entries**")
    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    for disease_key, enc in filtered_entries.items():
        with st.expander(f"🌿 **{enc['name']}** — {enc['crop']}"):
            sev_html = f"""<div class="severity-container">
<div class="severity-label-row">
<span>Severity</span>
<span>{enc['severity']}% — {enc['severity_label']}</span>
</div>
<div class="severity-bar-bg">
<div class="severity-bar-fill" style="width: {enc['severity']}%; background: linear-gradient(90deg, #52B788, #E9A319);"></div>
</div>
</div>"""
            st.markdown(sev_html, unsafe_allow_html=True)

            card_html = f"""<div class="encyclopedia-card">
<div class="encyclopedia-row">
<div class="encyclopedia-key">Crop</div>
<div class="encyclopedia-val">{enc['crop']}</div>
</div>
<div class="encyclopedia-row">
<div class="encyclopedia-key">Pathogen</div>
<div class="encyclopedia-val">{enc['pathogen']}</div>
</div>
<div class="encyclopedia-row">
<div class="encyclopedia-key">Symptoms</div>
<div class="encyclopedia-val">{enc['symptoms']}</div>
</div>
<div class="encyclopedia-row">
<div class="encyclopedia-key">How it spreads</div>
<div class="encyclopedia-val">{enc['spread']}</div>
</div>
<div class="encyclopedia-row">
<div class="encyclopedia-key">Treatment</div>
<div class="encyclopedia-val">{enc['treatment']}</div>
</div>
</div>"""
            st.markdown(card_html, unsafe_allow_html=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.markdown("### 📋 Full List — All 38 Detectable Conditions")

    diseases_summary = {
        "🌽 Maize (Corn)": ["Northern Leaf Blight", "Common Rust", "Cercospora Leaf Spot", "Healthy"],
        "🍅 Tomato": ["Early Blight", "Late Blight", "Bacterial Spot", "Leaf Mold", "Septoria", "Target Spot", "Spider Mites", "Mosaic Virus", "Yellow Leaf Curl", "Healthy"],
        "🥔 Potato": ["Early Blight", "Late Blight", "Healthy"],
        "🍎 Apple": ["Apple Scab", "Black Rot", "Cedar Apple Rust", "Healthy"],
        "🍇 Grape": ["Black Rot", "Esca", "Leaf Blight", "Healthy"],
        "🍓 Strawberry": ["Leaf Scorch", "Healthy"],
        "🍑 Peach": ["Bacterial Spot", "Healthy"],
        "🍊 Orange": ["Citrus Greening (HLB)"],
        "🌶 Pepper": ["Bacterial Spot", "Healthy"],
        "🫐 Blueberry": ["Healthy"],
        "🌱 Raspberry": ["Healthy"],
        "🌾 Soybean": ["Healthy"],
        "🎃 Squash": ["Powdery Mildew"],
    }

    for crop, list_diseases in diseases_summary.items():
        with st.expander(f"{crop} — {len(list_diseases)} conditions"):
            for d in list_diseases:
                if "Healthy" in d:
                    st.markdown(f"- ✅ **{d}**")
                else:
                    st.markdown(f"- ⚠ {d}")

# ========== TAB 3: HOW IT WORKS ==========
with tab3:
    st.markdown("<div class='section-header'>How It Works</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-subtitle'>Three simple steps to diagnose any crop</div>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""<div class="feature-card">
<div class="feature-icon">📷</div>
<div class="feature-title">1. Take a Photo</div>
<div class="feature-desc">Use your phone or camera to capture a clear image of the sick leaf.</div>
</div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="feature-card">
<div class="feature-icon">🤖</div>
<div class="feature-title">2. AI Analyzes</div>
<div class="feature-desc">A trained neural network examines the leaf against 38 known diseases.</div>
</div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class="feature-card">
<div class="feature-icon">💊</div>
<div class="feature-title">3. Get Treatment</div>
<div class="feature-desc">Receive a clear diagnosis and recommended treatment in your language.</div>
</div>""", unsafe_allow_html=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>Trained On 54,305 Images</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-subtitle'>94.6% accuracy across 38 disease classes</div>", unsafe_allow_html=True)

    st.markdown("""<div class="stat-card" style="margin-top: 20px;">
<div class="stat-number">54,305</div>
<div class="stat-label">Training Images</div>
<div style="font-size: 13px; color: #636E72; margin-top: 8px;">
Curated from the PlantVillage dataset, covering 38 crop disease classes across 13 crop types.
</div>
</div>""", unsafe_allow_html=True)

# ========== TAB 4: WHY IT MATTERS ==========
with tab4:
    st.markdown("<div class='section-header'>Why This Matters</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-subtitle'>The challenge — and the opportunity</div>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("""<div class="stat-card">
<div class="stat-number">60%</div>
<div class="stat-label">Harvest Lost</div>
<div style="font-size: 13px; color: #636E72; margin-top: 8px;">
Farmers lose up to 60% of their crops to preventable diseases.
</div>
</div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="stat-card">
<div class="stat-number">75%</div>
<div class="stat-label">Rural Population</div>
<div style="font-size: 13px; color: #636E72; margin-top: 8px;">
75% of Lesotho's population depends on agriculture.
</div>
</div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class="stat-card">
<div class="stat-number">1:500</div>
<div class="stat-label">Extension Officers</div>
<div style="font-size: 13px; color: #636E72; margin-top: 8px;">
Too few officers to reach every village in time.
</div>
</div>""", unsafe_allow_html=True)
    with c4:
        st.markdown("""<div class="stat-card">
<div class="stat-number">94%</div>
<div class="stat-label">AI Accuracy</div>
<div style="font-size: 13px; color: #636E72; margin-top: 8px;">
Our model detects 38 crop diseases with 94.6% accuracy.
</div>
</div>""", unsafe_allow_html=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    st.markdown("""<div style="background: linear-gradient(135deg, #1B4332, #2D6A4F); padding: 50px; border-radius: 20px; color: white; text-align: center;">
<div style="font-size: 40px; margin-bottom: 15px;">🌿</div>
<div style="font-size: 32px; font-weight: 800; letter-spacing: -0.5px; margin-bottom: 15px;">
An expert in every farmer's pocket
</div>
<div style="font-size: 17px; opacity: 0.9; max-width: 700px; margin: 0 auto; line-height: 1.7;">
By combining AI, mobile technology, and local languages, we're making
agricultural expertise accessible to every farmer in Lesotho —
regardless of internet access, literacy, or location.
</div>
</div>""", unsafe_allow_html=True)

# ========== HISTORY ==========
st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
with st.expander("📜 View Diagnosis History"):
    if os.path.isfile(HISTORY_FILE):
        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            content = f.read()
        st.code(content, language=None)

        st.download_button(
            "⬇️  Download History CSV",
            data=content,
            file_name=f"diagnosis_history_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    else:
        st.caption("No history yet — your diagnoses will appear here.")

# ========== FOOTER ==========
st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
st.markdown("""<div style="text-align: center; padding: 30px; color: #636E72;">
<div style="font-size: 24px; margin-bottom: 8px;">🌿</div>
<div style="font-weight: 700; color: #1B4332; font-size: 16px;">Tsana Africa</div>
<div style="font-size: 13px; margin-top: 4px;">AI for Agriculture  •  Lesotho  •  2026</div>
<div style="font-size: 12px; margin-top: 10px; opacity: 0.7;">
Built with TensorFlow  •  Streamlit  •  Powered by a neural network trained on 54,305 crop images
</div>
</div>""", unsafe_allow_html=True)