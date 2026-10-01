import tkinter as tk
from tkinter import filedialog, Label, Button, Frame, Text, Scrollbar, messagebox, Canvas, Toplevel
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np
import csv
import os
import threading
import time
from datetime import datetime

# ========== CONFIGURATION ==========
MODEL_PATH = 'crop_disease_model.h5'
HISTORY_FILE = 'diagnosis_history.csv'
REPORT_FOLDER = 'diagnosis_reports'
THUMB_FOLDER = 'thumbnails'

for folder in [REPORT_FOLDER, THUMB_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# ========== COLOR PALETTE ==========
COLORS = {
    'bg':          '#FAF9F6',
    'card':        '#FFFFFF',
    'primary':     '#1B4332',
    'secondary':   '#52B788',
    'accent':      '#D4A373',
    'accent_dark': '#B07D4F',
    'text':        '#2D3436',
    'text_muted':  '#636E72',
    'border':      '#E8E5DE',
    'success':     '#2D6A4F',
    'warning':     '#E9A319',
    'danger':      '#BC4749',
    'splash_bg':   '#1B4332',
}

FONT_HEADING = ('Segoe UI', 20, 'bold')
FONT_TITLE   = ('Segoe UI', 26, 'bold')
FONT_SUBHEAD = ('Segoe UI', 13, 'bold')
FONT_BODY    = ('Segoe UI', 11)
FONT_SMALL   = ('Segoe UI', 9)
FONT_BUTTON  = ('Segoe UI', 11, 'bold')

# ========== LOAD MODEL ==========
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    MODEL_LOADED = True
except Exception as e:
    MODEL_LOADED = False
    print(f"Error loading model: {e}")

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

# ========== FULL LANGUAGE STRINGS ==========
LANG = {
    'en': {
        'app_title': 'Crop Disease Detector',
        'tagline': 'AI for Agriculture  •  Tsana Africa',
        'welcome_title': 'Welcome to Crop Disease Detector',
        'welcome_text': 'Instantly diagnose crop diseases using AI.\n\nHow it works:\n\n   1.  Upload or take a photo of a sick crop leaf\n   2.  The AI identifies the disease in seconds\n   3.  Receive treatment recommendations\n\nWorks offline — perfect for rural farmers.',
        'get_started': 'Get Started',
        'upload_btn': '📷   Upload Leaf Image',
        'no_image': 'No image selected\n\nClick below to upload a leaf photo',
        'awaiting': 'Awaiting Diagnosis',
        'upload_hint': 'Upload an image to receive a diagnosis.',
        'analyzing': 'Analyzing...',
        'please_wait': 'Please wait while the AI examines the leaf.',
        'treatment': 'Treatment Recommendation:',
        'other_poss': 'Other Possibilities:',
        'high_conf': 'High Confidence',
        'mod_conf': 'Moderate Confidence',
        'low_conf': 'Low Confidence',
        'warning_low': '⚠  Low confidence. Please retake the photo with better lighting and a clear view of the leaf.',
        'severity': 'Severity',
        'healthy': 'Healthy',
        'mild': 'Mild',
        'moderate': 'Moderate',
        'severe': 'Severe',
        'speak': '🔊  Speak',
        'export': '📄  Export Report',
        'history': '📜  History',
        'stats': '📊  Statistics',
        'footer': 'Built by Tsana Africa  •  AI for Agriculture  •  Lesotho',
        'lang_label': 'Language',
        'no_history': 'No history yet.',
        'no_data': 'No data yet.',
        'stats_title': 'Statistics Dashboard',
        'total_diag': 'Total Diagnoses:',
        'install_tts': 'Install pyttsx3:\npip install pyttsx3',
        'history_title': 'Diagnosis History',
        'close': 'Close',
        'details': 'Details',
        'full_report': 'Full Report',
    },
    'st': {
        'app_title': 'Sesebelisoa sa Ho Fumana Mafu a Lijalo',
        'tagline': 'AI bakeng sa Temo  •  Tsana Africa',
        'welcome_title': 'Rea u amohela ho Sesebelisoa sa Ho Fumana Mafu a Lijalo',
        'welcome_text': 'Fumana mafu a lijalo kapele ka AI.\n\nKamoo e sebetsang:\n\n   1.  Kenya setšoantšo sa lekhasi le kulang\n   2.  AI e tseba lefu ka motsotsoana\n   3.  Fumana pheko e khothalletsoang\n\nE sebetsa ntle le inthanete — e loketse lihoai tsa mahaeng.',
        'get_started': 'Qala Hona Joale',
        'upload_btn': '📷   Kenya Setšoantšo sa Lekhasi',
        'no_image': 'Ha ho setšoantšo se khethiloeng\n\nTobetsa ka tlase ho kenya setšoantšo',
        'awaiting': 'E emetse tlhahlobo',
        'upload_hint': 'Kenya setšoantšo ho fumana tlhahlobo.',
        'analyzing': 'Ea hlahloba...',
        'please_wait': 'Ka kopo ema ha AI e hlahloba lekhasi.',
        'treatment': 'Pheko e Khothalletsoang:',
        'other_poss': 'Menahano e Meng:',
        'high_conf': 'Tšepo e Phahameng',
        'mod_conf': 'Tšepo e Mahareng',
        'low_conf': 'Tšepo e Tlaase',
        'warning_low': '⚠  Tšepo e tlaase. Ka kopo nka setšoantšo se ncha ka leseli le letle.',
        'severity': 'Boemo',
        'healthy': 'E Phetse',
        'mild': 'Hanyane',
        'moderate': 'Mahareng',
        'severe': 'Haholo',
        'speak': '🔊  Bua',
        'export': '📄  Romela Tlaleho',
        'history': '📜  Nalane',
        'stats': '📊  Lipalo-palo',
        'footer': 'E entsoe ke Tsana Africa  •  AI bakeng sa Temo  •  Lesotho',
        'lang_label': 'Puo',
        'no_history': 'Ha ho nalane hajoale.',
        'no_data': 'Ha ho lintlha hajoale.',
        'stats_title': 'Letlapa la Lipalo-palo',
        'total_diag': 'Kakaretso ea Tlhahlobo:',
        'install_tts': 'Kenya pyttsx3:\npip install pyttsx3',
        'history_title': 'Nalane ea Tlhahlobo',
        'close': 'Koala',
        'details': 'Lintlha',
        'full_report': 'Tlaleho e Felletseng',
    },
    'fr': {
        'app_title': 'Détecteur de Maladies des Cultures',
        'tagline': 'IA pour l\'Agriculture  •  Tsana Africa',
        'welcome_title': 'Bienvenue dans le Détecteur de Maladies des Cultures',
        'welcome_text': 'Diagnostiquez instantanément les maladies des cultures grâce à l\'IA.\n\nComment ça marche :\n\n   1.  Téléchargez ou prenez une photo d\'une feuille malade\n   2.  L\'IA identifie la maladie en quelques secondes\n   3.  Recevez des recommandations de traitement\n\nFonctionne hors ligne — parfait pour les agriculteurs ruraux.',
        'get_started': 'Commencer',
        'upload_btn': '📷   Télécharger une Image',
        'no_image': 'Aucune image sélectionnée\n\nCliquez ci-dessous pour télécharger une photo',
        'awaiting': 'En attente de diagnostic',
        'upload_hint': 'Téléchargez une image pour recevoir un diagnostic.',
        'analyzing': 'Analyse en cours...',
        'please_wait': 'Veuillez patienter pendant que l\'IA examine la feuille.',
        'treatment': 'Traitement Recommandé :',
        'other_poss': 'Autres Possibilités :',
        'high_conf': 'Confiance Élevée',
        'mod_conf': 'Confiance Modérée',
        'low_conf': 'Faible Confiance',
        'warning_low': '⚠  Faible confiance. Veuillez reprendre la photo avec un meilleur éclairage.',
        'severity': 'Gravité',
        'healthy': 'Saine',
        'mild': 'Légère',
        'moderate': 'Modérée',
        'severe': 'Sévère',
        'speak': '🔊  Parler',
        'export': '📄  Exporter le Rapport',
        'history': '📜  Historique',
        'stats': '📊  Statistiques',
        'footer': 'Créé par Tsana Africa  •  IA pour l\'Agriculture  •  Lesotho',
        'lang_label': 'Langue',
        'no_history': 'Aucun historique pour l\'instant.',
        'no_data': 'Aucune donnée pour l\'instant.',
        'stats_title': 'Tableau de Bord Statistique',
        'total_diag': 'Diagnostics Totaux :',
        'install_tts': 'Installer pyttsx3:\npip install pyttsx3',
        'history_title': 'Historique des Diagnostics',
        'close': 'Fermer',
        'details': 'Détails',
        'full_report': 'Rapport Complet',
    },
    'zu': {
        'app_title': 'Isitholi Sezifo Sezitshalo',
        'tagline': 'I-AI Yezolimo  •  Tsana Africa',
        'welcome_title': 'Siyakwamukela ku-Isitholi Sezifo Sezitshalo',
        'welcome_text': 'Hlonza izifo zezitshalo ngokushesha usebenzisa i-AI.\n\nKusebenza kanjani:\n\n   1.  Layisha noma thatha isithombe seqabunga eligulayo\n   2.  I-AI ibona isifo ngemizuzwana\n   3.  Thola izincomo zokwelapha\n\nIsebenza ngaphandle kwe-inthanethi — ilungele abalimi basemakhaya.',
        'get_started': 'Qala Manje',
        'upload_btn': '📷   Layisha Isithombe',
        'no_image': 'Ayikho isithombe esikhethiwe\n\nChofoza ngezansi ukulayisha isithombe',
        'awaiting': 'Ilinde ukuhlolwa',
        'upload_hint': 'Layisha isithombe ukuthola ukuhlolwa.',
        'analyzing': 'Iyahlola...',
        'please_wait': 'Sicela ulinde ngenkathi i-AI ihlola iqabunga.',
        'treatment': 'Isincomo Sokwelapha:',
        'other_poss': 'Amanye Amathuba:',
        'high_conf': 'Ukuqiniseka Okuphezulu',
        'mod_conf': 'Ukuqiniseka Okuphakathi',
        'low_conf': 'Ukuqiniseka Okuphansi',
        'warning_low': '⚠  Ukuqiniseka okuphansi. Sicela uthathe isithombe esisha ngokukhanya okungcono.',
        'severity': 'Ububi',
        'healthy': 'Inempilo',
        'mild': 'Kancane',
        'moderate': 'Phakathi',
        'severe': 'Kubi Kakhulu',
        'speak': '🔊  Khuluma',
        'export': '📄  Thumela Umbiko',
        'history': '📜  Umlando',
        'stats': '📊  Izibalo',
        'footer': 'Yakhiwe yi-Tsana Africa  •  I-AI Yezolimo  •  Lesotho',
        'lang_label': 'Ulimi',
        'no_history': 'Awukho umlando okwamanje.',
        'no_data': 'Ayikho idatha okwamanje.',
        'stats_title': 'Idashibhodi Yezibalo',
        'total_diag': 'Ukuhlolwa Konke:',
        'install_tts': 'Faka pyttsx3:\npip install pyttsx3',
        'history_title': 'Umlando Wokuhlolwa',
        'close': 'Vala',
        'details': 'Imininingwane',
        'full_report': 'Umbiko Ogcwele',
    },
    'xh': {
        'app_title': 'Isifumanisi Sezifo Zezityalo',
        'tagline': 'I-AI Yezolimo  •  Tsana Africa',
        'welcome_title': 'Wamkelekile kwi-Isifumanisi Sezifo Zezityalo',
        'welcome_text': 'Fumana izifo zezityalo ngokukhawuleza usebenzisa i-AI.\n\nIsebenza njani:\n\n   1.  Layisha okanye thatha ifoto yegqabi eligulayo\n   2.  I-AI ichonga isifo ngemizuzwana\n   3.  Fumana iingcebiso zonyango\n\nIsebenza ngaphandle kwe-intanethi — ilungele amafama asemaphandleni.',
        'get_started': 'Qala Ngoku',
        'upload_btn': '📷   Layisha Umfanekiso',
        'no_image': 'Akukho mfanekiso ukhethiweyo\n\nCofa ngezantsi ukulayisha ifoto',
        'awaiting': 'Ilinde uhlolo',
        'upload_hint': 'Layisha umfanekiso ukufumana uhlolo.',
        'analyzing': 'Iyahlola...',
        'please_wait': 'Nceda linda ngeli xesha i-AI ihlola igqabi.',
        'treatment': 'Ingcebiso Yonyango:',
        'other_poss': 'Amanye Amathuba:',
        'high_conf': 'Ukuqiniseka Okuphezulu',
        'mod_conf': 'Ukuqiniseka Okuphakathi',
        'low_conf': 'Ukuqiniseka Okuphantsi',
        'warning_low': '⚠  Ukuqiniseka okuphantsi. Nceda thatha ifoto entsha ngokukhanya okungcono.',
        'severity': 'Ubunzima',
        'healthy': 'Iphilileyo',
        'mild': 'Kancinci',
        'moderate': 'Phakathi',
        'severe': 'Kubi Kakhulu',
        'speak': '🔊  Thetha',
        'export': '📄  Thumela Ingxelo',
        'history': '📜  Imbali',
        'stats': '📊  Izibalo',
        'footer': 'Yenziwe yi-Tsana Africa  •  I-AI Yezolimo  •  Lesotho',
        'lang_label': 'Ulwimi',
        'no_history': 'Akukho mbali okwangoku.',
        'no_data': 'Akukho datha okwangoku.',
        'stats_title': 'Idashbhodi Yezibalo',
        'total_diag': 'Uhlolo Lulonke:',
        'install_tts': 'Faka pyttsx3:\npip install pyttsx3',
        'history_title': 'Imbali Yohlolo',
        'close': 'Vala',
        'details': 'Iinkcukacha',
        'full_report': 'Ingxelo Epheleleyo',
    },
    'af': {
        'app_title': 'Gewas Siekte Detektor',
        'tagline': 'KI vir Landbou  •  Tsana Africa',
        'welcome_title': 'Welkom by Gewas Siekte Detektor',
        'welcome_text': 'Diagnoseer gewassiektes onmiddellik met KI.\n\nHoe dit werk:\n\n   1.  Laai of neem \'n foto van \'n siek blaar\n   2.  Die KI identifiseer die siekte in sekondes\n   3.  Ontvang behandelingsaanbevelings\n\nWerk aanlyn — perfek vir landelike boere.',
        'get_started': 'Begin Nou',
        'upload_btn': '📷   Laai Blaarbeeld',
        'no_image': 'Geen beeld gekies nie\n\nKlik hieronder om \'n foto te laai',
        'awaiting': 'Wag vir diagnose',
        'upload_hint': 'Laai \'n beeld om \'n diagnose te ontvang.',
        'analyzing': 'Ontleed...',
        'please_wait': 'Wag asseblief terwyl die KI die blaar ondersoek.',
        'treatment': 'Behandelingsaanbeveling:',
        'other_poss': 'Ander Moontlikhede:',
        'high_conf': 'Hoë Vertroue',
        'mod_conf': 'Matige Vertroue',
        'low_conf': 'Lae Vertroue',
        'warning_low': '⚠  Lae vertroue. Neem asseblief \'n nuwe foto met beter beligting.',
        'severity': 'Erns',
        'healthy': 'Gesond',
        'mild': 'Effens',
        'moderate': 'Matig',
        'severe': 'Ernstig',
        'speak': '🔊  Praat',
        'export': '📄  Voer Verslag Uit',
        'history': '📜  Geskiedenis',
        'stats': '📊  Statistieke',
        'footer': 'Gebou deur Tsana Africa  •  KI vir Landbou  •  Lesotho',
        'lang_label': 'Taal',
        'no_history': 'Nog geen geskiedenis nie.',
        'no_data': 'Nog geen data nie.',
        'stats_title': 'Statistiek Paneelbord',
        'total_diag': 'Totale Diagnoses:',
        'install_tts': 'Installeer pyttsx3:\npip install pyttsx3',
        'history_title': 'Diagnose Geskiedenis',
        'close': 'Sluit',
        'details': 'Besonderhede',
        'full_report': 'Volle Verslag',
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
    'Grape___Black_rot': 'Susa iziqhamo ezichaphazelekileyo. Sebenzisa isibulali-zifo.',
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
    'en': TREATMENTS_EN,
    'st': TREATMENTS_ST,
    'fr': TREATMENTS_FR,
    'zu': TREATMENTS_ZU,
    'xh': TREATMENTS_XH,
    'af': TREATMENTS_AF,
}

FALLBACK_TREATMENT = {
    'en': 'Consult your local agricultural extension officer for treatment advice.',
    'st': 'Buisana le ofisiri ea temo ea sebaka sa heno bakeng sa thuso.',
    'fr': 'Consultez votre agent de vulgarisation agricole local.',
    'zu': 'Xhumana nomeluleki wakho wezolimo wendawo.',
    'xh': 'Qhagamshelana neGosa lakho lezolimo lendawo.',
    'af': 'Raadpleeg u plaaslike landbou-uitbreidingsbeampte.',
}

def get_treatment(disease, language='en'):
    return TREATMENT_MAP.get(language, TREATMENTS_EN).get(disease, FALLBACK_TREATMENT.get(language, FALLBACK_TREATMENT['en']))

def get_display_name(disease, language='en'):
    if language in LANG_NAMES and disease in LANG_NAMES[language]:
        return LANG_NAMES[language][disease]
    return disease.replace('___', ' • ').replace('_', ' ')

def get_severity(disease):
    d = disease.lower()
    if 'healthy' in d:
        return ('healthy', COLORS['success'])
    if any(x in d for x in ['late_blight', 'yellow_leaf_curl', 'mosaic_virus', 'greening']):
        return ('severe', COLORS['danger'])
    if any(x in d for x in ['early_blight', 'black_rot', 'rust', 'blight']):
        return ('moderate', COLORS['warning'])
    return ('mild', COLORS['accent'])

# ========== PREDICTION ==========
def predict_disease(image_path):
    img = tf.keras.utils.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    predictions = model.predict(img_array, verbose=0)
    top_3_idx = np.argsort(predictions[0])[-3:][::-1]
    return [{'disease': CLASS_NAMES[i], 'confidence': round(float(100 * predictions[0][i]), 2)} for i in top_3_idx]

def save_to_history(image_name, disease, confidence, thumb_path):
    """Save history with correct column order."""
    file_exists = os.path.isfile(HISTORY_FILE)
    with open(HISTORY_FILE, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Date', 'Image', 'Thumbnail', 'Disease', 'Confidence'])
        writer.writerow([
            datetime.now().strftime('%Y-%m-%d %H:%M'),
            image_name,
            thumb_path,
            disease,
            f'{confidence:.1f}%'
        ])

def save_thumbnail(image_path):
    try:
        img = Image.open(image_path)
        img.thumbnail((150, 150), Image.LANCZOS)
        thumb_name = f"thumb_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.png"
        thumb_path = os.path.join(THUMB_FOLDER, thumb_name)
        img.save(thumb_path)
        return thumb_path
    except Exception as e:
        print(f"Thumbnail error: {e}")
        return ""

def export_report(image_path, results):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = os.path.join(REPORT_FOLDER, f'report_{timestamp}.txt')
    severity_key, _ = get_severity(results[0]['disease'])
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 55 + "\n")
        f.write("       AI CROP DISEASE DETECTOR - DIAGNOSIS REPORT\n")
        f.write("              Tsana Africa | AI for Agriculture\n")
        f.write("=" * 55 + "\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Image: {os.path.basename(image_path)}\n")
        f.write(f"Severity: {severity_key.upper()}\n\n")
        f.write("TOP PREDICTIONS:\n")
        for i, r in enumerate(results, 1):
            f.write(f"  {i}. {r['disease']} - {r['confidence']:.1f}%\n")
        f.write(f"\nRECOMMENDED TREATMENT:\n  {get_treatment(results[0]['disease'], 'en')}\n\n")
        f.write("=" * 55 + "\n")
    messagebox.showinfo("Report Saved", f"Report saved:\n{report_path}")

def speak_text(text):
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    except ImportError:
        messagebox.showinfo("Voice", LANG[current_language]['install_tts'])

# ========== STATE ==========
current_language = 'en'
last_image_path = None
last_results = None

# ========== LANGUAGE SELECTOR ==========
LANGUAGES = {
    'English': 'en',
    'Sesotho': 'st',
    'Français': 'fr',
    'isiZulu': 'zu',
    'isiXhosa': 'xh',
    'Afrikaans': 'af',
}

def set_language(lang_code):
    global current_language
    current_language = lang_code
    translate_ui()

# ========== TRANSLATE UI ==========
def translate_ui():
    t = LANG[current_language]
    root.title(f"AI {t['app_title']} | Tsana Africa")

    header_title.config(text=f"🌿 {t['app_title']}")
    header_tagline.config(text=t['tagline'])

    welcome_title.config(text=f"🌿 {t['welcome_title']}")
    welcome_body.config(text=t['welcome_text'])
    start_button.config(text=t['get_started'])

    upload_button.config(text=t['upload_btn'])
    if last_image_path is None:
        image_label.config(text=t['no_image'], image='', width=60, height=18)
    if last_results is None:
        result_title.config(text=t['awaiting'])
        result_body.config(text=t['upload_hint'])
    speak_button.config(text=t['speak'])
    export_button.config(text=t['export'])
    history_button.config(text=t['history'])
    stats_button.config(text=t['stats'])
    footer_label.config(text=t['footer'])
    lang_label.config(text=t['lang_label'])

    if last_results is not None:
        render_results()

def render_results():
    if last_results is None:
        return
    t = LANG[current_language]
    disease = last_results[0]['disease']
    confidence = last_results[0]['confidence']

    display_name = get_display_name(disease, current_language)
    treatment = get_treatment(disease, current_language)
    severity_key, severity_color = get_severity(disease)

    if confidence >= 85:
        status = t['high_conf']
        status_color = COLORS['success']
    elif confidence >= 60:
        status = t['mod_conf']
        status_color = COLORS['accent']
    else:
        status = t['low_conf']
        status_color = COLORS['danger']

    result_title.config(text=f"🌿 {display_name}")
    confidence_label.config(text=f"{confidence:.1f}%", fg=status_color)
    status_label.config(text=status, fg=status_color)

    severity_display = t.get(severity_key, severity_key)
    severity_label.config(text=f"{t['severity']}:  {severity_display}", fg=severity_color)

    if confidence < 50:
        warning_label.config(text=t['warning_low'])
        warning_label.pack(fill="x", padx=25, pady=(0, 8))
    else:
        warning_label.pack_forget()

    body = f"{t['treatment']}\n{treatment}\n\n{t['other_poss']}\n"
    for r in last_results[1:]:
        name = r['disease'].replace('___', ' • ').replace('_', ' ')
        body += f"   •  {name}  ({r['confidence']:.1f}%)\n"

    result_body.config(text=body)

# ========== HOVER ==========
def add_hover(button, normal_bg, hover_bg):
    button.bind("<Enter>", lambda e: button.config(bg=hover_bg))
    button.bind("<Leave>", lambda e: button.config(bg=normal_bg))

# ========== UPLOAD ==========
def upload_image():
    global last_image_path
    file_path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp")])
    if not file_path:
        return
    last_image_path = file_path

    img = Image.open(file_path)
    img.thumbnail((400, 400), Image.LANCZOS)
    img_tk = ImageTk.PhotoImage(img)
    image_label.config(image=img_tk, text="", width=img.width, height=img.height)
    image_label.image = img_tk

    t = LANG[current_language]
    result_title.config(text=t['analyzing'])
    result_body.config(text=t['please_wait'])
    root.update()

    threading.Thread(target=run_prediction, args=(file_path,)).start()

def run_prediction(file_path):
    global last_results
    try:
        results = predict_disease(file_path)
        last_results = results
        disease = results[0]['disease']
        confidence = results[0]['confidence']

        thumb_path = save_thumbnail(file_path)

        root.after(0, render_results)
        root.after(0, lambda: speak_button.config(state="normal"))
        root.after(0, lambda: export_button.config(state="normal"))

        save_to_history(os.path.basename(file_path), disease, confidence, thumb_path)
    except Exception as e:
        root.after(0, lambda: result_body.config(text=f"Error: {str(e)}"))

def speak_result():
    if last_results:
        disease = last_results[0]['disease']
        name = get_display_name(disease, current_language)
        speak_text(f"{name}. {get_treatment(disease, current_language)}")

# ========== HISTORY WITH THUMBNAILS + FIXED DETAIL VIEW ==========
def view_history():
    if not os.path.isfile(HISTORY_FILE):
        messagebox.showinfo("History", LANG[current_language]['no_history'])
        return

    t = LANG[current_language]
    win = Toplevel(root)
    win.title(t['history_title'])
    win.geometry("820x660")
    win.configure(bg=COLORS['bg'])

    header = Frame(win, bg=COLORS['primary'], height=60)
    header.pack(fill="x")
    header.pack_propagate(False)
    tk.Label(header, text=f"📜 {t['history_title']}", font=('Segoe UI', 16, 'bold'),
             bg=COLORS['primary'], fg='white').pack(pady=15)

    container = Frame(win, bg=COLORS['bg'])
    container.pack(fill="both", expand=True)

    canvas_h = Canvas(container, bg=COLORS['bg'], highlightthickness=0)
    scrollbar_h = Scrollbar(container, orient="vertical", command=canvas_h.yview)
    scroll_frame = Frame(canvas_h, bg=COLORS['bg'])

    scroll_frame.bind("<Configure>", lambda e: canvas_h.configure(scrollregion=canvas_h.bbox("all")))
    canvas_window = canvas_h.create_window((0, 0), window=scroll_frame, anchor="nw")
    canvas_h.bind("<Configure>", lambda e: canvas_h.itemconfig(canvas_window, width=e.width))
    canvas_h.configure(yscrollcommand=scrollbar_h.set)

    canvas_h.pack(side="left", fill="both", expand=True)
    scrollbar_h.pack(side="right", fill="y")

    # Load all entries
    entries = []
    with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            entries.append(row)
    entries.reverse()

    thumb_refs = []

    def open_details(entry):
        """Show details of a single diagnosis with scrollable content."""
        d_win = Toplevel(win)
        d_win.title(t['details'])
        d_win.geometry("640x740")
        d_win.configure(bg=COLORS['bg'])
        d_win.minsize(500, 500)

        # Header (fixed at top)
        head = Frame(d_win, bg=COLORS['primary'], height=60)
        head.pack(fill="x")
        head.pack_propagate(False)
        tk.Label(head, text=f"📋 {t['details']}", font=('Segoe UI', 15, 'bold'),
                 bg=COLORS['primary'], fg='white').pack(pady=15)

        # Scrollable area
        container = Frame(d_win, bg=COLORS['bg'])
        container.pack(fill="both", expand=True)

        d_canvas = Canvas(container, bg=COLORS['bg'], highlightthickness=0)
        d_scroll = Scrollbar(container, orient="vertical", command=d_canvas.yview)
        d_scroll_frame = Frame(d_canvas, bg=COLORS['bg'])

        d_scroll_frame.bind(
            "<Configure>",
            lambda e: d_canvas.configure(scrollregion=d_canvas.bbox("all"))
        )

        d_canvas_window = d_canvas.create_window((0, 0), window=d_scroll_frame, anchor="nw")
        d_canvas.bind("<Configure>", lambda e: d_canvas.itemconfig(d_canvas_window, width=e.width))
        d_canvas.configure(yscrollcommand=d_scroll.set)

        d_canvas.pack(side="left", fill="both", expand=True)
        d_scroll.pack(side="right", fill="y")

        # Mouse wheel scroll for this window
        def _on_detail_mousewheel(event):
            d_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        d_canvas.bind_all("<MouseWheel>", _on_detail_mousewheel)

        # ===== IMAGE =====
        img_frame = Frame(d_scroll_frame, bg=COLORS['bg'])
        img_frame.pack(pady=15)

        thumb_path = (entry.get('Thumbnail') or '').strip()
        original = (entry.get('Image') or '').strip()
        img_loaded = False

        # Try thumbnail first
        if thumb_path and os.path.isfile(thumb_path):
            try:
                img = Image.open(thumb_path)
                img = img.resize((320, 320), Image.LANCZOS)
                tk_img = ImageTk.PhotoImage(img)
                lbl = Label(img_frame, image=tk_img, bg=COLORS['bg'])
                lbl.image = tk_img
                lbl.pack()
                img_loaded = True
            except Exception as e:
                print(f"Thumbnail load error: {e}")

        # Fallback: original image
        if not img_loaded and original and os.path.isfile(original):
            try:
                img = Image.open(original)
                img.thumbnail((320, 320), Image.LANCZOS)
                tk_img = ImageTk.PhotoImage(img)
                lbl = Label(img_frame, image=tk_img, bg=COLORS['bg'])
                lbl.image = tk_img
                lbl.pack()
                img_loaded = True
            except Exception as e:
                print(f"Original image error: {e}")

        # Final fallback
        if not img_loaded:
            Label(img_frame, text="🌿", font=('Segoe UI', 60),
                  bg=COLORS['bg'], fg=COLORS['secondary']).pack()

        # ===== INFO CARD =====
        info_frame = Frame(d_scroll_frame, bg=COLORS['card'],
                           highlightbackground=COLORS['border'], highlightthickness=1)
        info_frame.pack(fill="x", padx=25, pady=10)

        disease_raw = entry['Disease']
        disease_name = get_display_name(disease_raw, current_language)
        treatment = get_treatment(disease_raw, current_language)
        severity_key, severity_color = get_severity(disease_raw)
        severity_display = t.get(severity_key, severity_key)

        tk.Label(info_frame, text=f"🌿 {disease_name}", font=('Segoe UI', 15, 'bold'),
                 bg=COLORS['card'], fg=COLORS['primary'], anchor="w",
                 wraplength=540, justify="left").pack(fill="x", padx=20, pady=(20, 5))

        tk.Label(info_frame, text=f"📅 {entry['Date']}", font=FONT_BODY,
                 bg=COLORS['card'], fg=COLORS['text_muted'], anchor="w").pack(fill="x", padx=20)

        tk.Label(info_frame, text=f"📊 {entry['Confidence']}", font=FONT_BODY,
                 bg=COLORS['card'], fg=COLORS['text_muted'], anchor="w").pack(fill="x", padx=20)

        tk.Label(info_frame, text=f"{t['severity']}:  {severity_display}", font=FONT_BODY,
                 bg=COLORS['card'], fg=severity_color, anchor="w").pack(fill="x", padx=20, pady=(0, 10))

        tk.Label(info_frame, text=f"{t['treatment']}", font=('Segoe UI', 11, 'bold'),
                 bg=COLORS['card'], fg=COLORS['text'], anchor="w").pack(fill="x", padx=20, pady=(10, 3))

        tk.Label(info_frame, text=treatment, font=FONT_BODY,
                 bg=COLORS['card'], fg=COLORS['text'], anchor="w",
                 wraplength=540, justify="left").pack(fill="x", padx=20, pady=(0, 20))

        # ===== CLOSE BUTTON (fixed at bottom of window) =====
        bottom_frame = Frame(d_win, bg=COLORS['bg'])
        bottom_frame.pack(fill="x", pady=10)

        def close_detail():
            d_canvas.unbind_all("<MouseWheel>")
            d_win.destroy()

        Button(bottom_frame, text=t['close'], font=FONT_BUTTON, command=close_detail,
               bg=COLORS['primary'], fg='white', relief="flat",
               padx=25, pady=8, cursor="hand2", bd=0).pack()

    for entry in entries:
        card = Frame(scroll_frame, bg=COLORS['card'],
                     highlightbackground=COLORS['border'], highlightthickness=1)
        card.pack(fill="x", padx=20, pady=8)

        thumb_label = Label(card, bg=COLORS['card'])
        thumb_label.pack(side="left", padx=15, pady=15)

        thumb_path = (entry.get('Thumbnail') or '').strip()
        if thumb_path and os.path.isfile(thumb_path):
            try:
                img = Image.open(thumb_path)
                img.thumbnail((120, 120), Image.LANCZOS)
                tk_img = ImageTk.PhotoImage(img)
                thumb_label.config(image=tk_img)
                thumb_label.image = tk_img
                thumb_refs.append(tk_img)
            except Exception:
                thumb_label.config(text="🌿", font=('Segoe UI', 36), fg=COLORS['secondary'])
        else:
            thumb_label.config(text="🌿", font=('Segoe UI', 36), fg=COLORS['secondary'])

        info = Frame(card, bg=COLORS['card'])
        info.pack(side="left", fill="both", expand=True, pady=15)

        disease_name = get_display_name(entry['Disease'], current_language)
        tk.Label(info, text=disease_name, font=('Segoe UI', 12, 'bold'),
                 bg=COLORS['card'], fg=COLORS['primary'], anchor="w").pack(fill="x")

        tk.Label(info, text=f"📅 {entry['Date']}", font=FONT_SMALL,
                 bg=COLORS['card'], fg=COLORS['text_muted'], anchor="w").pack(fill="x")

        tk.Label(info, text=f"📊 {entry['Confidence']}", font=FONT_SMALL,
                 bg=COLORS['card'], fg=COLORS['text_muted'], anchor="w").pack(fill="x")

        Button(card, text=t['details'], font=('Segoe UI', 9, 'bold'),
               command=lambda e=entry: open_details(e),
               bg=COLORS['secondary'], fg='white', relief="flat",
               padx=12, pady=6, cursor="hand2", bd=0).pack(side="right", padx=15)

    Button(win, text=t['close'], font=FONT_BUTTON,
           command=win.destroy, bg=COLORS['primary'], fg='white',
           relief="flat", padx=25, pady=8, cursor="hand2", bd=0).pack(pady=10)

def view_stats():
    if not os.path.isfile(HISTORY_FILE):
        messagebox.showinfo("Stats", LANG[current_language]['no_data'])
        return
    diseases = {}
    total = 0
    with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            diseases[row['Disease']] = diseases.get(row['Disease'], 0) + 1
            total += 1
    sorted_d = sorted(diseases.items(), key=lambda x: x[1], reverse=True)

    t = LANG[current_language]
    win = Toplevel(root)
    win.title(t['stats'])
    win.geometry("540x520")
    win.configure(bg=COLORS['bg'])

    tk.Label(win, text=t['stats_title'], font=FONT_HEADING, bg=COLORS['bg'], fg=COLORS['primary']).pack(pady=15)
    tk.Label(win, text=f"{t['total_diag']} {total}", font=FONT_SUBHEAD, bg=COLORS['bg'], fg=COLORS['text']).pack()

    frame = Frame(win, bg=COLORS['card'], padx=20, pady=20)
    frame.pack(fill="both", expand=True, padx=25, pady=15)

    for disease, count in sorted_d[:10]:
        name = get_display_name(disease, current_language)
        tk.Label(frame, text=f"•  {name}", font=FONT_BODY, bg=COLORS['card'],
                 fg=COLORS['text'], anchor="w").pack(fill="x", pady=3)

# ========== SHOW MAIN UI ==========
def show_main_ui():
    welcome_frame.pack_forget()
    main_frame.pack(fill="both", expand=True)

# ========== MAIN WINDOW ==========
root = tk.Tk()
root.title("AI Crop Disease Detector | Tsana Africa")
root.geometry("820x900")
root.configure(bg=COLORS['bg'])
root.minsize(700, 600)

# ========== SPLASH SCREEN ==========
splash = Frame(root, bg=COLORS['splash_bg'])
splash.pack(fill="both", expand=True)

splash_content = Frame(splash, bg=COLORS['splash_bg'])
splash_content.place(relx=0.5, rely=0.5, anchor="center")

splash_logo_canvas = tk.Canvas(splash_content, width=140, height=140,
                               bg=COLORS['splash_bg'], highlightthickness=0)
splash_logo_canvas.pack()
splash_logo_canvas.create_oval(15, 15, 125, 125, fill=COLORS['accent'], outline="")
splash_logo_canvas.create_text(70, 70, text="🌿", font=('Segoe UI', 50))

splash_title = Label(splash_content, text="Crop Disease Detector",
                     font=('Segoe UI', 22, 'bold'),
                     bg=COLORS['splash_bg'], fg='white')
splash_title.pack(pady=(20, 5))

splash_sub = Label(splash_content, text="Tsana Africa  •  AI for Agriculture",
                   font=('Segoe UI', 11),
                   bg=COLORS['splash_bg'], fg='#B7DCC7')
splash_sub.pack()

splash_status = Label(splash, text="Loading...",
                      font=('Segoe UI', 9),
                      bg=COLORS['splash_bg'], fg='#B7DCC7')
splash_status.pack(side="bottom", pady=20)

def run_splash():
    time.sleep(1.2)
    splash_status.config(text="Initializing AI model...")
    root.update()
    time.sleep(1.0)
    splash.destroy()
    show_welcome()

def show_welcome():
    welcome_frame.pack(fill="both", expand=True)

root.after(100, lambda: threading.Thread(target=run_splash, daemon=True).start())

# ========== WELCOME FRAME ==========
welcome_frame = Frame(root, bg=COLORS['bg'])

welcome_header = Frame(welcome_frame, bg=COLORS['primary'], height=180)
welcome_header.pack(fill="x")
welcome_header.pack_propagate(False)

logo_canvas_w = tk.Canvas(welcome_header, width=100, height=100,
                          bg=COLORS['primary'], highlightthickness=0)
logo_canvas_w.place(relx=0.5, y=25, anchor="n")
logo_canvas_w.create_oval(10, 10, 90, 90, fill=COLORS['accent'], outline="")
logo_canvas_w.create_text(50, 50, text="🌿", font=('Segoe UI', 40))

welcome_title = Label(welcome_frame, text="🌿 Welcome to Crop Disease Detector",
                      font=FONT_TITLE, bg=COLORS['bg'], fg=COLORS['primary'],
                      wraplength=700)
welcome_title.pack(pady=(30, 15))

welcome_body = Label(welcome_frame, text="", font=FONT_BODY,
                     bg=COLORS['bg'], fg=COLORS['text'],
                     justify="left", wraplength=600)
welcome_body.pack(pady=15, padx=50)

start_button = Button(welcome_frame, text="Get Started", font=('Segoe UI', 14, 'bold'),
                      command=show_main_ui, bg=COLORS['primary'], fg='white',
                      relief="flat", padx=40, pady=15, cursor="hand2", bd=0)
start_button.pack(pady=30)
add_hover(start_button, COLORS['primary'], COLORS['secondary'])

# ========== MAIN FRAME (scrollable) ==========
main_frame = Frame(root, bg=COLORS['bg'])

outer_frame = Frame(main_frame, bg=COLORS['bg'])
outer_frame.pack(fill="both", expand=True)

canvas = Canvas(outer_frame, bg=COLORS['bg'], highlightthickness=0)
scrollbar = Scrollbar(outer_frame, orient="vertical", command=canvas.yview)
scrollable_frame = Frame(canvas, bg=COLORS['bg'])

scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
canvas_window = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
canvas.bind("<Configure>", lambda e: canvas.itemconfig(canvas_window, width=e.width))
canvas.configure(yscrollcommand=scrollbar.set)

canvas.pack(side="left", fill="both", expand=True)
scrollbar.pack(side="right", fill="y")

def _on_mousewheel(event):
    canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

canvas.bind_all("<MouseWheel>", _on_mousewheel)
canvas.bind_all("<Button-4>", lambda e: canvas.yview_scroll(-1, "units"))
canvas.bind_all("<Button-5>", lambda e: canvas.yview_scroll(1, "units"))

# ---------- TOP BAR ----------
top_bar = Frame(scrollable_frame, bg=COLORS['primary'], height=100)
top_bar.pack(fill="x")
top_bar.pack_propagate(False)

logo_canvas = tk.Canvas(top_bar, width=60, height=60,
                        bg=COLORS['primary'], highlightthickness=0)
logo_canvas.place(x=30, y=20)
logo_canvas.create_oval(5, 5, 55, 55, fill=COLORS['accent'], outline="")
logo_canvas.create_text(30, 30, text="🌿", font=('Segoe UI', 22))

header_title = tk.Label(top_bar, text="🌿 Crop Disease Detector",
                        font=('Segoe UI', 20, 'bold'),
                        bg=COLORS['primary'], fg='white')
header_title.place(x=110, y=20)

header_tagline = tk.Label(top_bar, text="AI for Agriculture  •  Tsana Africa",
                          font=('Segoe UI', 10),
                          bg=COLORS['primary'], fg='#B7DCC7')
header_tagline.place(x=112, y=55)

lang_label = tk.Label(top_bar, text="Language", font=('Segoe UI', 9),
                      bg=COLORS['primary'], fg='#B7DCC7')
lang_label.place(relx=0.88, y=15, anchor="ne")

lang_var = tk.StringVar(root)
lang_var.set("English")
lang_menu = tk.OptionMenu(top_bar, lang_var, *LANGUAGES.keys(),
                          command=lambda choice: set_language(LANGUAGES[choice]))
lang_menu.config(font=('Segoe UI', 10, 'bold'), bg=COLORS['accent'], fg='white',
                 relief="flat", padx=10, pady=4, bd=0, highlightthickness=0,
                 activebackground=COLORS['accent_dark'])
lang_menu["menu"].config(bg=COLORS['card'], fg=COLORS['text'], font=('Segoe UI', 10))
lang_menu.place(relx=0.88, y=42, anchor="ne")

# ---------- UPLOAD CARD ----------
upload_card = Frame(scrollable_frame, bg=COLORS['card'],
                    highlightbackground=COLORS['border'], highlightthickness=1)
upload_card.pack(fill="x", padx=30, pady=(25, 15))

image_label = Label(upload_card, bg=COLORS['card'], fg=COLORS['text_muted'],
                    text=LANG['en']['no_image'], font=FONT_BODY,
                    width=60, height=18)
image_label.pack(pady=20)

upload_button = Button(upload_card, text=LANG['en']['upload_btn'], font=FONT_BUTTON,
                       command=upload_image, bg=COLORS['primary'], fg='white',
                       relief="flat", padx=30, pady=12, cursor="hand2", bd=0)
upload_button.pack(pady=(0, 20))
add_hover(upload_button, COLORS['primary'], COLORS['secondary'])

# ---------- RESULT CARD ----------
result_card = Frame(scrollable_frame, bg=COLORS['card'],
                    highlightbackground=COLORS['border'], highlightthickness=1)
result_card.pack(fill="both", expand=True, padx=30, pady=(0, 15))

result_title = Label(result_card, text=LANG['en']['awaiting'],
                     font=('Segoe UI', 16, 'bold'),
                     bg=COLORS['card'], fg=COLORS['primary'], anchor="w")
result_title.pack(fill="x", padx=25, pady=(20, 5))

conf_frame = Frame(result_card, bg=COLORS['card'])
conf_frame.pack(fill="x", padx=25)

confidence_label = Label(conf_frame, text="—", font=('Segoe UI', 26, 'bold'),
                         bg=COLORS['card'], fg=COLORS['text_muted'])
confidence_label.pack(side="left")

status_label = Label(conf_frame, text="", font=('Segoe UI', 11, 'italic'),
                     bg=COLORS['card'], fg=COLORS['text_muted'])
status_label.pack(side="left", padx=15)

severity_label = Label(result_card, text="", font=('Segoe UI', 11, 'bold'),
                       bg=COLORS['card'], fg=COLORS['text_muted'], anchor="w")
severity_label.pack(fill="x", padx=25, pady=(5, 0))

warning_label = Label(result_card, text="", font=('Segoe UI', 10, 'italic'),
                      bg=COLORS['card'], fg=COLORS['danger'],
                      wraplength=680, justify="left", anchor="w")

divider = Frame(result_card, bg=COLORS['border'], height=1)
divider.pack(fill="x", padx=25, pady=15)

result_body = Label(result_card, text=LANG['en']['upload_hint'], font=FONT_BODY,
                    bg=COLORS['card'], fg=COLORS['text'],
                    wraplength=680, justify="left", anchor="nw")
result_body.pack(fill="both", expand=True, padx=25, pady=(0, 15))

# ---------- ACTION BAR ----------
action_bar = Frame(scrollable_frame, bg=COLORS['bg'])
action_bar.pack(fill="x", padx=30, pady=(0, 20))

def make_action_btn(parent, text, command, color):
    b = Button(parent, text=text, font=('Segoe UI', 10, 'bold'),
               command=command, bg=color, fg='white',
               relief="flat", padx=18, pady=9, cursor="hand2", bd=0)
    add_hover(b, color, COLORS['accent_dark'])
    return b

speak_button = make_action_btn(action_bar, LANG['en']['speak'], speak_result, COLORS['secondary'])
speak_button.config(state="disabled")
speak_button.pack(side="left", padx=(0, 8))

export_button = make_action_btn(action_bar, LANG['en']['export'],
                               lambda: export_report(last_image_path, last_results) if last_results else None,
                               COLORS['secondary'])
export_button.config(state="disabled")
export_button.pack(side="left", padx=8)

history_button = make_action_btn(action_bar, LANG['en']['history'], view_history, COLORS['text_muted'])
history_button.pack(side="left", padx=8)

stats_button = make_action_btn(action_bar, LANG['en']['stats'], view_stats, COLORS['text_muted'])
stats_button.pack(side="left", padx=8)

# ---------- FOOTER ----------
footer_label = Label(scrollable_frame, text=LANG['en']['footer'], font=FONT_SMALL,
                     bg=COLORS['bg'], fg=COLORS['text_muted'])
footer_label.pack(pady=15)

# ========== INITIAL TRANSLATION ==========
translate_ui()

# ========== RUN ==========
root.mainloop()