import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import time
import datetime

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as architecture_preprocess_input

try:
    from influxdb_client import InfluxDBClient, Point
    from influxdb_client.client.write_api import SYNCHRONOUS
    INFLUXDB_AVAILABLE = True
except ImportError:
    INFLUXDB_AVAILABLE = False


# KONFIGURASI DASAR

st.set_page_config(
    page_title="Pilah Pilih — Klasifikasi Sampah AI",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

MODEL_PATH = "best_garbage_model.keras"
IMG_SIZE = (224, 224)

CLASS_LABELS = [
    "Botol_Plastik",
    "Kaca",
    "Karton_Minuman",
    "Kemasan_Logam",
    "Kertas_Karton",
    "Sampah_Rumah_Tangga",
]

CLASS_INFO = {
    "Botol_Plastik": {
        "emoji": "🧴",
        "nama": "Botol Plastik",
        "kategori": "Anorganik",
        "tips": "Bilas sisa cairan, tekan botol agar pipih, lepas tutupnya. Daur ulang di tempat sampah plastik.",
        "dampak": "Butuh 450+ tahun untuk terurai secara alami di alam.",
    },
    "Kaca": {
        "emoji": "🍾",
        "nama": "Kaca",
        "kategori": "Anorganik",
        "tips": "Bungkus pecahan kaca dengan koran sebelum dibuang agar tidak melukai petugas kebersihan.",
        "dampak": "Bisa didaur ulang berulang kali tanpa kehilangan kualitas.",
    },
    "Karton_Minuman": {
        "emoji": "🧃",
        "nama": "Karton Minuman (Tetra Pak)",
        "kategori": "Anorganik",
        "tips": "Bilas dan keringkan dahulu, lalu pipihkan sebelum dibuang ke tempat sampah karton.",
        "dampak": "Terdiri dari campuran kertas, plastik, dan alumunium — butuh proses daur ulang khusus.",
    },
    "Kemasan_Logam": {
        "emoji": "🥫",
        "nama": "Kemasan Logam",
        "kategori": "Anorganik",
        "tips": "Bilas sisa makanan/minuman, tidak perlu melepas label. Daur ulang di tempat sampah logam.",
        "dampak": "Alumunium bisa didaur ulang berkali-kali dan hemat 95% energi dibanding produksi baru.",
    },
    "Kertas_Karton": {
        "emoji": "📦",
        "nama": "Kertas & Karton",
        "kategori": "Anorganik",
        "tips": "Pastikan kering dan tidak terkontaminasi minyak/makanan. Lipat agar hemat tempat.",
        "dampak": "Satu ton kertas daur ulang menyelamatkan sekitar 17 pohon.",
    },
    "Sampah_Rumah_Tangga": {
        "emoji": "🗑️",
        "nama": "Sampah Rumah Tangga",
        "kategori": "Organik / Residu",
        "tips": "Pisahkan sisa makanan untuk kompos bila memungkinkan. Sisanya masuk kategori residu.",
        "dampak": "Sampah organik yang tercampur residu menghasilkan gas metana di TPA.",
    },
}

MODEL_METRICS = {
    "Baseline CNN": 0.7502,
    "ResNet50 (sebelum tuning)": 0.9715,
    "MobileNetV2 (sebelum tuning)": 0.9763,
    "EfficientNetB0 (sebelum tuning)": 0.9763,
    "MobileNetV2 (final, setelah fine-tuning)": 0.9839,
}

DATASET_DISTRIBUTION = {
    "Botol_Plastik": 1437,
    "Kaca": 1225,
    "Kemasan_Logam": 1217,
    "Sampah_Rumah_Tangga": 1050,
    "Karton_Minuman": 1047,
    "Kertas_Karton": 1042,
}


# CUSTOM CSS — tema biru & kuning senada dengan referensi

st.markdown(
    """
    <style>
    #MainMenu, footer, header {visibility: hidden;}

    :root {
        --pp-blue: #2563EB;
        --pp-blue-dark: #1D4ED8;
        --pp-yellow: #FBBF24;
        --pp-yellow-light: #FEF3C7;
        --pp-green: #16A34A;
        --pp-bg: #F8FAFC;
        --pp-text: #1E293B;
        --pp-muted: #64748B;
    }

    .stApp {
        background:
            radial-gradient(circle at 8% 15%, rgba(37,99,235,0.22) 0%, transparent 45%),
            radial-gradient(circle at 92% 10%, rgba(251,191,36,0.30) 0%, transparent 45%),
            radial-gradient(circle at 88% 78%, rgba(22,163,74,0.22) 0%, transparent 48%),
            radial-gradient(circle at 12% 85%, rgba(37,99,235,0.20) 0%, transparent 45%),
            radial-gradient(circle at 50% 50%, rgba(251,191,36,0.12) 0%, transparent 65%),
            var(--pp-bg);
        background-attachment: fixed;
    }

    /* Elemen dekoratif bertema sampah/lingkungan, mengambang pelan di belakang konten */
    .pp-floaty {
        position: fixed; z-index: 0; opacity: 0.35; font-size: 4.5rem;
        animation: pp-float 6s ease-in-out infinite;
        pointer-events: none;
        filter: drop-shadow(0 2px 4px rgba(0,0,0,0.05));
    }
    @keyframes pp-float {
        0%, 100% { transform: translateY(0px) rotate(0deg); }
        50% { transform: translateY(-22px) rotate(8deg); }
    }
    .pp-floaty.f1 { top: 8%;  left: 3%;  animation-delay: 0s; }
    .pp-floaty.f2 { top: 18%; right: 4%; animation-delay: 1.2s; font-size: 3.8rem; }
    .pp-floaty.f3 { top: 55%; left: 1%;  animation-delay: 2.4s; font-size: 4rem; }
    .pp-floaty.f4 { top: 68%; right: 3%; animation-delay: 0.8s; }
    .pp-floaty.f5 { top: 87%; left: 8%;  animation-delay: 1.8s; font-size: 3.5rem; }
    .pp-floaty.f6 { top: 38%; right: 1%; animation-delay: 3s;   font-size: 3.2rem; }

    /* Konten utama tetap di atas elemen dekoratif */
    .block-container { position: relative; z-index: 1; }

    /* Kartu jadi sedikit interaktif saat disentuh mouse */
    .pp-card, .pp-class-card, .pp-stat-card {
        transition: transform 0.18s ease, box-shadow 0.18s ease;
    }
    .pp-card:hover, .pp-class-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 24px rgba(37,99,235,0.14);
    }
    .pp-stat-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 18px rgba(0,0,0,0.08);
    }

    .block-container { padding-top: 2rem; max-width: 1100px; }

    /* ---- Top banner ---- */
    .pp-banner {
        background: linear-gradient(90deg, var(--pp-yellow) 0%, #FCD34D 100%);
        padding: 10px 24px; border-radius: 12px; text-align: center;
        font-weight: 600; color: #1E293B; margin-bottom: 1.5rem; font-size: 0.95rem;
    }

    /* ---- Hero ---- */
    .pp-badge {
        display: inline-block; background: #DBEAFE; color: var(--pp-blue);
        padding: 6px 14px; border-radius: 999px; font-size: 0.85rem; font-weight: 600;
        margin-bottom: 1rem;
    }
    .pp-hero-title { font-size: 2.6rem; font-weight: 800; line-height: 1.15; color: var(--pp-text); }
    .pp-hero-title .accent-blue { color: var(--pp-blue); }
    .pp-hero-title .accent-yellow { color: var(--pp-yellow); }
    .pp-hero-subtitle { font-size: 1.05rem; color: var(--pp-muted); margin: 1rem 0 1.5rem 0; line-height: 1.6; }

    /* ---- Cards ---- */
    .pp-card {
        background: white; border-radius: 16px; padding: 1.25rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06); border: 1px solid #EEF2F7;
        height: 100%;
    }
    .pp-card h4 { margin: 0.5rem 0 0.25rem 0; font-size: 1rem; color: var(--pp-text); }
    .pp-card p { margin: 0; font-size: 0.85rem; color: var(--pp-muted); }
    .pp-icon-circle {
        width: 42px; height: 42px; border-radius: 12px; display: flex;
        align-items: center; justify-content: center; font-size: 1.3rem;
        background: var(--pp-yellow-light);
    }

    .pp-stat-card {
        background: white; border-radius: 16px; padding: 1rem 1.25rem;
        border: 1px solid #EEF2F7; display: flex; align-items: center; gap: 12px;
    }
    .pp-stat-number { font-size: 1.4rem; font-weight: 800; color: var(--pp-text); }
    .pp-stat-label { font-size: 0.8rem; color: var(--pp-muted); }

    .pp-section-title {
        font-size: 1.6rem; font-weight: 800; color: var(--pp-text);
        text-align: center; margin: 2.5rem 0 0.3rem 0;
    }
    .pp-section-underline {
        width: 50px; height: 4px; background: var(--pp-yellow); margin: 0 auto 1.8rem auto;
        border-radius: 2px;
    }

    .pp-class-card {
        background: white; border-radius: 16px; padding: 1.4rem; border: 1px solid #EEF2F7;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .pp-class-emoji { font-size: 2.2rem; }
    .pp-class-name { font-weight: 700; font-size: 1.05rem; margin: 0.4rem 0 0.1rem 0; color: var(--pp-text); }
    .pp-class-kategori {
        display: inline-block; font-size: 0.72rem; font-weight: 600; padding: 2px 10px;
        border-radius: 999px; background: #DBEAFE; color: var(--pp-blue); margin-bottom: 0.6rem;
    }
    .pp-class-label { font-weight: 600; font-size: 0.8rem; color: var(--pp-text); margin-top: 0.5rem;}
    .pp-class-text { font-size: 0.82rem; color: var(--pp-muted); line-height: 1.5; }

    /* Nav buttons */
    div[data-testid="stHorizontalBlock"] button {
        border-radius: 999px !important;
    }

    /* Primary CTA button styling for st.button (klasifikasi/nav) */
    .stButton>button {
        border-radius: 10px;
        font-weight: 600;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .stButton>button[kind="primary"] {
        background-color: var(--pp-blue) !important;
        color: white !important;
        border: none !important;
    }
    /* Tombol nav non-aktif & CTA sekunder — dipaksa terang, TIDAK ikut tema dark/light sistem */
    .stButton>button[kind="secondary"] {
        background-color: white !important;
        color: var(--pp-text) !important;
        border: 1px solid #E2E8F0 !important;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 14px rgba(37,99,235,0.18);
    }

    .pp-footer { text-align: center; color: var(--pp-muted); font-size: 0.8rem; margin-top: 3rem; padding: 1.5rem 0; }

    /* ---- Upload dropzone (Klasifikasi page) ---- */
    [data-testid="stFileUploaderDropzone"] {
        background: linear-gradient(135deg, #FEF9E7 0%, #EFF6FF 100%) !important;
        border: 2px dashed var(--pp-blue) !important;
        border-radius: 16px !important;
        padding: 1.5rem !important;
    }
    [data-testid="stFileUploaderDropzone"] button {
        background: var(--pp-blue) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
    }
    [data-testid="stFileUploaderDropzone"] svg { color: var(--pp-blue) !important; }
    [data-testid="stFileUploaderDropzoneInstructions"] span,
    [data-testid="stFileUploaderDropzoneInstructions"] small {
        color: var(--pp-text) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Elemen dekoratif mengambang bertema sampah/lingkungan — tampil di semua halaman
st.markdown(
    """
    <div class="pp-floaty f1">♻️</div>
    <div class="pp-floaty f2">🍃</div>
    <div class="pp-floaty f3">🌱</div>
    <div class="pp-floaty f4">🗑️</div>
    <div class="pp-floaty f5">🍾</div>
    <div class="pp-floaty f6">🧃</div>
    """,
    unsafe_allow_html=True,
)


# STATE & NAVIGASI

if "page" not in st.session_state:
    st.session_state.page = "Beranda"
if "session_count" not in st.session_state:
    st.session_state.session_count = 0
if "session_conf_sum" not in st.session_state:
    st.session_state.session_conf_sum = 0.0


def goto(page_name):
    st.session_state.page = page_name


PAGES = ["Beranda", "Klasifikasi", "Panduan Sampah", "Statistik", "Tentang"]

st.markdown(
    '<div class="pp-banner">♻️ Pilah dengan benar, bumi makin bersih 🌱 — '
    'Project Final Machine Learning: Klasifikasi Sampah dengan Computer Vision</div>',
    unsafe_allow_html=True,
)

nav_cols = st.columns(len(PAGES) + 1)
with nav_cols[0]:
    st.markdown("### ♻️ **Pilah·Pilih**")
for i, page_name in enumerate(PAGES):
    with nav_cols[i + 1]:
        is_active = st.session_state.page == page_name
        if st.button(page_name, key=f"nav_{page_name}",
                      type="primary" if is_active else "secondary",
                      width="stretch"):
            goto(page_name)
            st.rerun()

st.write("")



# MODEL LOADING (dipakai di halaman Klasifikasi)

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={"preprocess_input": architecture_preprocess_input},
        safe_mode=False,
    )


def predict_image(pil_image, model):
    img_resized = pil_image.resize(IMG_SIZE)
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    pred_probs = model.predict(img_array, verbose=0)[0]
    return pred_probs



# MONITORING — kirim metrik ke InfluxDB Cloud (untuk dashboard Grafana)

@st.cache_resource
def get_influx_client():
    """Buat koneksi ke InfluxDB Cloud. Return None kalau library/kredensial tidak tersedia
    (app tetap jalan normal tanpa monitoring)."""
    if not INFLUXDB_AVAILABLE:
        return None
    try:
        required = {"INFLUXDB_URL", "INFLUXDB_TOKEN", "INFLUXDB_ORG", "INFLUXDB_BUCKET"}
        if not required.issubset(st.secrets.keys()):
            return None
        client = InfluxDBClient(
            url=st.secrets["INFLUXDB_URL"],
            token=st.secrets["INFLUXDB_TOKEN"],
            org=st.secrets["INFLUXDB_ORG"],
        )
        return client
    except Exception:
        # Kalau secrets belum dikonfigurasi sama sekali (misal saat run lokal tanpa secrets.toml),
        # jangan sampai app crash -- monitoring cuma fitur tambahan, bukan fitur inti.
        return None


def log_prediction_metric(pred_label, confidence):
    """Kirim satu titik data (1 prediksi) ke InfluxDB Cloud. Gagal diam-diam kalau
    monitoring belum di-setup, supaya tidak mengganggu fitur klasifikasi utama."""
    client = get_influx_client()
    if client is None:
        return
    try:
        write_api = client.write_api(write_options=SYNCHRONOUS)
        point = (
            Point("klasifikasi_sampah")
            .tag("kelas", pred_label)
            .field("confidence", float(confidence))
            .field("count", 1)
            .time(datetime.datetime.utcnow())
        )
        write_api.write(bucket=st.secrets["INFLUXDB_BUCKET"], record=point)
    except Exception:
        pass  # monitoring tidak boleh menghentikan fitur utama kalau ada masalah koneksi



# HALAMAN: BERANDA

def render_beranda():
    left, right = st.columns([1.1, 1])

    with left:
        st.markdown('<div class="pp-badge">✨ AI Untuk Lingkungan Bersih</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="pp-hero-title">'
            '<span class="accent-blue">Pilah</span> Sampah,<br>'
            '<span class="accent-yellow">Pilih</span> Masa Depan</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<p class="pp-hero-subtitle">Gunakan AI untuk mengklasifikasikan sampahmu ke dalam '
            '6 kategori secara instan. Pilah dengan benar, kurangi pencemaran, jaga bumi kita bersama.</p>',
            unsafe_allow_html=True,
        )
        c1, c2 = st.columns(2)
        with c1:
            if st.button("📤  Klasifikasi Sekarang", type="primary", width="stretch"):
                goto("Klasifikasi")
                st.rerun()
        with c2:
            if st.button("📖  Lihat Panduan", width="stretch"):
                goto("Panduan Sampah")
                st.rerun()

        st.write("")
        f1, f2, f3 = st.columns(3)
        feature_cards = [
            ("📷", "Deteksi Cepat", "AI memproses gambar dalam hitungan detik"),
            ("🛡️", "Akurasi Tinggi", "Model terlatih untuk klasifikasi akurat"),
            ("🌿", "Ramah Lingkungan", "Setiap pilahmu berdampak untuk bumi"),
        ]
        for col, (icon, title, desc) in zip([f1, f2, f3], feature_cards):
            with col:
                st.markdown(
                    f'<div class="pp-card"><div class="pp-icon-circle">{icon}</div>'
                    f'<h4>{title}</h4><p>{desc}</p></div>',
                    unsafe_allow_html=True,
                )

    with right:
        st.markdown(
            """
            <div style="background: linear-gradient(135deg, #FEF3C7 0%, #DBEAFE 100%);
                        border-radius: 24px; padding: 2.5rem; text-align:center;">
                <div style="font-size: 5rem;">🗑️♻️</div>
                <p style="color:#1E293B; font-weight:700; font-size:1.1rem; margin-top:0.5rem;">
                    6 Kategori Sampah Dikenali AI
                </p>
                <div style="display:flex; flex-wrap:wrap; gap:8px; justify-content:center; margin-top:1rem;">
            """
            + "".join(
                f'<span style="background:white; color:#1E293B; padding:6px 12px; '
                f'border-radius:999px; font-size:0.8rem; font-weight:600; '
                f'box-shadow:0 1px 2px rgba(0,0,0,0.08);">'
                f'{info["emoji"]} {info["nama"]}</span>'
                for info in CLASS_INFO.values()
            )
            + "</div></div>",
            unsafe_allow_html=True,
        )

    st.write("")
    st.write("")
    st.markdown('<div class="pp-section-title">Cara Kerja</div>', unsafe_allow_html=True)
    st.markdown('<div class="pp-section-underline"></div>', unsafe_allow_html=True)
    hc1, hc2, hc3 = st.columns(3)
    how_it_works = [
        ("1️⃣", "Upload Foto", "Ambil atau upload foto sampah yang mau kamu identifikasi"),
        ("2️⃣", "AI Menganalisis", "Model AI memproses gambar dan mengenali pola visualnya dalam hitungan detik"),
        ("3️⃣", "Dapatkan Hasil", "Lihat kategori sampah beserta confidence score dan tips cara membuangnya"),
    ]
    for col, (num, title, desc) in zip([hc1, hc2, hc3], how_it_works):
        with col:
            st.markdown(
                f'<div class="pp-card" style="text-align:center;">'
                f'<div style="font-size:1.8rem;">{num}</div>'
                f'<h4>{title}</h4><p>{desc}</p></div>',
                unsafe_allow_html=True,
            )

    st.write("")
    st.write("")
    box1, box2 = st.columns([1.3, 1])
    with box1:
        st.markdown(
            """
            <div class="pp-card" style="background:#FEF9E7;">
                <h4 style="margin-top:0;">🌍 Mengapa Pilah Sampah Itu Penting?</h4>
                <p>🔻 Mengurangi pencemaran lingkungan</p>
                <p>♻️ Mendukung daur ulang dan ekonomi sirkular</p>
                <p>🌱 Menjaga bumi untuk generasi mendatang</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with box2:
        avg_conf = (
            (st.session_state.session_conf_sum / st.session_state.session_count)
            if st.session_state.session_count > 0 else None
        )
        st.markdown(
            f"""
            <div class="pp-card">
                <h4 style="margin-top:0;">📊 Statistik Sesi Ini</h4>
                <div class="pp-stat-card" style="margin-bottom:8px;">
                    <div>🖼️</div>
                    <div><div class="pp-stat-number">{st.session_state.session_count}</div>
                    <div class="pp-stat-label">Gambar diklasifikasi</div></div>
                </div>
                <div class="pp-stat-card">
                    <div>🎯</div>
                    <div><div class="pp-stat-number">{f'{avg_conf:.0f}%' if avg_conf else '—'}</div>
                    <div class="pp-stat-label">Rata-rata confidence</div></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption("Statistik ini dihitung dari sesi kamu saat ini, akan reset kalau halaman di-refresh.")



# HALAMAN: KLASIFIKASI

def render_klasifikasi():
    st.markdown('<div class="pp-section-title">Klasifikasi Sampah</div>', unsafe_allow_html=True)
    st.markdown('<div class="pp-section-underline"></div>', unsafe_allow_html=True)
    st.write("Upload foto sampah, dan model AI akan memprediksi kategorinya dari 6 kelas yang dikenali.")

    model = load_model()
    if model is None:
        st.error(
            f"⚠️ File model `{MODEL_PATH}` tidak ditemukan di folder aplikasi. "
            "Pastikan file model sudah ditambahkan ke repository."
        )
        return

    st.markdown(
        '<p style="font-weight:600; color:#1E293B; margin-bottom:0.4rem;">📤 Upload Foto Sampah</p>',
        unsafe_allow_html=True,
    )
    uploaded_file = st.file_uploader(
        "Seret & lepas gambar di sini, atau klik untuk pilih file (JPG/JPEG/PNG)",
        type=["jpg", "jpeg", "png"],
        label_visibility="visible",
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        col1, col2 = st.columns([1, 1])

        with col1:
            st.image(image, caption="Gambar yang diupload", width="stretch")

        with st.spinner("Menganalisis gambar..."):
            time.sleep(0.3)
            pred_probs = predict_image(image, model)

        pred_idx = int(np.argmax(pred_probs))
        pred_label = CLASS_LABELS[pred_idx]
        confidence = float(pred_probs[pred_idx]) * 100
        info = CLASS_INFO[pred_label]

        # update session stats sekali per gambar baru
        last_file_id = st.session_state.get("last_file_id")
        if last_file_id != uploaded_file.name:
            st.session_state.session_count += 1
            st.session_state.session_conf_sum += confidence
            st.session_state.last_file_id = uploaded_file.name
            log_prediction_metric(pred_label, confidence)  # kirim ke InfluxDB untuk dashboard Grafana

        with col2:
            st.markdown(
                f"""
                <div class="pp-card">
                    <div class="pp-class-emoji">{info['emoji']}</div>
                    <div class="pp-class-name">{info['nama']}</div>
                    <div class="pp-class-kategori">{info['kategori']}</div>
                    <div class="pp-class-label">Confidence</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.progress(min(max(confidence / 100, 0.0), 1.0))
            st.write(f"**{confidence:.1f}%**")
            if confidence < 60:
                st.warning("⚠️ Model kurang yakin. Coba foto dengan pencahayaan lebih baik atau objek lebih jelas.")
            st.info(f"💡 **Tips buang:** {info['tips']}")

        st.divider()
        st.subheader("Detail Prediksi — Semua Kelas")
        sorted_idx = np.argsort(pred_probs)[::-1]
        for idx in sorted_idx:
            label = CLASS_LABELS[idx]
            prob = float(pred_probs[idx]) * 100
            label_info = CLASS_INFO[label]
            st.write(f"{label_info['emoji']} **{label_info['nama']}** — {prob:.2f}%")
            st.progress(min(max(prob / 100, 0.0), 1.0))
    else:
        st.info("👆 Silakan upload gambar untuk memulai prediksi.")



# HALAMAN: PANDUAN SAMPAH

def render_panduan():
    st.markdown('<div class="pp-section-title">Panduan Jenis Sampah</div>', unsafe_allow_html=True)
    st.markdown('<div class="pp-section-underline"></div>', unsafe_allow_html=True)
    st.write("Kenali 6 kategori sampah yang bisa dikenali oleh model AI, beserta cara membuang yang benar.")
    st.write("")

    items = list(CLASS_INFO.items())
    for row_start in range(0, len(items), 3):
        cols = st.columns(3)
        for col, (label, info) in zip(cols, items[row_start:row_start + 3]):
            with col:
                st.markdown(
                    f"""
                    <div class="pp-class-card">
                        <div class="pp-class-emoji">{info['emoji']}</div>
                        <div class="pp-class-name">{info['nama']}</div>
                        <div class="pp-class-kategori">{info['kategori']}</div>
                        <div class="pp-class-label">♻️ Cara membuang</div>
                        <div class="pp-class-text">{info['tips']}</div>
                        <div class="pp-class-label">🌍 Fakta lingkungan</div>
                        <div class="pp-class-text">{info['dampak']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        st.write("")



# HALAMAN: STATISTIK

def render_statistik():
    st.markdown('<div class="pp-section-title">Statistik Model</div>', unsafe_allow_html=True)
    st.markdown('<div class="pp-section-underline"></div>', unsafe_allow_html=True)

    s1, s2, s3, s4 = st.columns(4)
    stats = [
        ("🖼️", "7.018", "Total gambar dataset"),
        ("🏷️", "6", "Kelas sampah"),
        ("🎯", "98.39%", "Best validation accuracy"),
        ("🧠", "MobileNetV2", "Arsitektur model final"),
    ]
    for col, (icon, val, label) in zip([s1, s2, s3, s4], stats):
        with col:
            st.markdown(
                f'<div class="pp-stat-card">{icon}<div>'
                f'<div class="pp-stat-number">{val}</div>'
                f'<div class="pp-stat-label">{label}</div></div></div>',
                unsafe_allow_html=True,
            )

    st.write("")
    st.write("")
    left, right = st.columns(2)

    with left:
        st.markdown("#### 📈 Perbandingan Akurasi Model")
        st.bar_chart(MODEL_METRICS)
        st.caption(
            "Baseline CNN dilatih dari nol, sisanya menggunakan transfer learning dari model ImageNet. "
            "MobileNetV2, ResNet50, dan EfficientNetB0 sempat seri/berdekatan di percobaan awal (±97%), "
            "tapi MobileNetV2 hasil fine-tuning (lr=1e-4) tampil sebagai model final karena val accuracy "
            "tertinggi (98.39%) dengan gap train-validation paling kecil (0.31%)."
        )

    with right:
        st.markdown("#### 🗂️ Distribusi Data per Kelas")
        st.bar_chart(DATASET_DISTRIBUTION)
        st.caption(
            "Jumlah gambar training per kelas relatif seimbang (rasio kelas terbanyak vs tersedikit ±1.4x), "
            "sehingga tidak memerlukan penanganan class imbalance yang kompleks."
        )

    st.write("")
    st.markdown(
        """
        <div class="pp-card">
        <h4 style="margin-top:0;">📝 Catatan Evaluasi</h4>
        <p>Model final (MobileNetV2, fine-tuned) diuji pada test set (15% dari data, ~1.053 gambar)
        dengan hasil akurasi keseluruhan 98% (weighted F1-score 0.98). Kesalahan yang paling sering
        terjadi adalah kelas <b>Botol Plastik</b> yang terprediksi sebagai <b>Kaca</b> — wajar karena
        keduanya sering sama-sama transparan/mengkilap secara visual.</p>
        <p>Sebagai sanity-check tambahan, model juga diuji pada folder <i>test</i> bawaan dataset asli
        (157 gambar, timpang antar kelas) dan menghasilkan akurasi 70.7% — kesenjangan ini kemungkinan
        disebabkan oleh ukuran sampel yang kecil dan perbedaan kondisi pengambilan gambar pada sumber
        data tersebut, bukan indikasi model yang buruk.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )



# HALAMAN: TENTANG

def render_tentang():
    st.markdown('<div class="pp-section-title">Tentang Project</div>', unsafe_allow_html=True)
    st.markdown('<div class="pp-section-underline"></div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="pp-card">
        <h4 style="margin-top:0;">🎯 Latar Belakang</h4>
        <p>Pengelolaan sampah di Indonesia masih banyak dilakukan secara manual, yang berdampak pada
        lambatnya proses daur ulang dan rendahnya akurasi pemilahan. Project ini membangun sistem
        klasifikasi sampah otomatis berbasis Computer Vision untuk mempercepat proses sortir di
        fasilitas daur ulang atau smart bin.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            """
            <div class="pp-card">
            <h4 style="margin-top:0;">📚 Dataset</h4>
            <p><b>Sumber:</b> Kaggle — manonstr/tipe-webscraping</p>
            <p><b>Total gambar:</b> 7.018 (folder training)</p>
            <p><b>Jumlah kelas:</b> 6 kategori sampah</p>
            <p><b>Metode akuisisi:</b> Web scraping</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
            <div class="pp-card">
            <h4 style="margin-top:0;">🛠️ Teknologi</h4>
            <p><b>Model:</b> MobileNetV2 (transfer learning + fine-tuning)</p>
            <p><b>Framework:</b> TensorFlow / Keras</p>
            <p><b>Deployment:</b> Streamlit Community Cloud</p>
            <p><b>Monitoring:</b> InfluxDB Cloud + Grafana Cloud</p>
            <p><b>Pipeline:</b> ETL → EDA → Preprocessing → Modeling → Evaluasi → Deployment</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")
    st.markdown(
        """
        <div class="pp-card">
        <h4 style="margin-top:0;">⚠️ Catatan</h4>
        <p>Aplikasi ini dibuat sebagai bagian dari Final Project mata kuliah Machine Learning
        dan bersifat prototipe edukatif — belum divalidasi untuk penggunaan produksi skala besar.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )



# ROUTER

if st.session_state.page == "Beranda":
    render_beranda()
elif st.session_state.page == "Klasifikasi":
    render_klasifikasi()
elif st.session_state.page == "Panduan Sampah":
    render_panduan()
elif st.session_state.page == "Statistik":
    render_statistik()
elif st.session_state.page == "Tentang":
    render_tentang()

st.markdown(
    '<div class="pp-footer">♻️ Pilah·Pilih — Final Project Machine Learning: Klasifikasi Sampah dengan Computer Vision</div>',
    unsafe_allow_html=True,
)
