import json
import os
import streamlit as st
from PIL import Image
from predict_scoliosis_clinical import ClinicalScoliosisAnalyzer

st.set_page_config(page_title="Scoliosis Diagnostic System", layout="wide")

st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #1E3A8A 0%, #2563EB 100%);
        color: #FFFFFF !important;
        padding: 25px;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 25px;
    }
    .main-header h1 { color: #FFFFFF !important; font-weight: 700; margin-bottom: 8px; }
    .main-header p { color: #E0E7FF !important; margin: 0; }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div class="main-header">
        <h1>Scoliosis Diagnostic System</h1>
        <p>نظام التقييم الطبي الآلي وخرائط التركيز Grad-CAM++ والتحليل الهندسي للمنحنيات</p>
    </div>
""", unsafe_allow_html=True)

st.sidebar.header("معلومات المشروع")
st.sidebar.markdown("**Department:** Biomedical Engineering")
st.sidebar.markdown("**Mode:** Verified Clinical Diagnostics")

@st.cache_resource
def load_analyzer():
    return ClinicalScoliosisAnalyzer()

try:
    analyzer = load_analyzer()
    st.sidebar.success("النموذج ومحرك Grad-CAM++ جاهزان")
except Exception as e:
    st.sidebar.error(f"تعذر تحميل النموذج: {e}")

st.sidebar.divider()
uploaded_file = st.sidebar.file_uploader("اختر صورة أشعة سينية (X-Ray)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    temp_path = "temp_xray.jpg"
    image = Image.open(uploaded_file)
    image.save(temp_path)

    with st.spinner("جاري تحليل صورة الأشعة واستخراج خرائط التركيز والقياسات الهندسة..."):
        result = analyzer.analyze_xray(temp_path)

    col1, col2, col3 = st.columns([1, 1, 1.1])

    with col1:
        st.subheader("1. صورة الأشعة الأصلية")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("2. خريطة التركيز (Grad-CAM++)")
        if "gradcam_path" in result and os.path.exists(result["gradcam_path"]):
            st.image(result["gradcam_path"], use_container_width=True)
        else:
            st.info("غير متاح")

    with col3:
        st.subheader("3. التقرير الطبي والتحليل الهندسي")
        if "error" in result:
            st.error(result["error"])
        else:
            intensity = result["intensity"]
            if intensity == "Healthy":
                st.success(f"**التشخيص:** {result['stat']}")
            elif intensity in ["Mild", "Moderate"]:
                st.warning(f"**التشخيص:** {result['stat']}")
            else:
                st.error(f"**التشخيص:** {result['stat']}")

            m1, m2, m3 = st.columns(3)
            m1.metric("الشدة", result["intensity"])
            m2.metric("الشكل الهندسي", result["shape"])
            m3.metric("الموقع الرئيسي", result["location"])

            st.markdown("### قياسات زوايا كوب (Cobb Angles):")
            cobb = result["estimated_cobb_angles"]
            c1, c2 = st.columns(2)
            c1.write(f"- Upper Thoracic: **{cobb['upper_thoracic_angle']}°**")
            c1.write(f"- Main Thoracic: **{cobb['main_thoracic_angle']}°**")
            c2.write(f"- Lumbar: **{cobb['lumbar_angle']}°**")
            c2.write(f"- Max Cobb Angle: **{cobb['max_cobb_angle']}°**")

            st.markdown("---")
            st.json(result)

    # تنظيف الملفات المؤقتة
    if os.path.exists(temp_path):
        os.remove(temp_path)
    if "gradcam_path" in result and os.path.exists(result["gradcam_path"]):
        os.remove(result["gradcam_path"])
else:
    st.info("يرجى رفع صورة أشعة سينية من القائمة الجانبية لبدء التحليل التشخيصي الشامل.")