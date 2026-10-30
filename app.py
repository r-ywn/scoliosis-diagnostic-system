import json
import os
import streamlit as st
from PIL import Image
from predict_scoliosis_clinical import ClinicalScoliosisAnalyzer

st.set_page_config(
    page_title="Scoliosis Diagnostic System",
    layout="wide"
)

st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #1E3A8A 0%, #2563EB 100%);
        color: #FFFFFF !important;
        padding: 25px;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 25px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.15);
    }
    .main-header h1 {
        color: #FFFFFF !important;
        font-weight: 700;
        font-size: 2.2rem;
        margin-bottom: 8px;
    }
    .main-header p {
        color: #E0E7FF !important;
        font-size: 1.1rem;
        margin: 0;
    }
    div[data-testid="stMetric"] {
        background-color: var(--secondary-background-color) !important;
        border: 1px solid rgba(128, 128, 128, 0.25) !important;
        border-radius: 8px;
        padding: 12px 16px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    div[data-testid="stMetric"] label, 
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: var(--text-color) !important;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div class="main-header">
        <h1>Scoliosis Diagnostic System</h1>
        <p>نظام التقييم الطبي الآلي للجنف</p>
    </div>
""", unsafe_allow_html=True)

st.sidebar.header("معلومات المشروع")
st.sidebar.markdown("**Department:** Medical Instrumentation Engineering Techniques")
st.sidebar.markdown("**Mode:** Verified Diagnostics")
st.sidebar.divider()

@st.cache_resource
def load_analyzer():
    model_path = "best_scoliosis_model_balanced.pth" if os.path.exists("best_scoliosis_model_balanced.pth") else "best_scoliosis_model.pth"
    return ClinicalScoliosisAnalyzer(model_path=model_path)

try:
    analyzer = load_analyzer()
    st.sidebar.success("النموذج جاهز للاستخدام")
except Exception as e:
    st.sidebar.error(f"تعذر تحميل النموذج: {e}")

st.sidebar.divider()
st.sidebar.subheader("رفع صورة الأشعة")
uploaded_file = st.sidebar.file_uploader("اختر صورة أشعة سينية (X-Ray)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    temp_path = "temp_xray.jpg"
    image = Image.open(uploaded_file)
    image.save(temp_path)

    try:
        col1, col2 = st.columns([1, 1.2])

        with col1:
            st.subheader("صورة الأشعة الأصلية")
            st.image(image, use_container_width=True)

        with col2:
            st.subheader("التقرير الطبي التشخيصي المباشر")
            with st.spinner("جاري تحليل صورة الأشعة عبر النموذج الذكي..."):
                result = analyzer.analyze_xray(temp_path)

            if "error" in result:
                st.error(result["error"])
            else:
                intensity = result.get("intensity", "Healthy")
                cobb = result.get("estimated_cobb_angles", {})
                max_cobb = cobb.get('max_cobb_angle', 0.0)

                if intensity == "Healthy":
                    st.success(f"**التشخيص:** [ العمود الفقري سليم — Normal Spine ] ┆ **Max Cobb Angle:** [{max_cobb}°]")
                elif intensity == "Mild":
                    st.warning(f"**التشخيص:** [ جنف خفيف — Mild Scoliosis ] ┆ **Max Cobb Angle:** [{max_cobb}°]")
                elif intensity == "Moderate":
                    st.warning(f"**التشخيص:** [ جنف متوسط — Moderate Scoliosis ] ┆ **Max Cobb Angle:** [{max_cobb}°]")
                else:
                    st.error(f"**التشخيص:** [ جنف شديد — Severe Scoliosis ] ┆ **Max Cobb Angle:** [{max_cobb}°]")

                m1, m2, m3 = st.columns(3)
                m1.metric("درجة الشدة (Severity)", result.get("intensity", "N/A"))
                m2.metric("الشكل الهندسي (Shape)", result.get("shape", "N/A"))
                m3.metric("الموقع الرئيسي (Location)", result.get("location", "N/A"))

                st.markdown("### القياسات الرقمية المحتملة لزوايا كوب (Cobb Angles - estimated AI Output):")

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Upper Thoracic", f"{cobb.get('upper_thoracic_angle', 0.0)}°")
                c2.metric("Main Thoracic", f"{cobb.get('main_thoracic_angle', 0.0)}°")
                c3.metric("Lumbar", f"{cobb.get('lumbar_angle', 0.0)}°")
                c4.metric("Max Cobb Angle", f"{cobb.get('max_cobb_angle', 0.0)}°")

                st.markdown("---")

                with st.expander("التقرير الطبي الكامل (JSON Report)", expanded=False):
                    st.json(result)

                json_str = json.dumps(result, indent=2, ensure_ascii=False)
                st.download_button(
                    label="تحميل التقرير الطبي الكامل (JSON)",
                    data=json_str,
                    file_name="scoliosis_report.json",
                    mime="application/json"
                )

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
else:
    st.info("يرجى رفع صورة أشعة سينية من القائمة الجانبية لبدء التحليل التشخيصي المباشر.")