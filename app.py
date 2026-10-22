import json
import os
import streamlit as st
from PIL import Image
from predict_scoliosis_clinical import ClinicalScoliosisAnalyzer

# ==========================================
# 1. إعدادات الصفحة الرسمية
# ==========================================
st.set_page_config(
    page_title="Scoliosis Clinical Diagnostic System",
    layout="wide"
)

# ==========================================
# 2. التنسيق الأكاديمي المتوافق مع جميع الأوضاع (Light & Dark)
# ==========================================
st.markdown("""
    <style>
    /* خلفية الهيدر الرئيسي */
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

    /* بطاقات القياسات - تتكيف تلقائياً مع الفاتح والداكن */
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

# ==========================================
# 3. الهيدر الرئيسي
# ==========================================
st.markdown("""
    <div class="main-header">
        <h1>Scoliosis Diagnostic System</h1>
        <p>نظام التقييم الطبي الآلي للجنف وتقدير زوايا كوب (Cobb Angle Estimation)</p>
    </div>
""", unsafe_allow_html=True)

# ==========================================
# 4. الشريط الجانبي (Sidebar)
# ==========================================
st.sidebar.header("معلومات المشروع")
st.sidebar.markdown("**Department:** Biomedical Engineering")
st.sidebar.markdown("**System Mode:** Clinical Inference")
st.sidebar.divider()


@st.cache_resource
def load_analyzer():
    return ClinicalScoliosisAnalyzer()


try:
    analyzer = load_analyzer()
    st.sidebar.success("النموذج المتوازن جاهز للاستخدام")
except Exception as e:
    st.sidebar.error(f"تعذر تحميل النموذج: {e}")

st.sidebar.divider()
st.sidebar.subheader("رفع صورة الأشعة")
uploaded_file = st.sidebar.file_uploader("اختر صورة أشعة (X-Ray)...", type=["jpg", "jpeg", "png"])

# ==========================================
# 5. منطقة عرض النتائج والتحليل
# ==========================================
col1, col2 = st.columns([1, 1])

if uploaded_file is not None:
    temp_path = "temp_xray.jpg"
    image = Image.open(uploaded_file)
    image.save(temp_path)

    with col1:
        st.subheader("صورة الأشعة المرفوعة")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("التقرير الطبي التشخيصي")
        with st.spinner("جاري تحليل صورة الأشعة وحساب زوايا كوب..."):
            result = analyzer.analyze_xray(temp_path)

        if "error" in result:
            st.error(result["error"])
        else:
            intensity = result["intensity"]
            if intensity == "Healthy":
                st.success(f"**الحالة:** {result['stat']}")
            elif intensity in ["Mild", "Moderate"]:
                st.warning(f"**الحالة:** {result['stat']}")
            else:
                st.error(f"**الحالة:** {result['stat']}")

            # عرض المؤشرات الطبية
            m1, m2, m3 = st.columns(3)
            m1.metric("درجة الشدة (Intensity)", result["intensity"])
            m2.metric("شكل الانحناء (Shape)", result["shape"])
            m3.metric("موقع الانحناء (Location)", result["location"])

            st.markdown("### القياسات الرقمية لزوايا كوب (Cobb Angles):")
            cobb = result["estimated_cobb_angles"]

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Upper Thoracic", f"{cobb['upper_thoracic_angle']}°")
            c2.metric("Main Thoracic", f"{cobb['main_thoracic_angle']}°")
            c3.metric("Lumbar", f"{cobb['lumbar_angle']}°")
            c4.metric("Max Cobb", f"{cobb['max_cobb_angle']}°")

            st.markdown("---")
            st.markdown("### التقرير الطبي الخماسي المنسق (JSON Report):")
            st.json(result)

            json_str = json.dumps(result, indent=2, ensure_ascii=False)
            st.download_button(
                label="تحميل التقرير الطبي (JSON)",
                data=json_str,
                file_name="scoliosis_report.json",
                mime="application/json"
            )

    if os.path.exists(temp_path):
        os.remove(temp_path)
else:
    st.info("يرجى رفع صورة أشعة من القائمة الجانبية لبدء التحليل السريري.")