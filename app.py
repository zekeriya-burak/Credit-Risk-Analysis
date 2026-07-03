import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. SAYFA AYARLARI VE VERİ/MODEL YÜKLEME ---
st.set_page_config(page_title="Kredi Riski Analiz Merkezi", layout="wide")

if 'aktif_sayfa' not in st.session_state:
    st.session_state.aktif_sayfa = 'ana_menu'

def sayfa_degistir(yeni_sayfa):
    st.session_state.aktif_sayfa = yeni_sayfa

@st.cache_data
def load_data():
    df = pd.read_csv('credit_risk_dataset.csv')
    df = df.drop_duplicates()
    
    df = df[df['person_age'] <= 100]
    df = df[df['person_emp_length'] <= 70]
    df = df[(df['loan_int_rate'] >= 0) & (df['loan_int_rate'] <= 100)]
    
    df = df.dropna()
    return df

@st.cache_resource
def load_ml_elements():
    try:
        model = joblib.load('rf_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except:
        return None, None

df = load_data()
model, scaler = load_ml_elements()

# ==========================================
# 1. EKRAN: ANA MENÜ (YENİ MODERN TASARIM)
# ==========================================
if st.session_state.aktif_sayfa == 'ana_menu':
    
    # Ana sayfaya özel buton büyütme ve güzelleştirme CSS'i
    st.markdown("""
        <style>
        div.stButton > button {
            height: 80px;
            font-size: 22px !important;
            font-weight: 600 !important;
            border-radius: 15px;
            border: 2px solid #4CAF50;
            transition: all 0.3s ease;
        }
        div.stButton > button:hover {
            border-color: #2E86C1;
            color: #2E86C1;
            transform: scale(1.02);
        }
        </style>
    """, unsafe_allow_html=True)

    # Ekranı 3'e bölüp ortadaki sütunu kullanıyoruz (Tasarımı ortalamak için)
    bos1, orta, bos2 = st.columns([1, 2, 1])
    
    with orta:
        # Üstten biraz boşluk bırakalım
        st.write("<br><br><br>", unsafe_allow_html=True)
        
        # Modern, büyük bir başlık
        st.markdown("<h1 style='text-align: center; font-size: 3.5em;'>🏦 Kredi Riski Analiz Sistemi</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #a0a0a0; font-size: 1.2em;'>Yapay Zeka Destekli Finansal Karar Motoru</p>", unsafe_allow_html=True)
        
        # Butonlarla başlık arasına şık bir boşluk
        st.write("<br><br>", unsafe_allow_html=True)
        
        # Kocaman Ana Butonlar
        st.button("🔮 Kredi Risk Tahmini", on_click=sayfa_degistir, args=('tahmin',), use_container_width=True)
        st.write("<br>", unsafe_allow_html=True) # İki buton arası boşluk
        st.button("📊 Veri Analizi ve Model Görselleri", on_click=sayfa_degistir, args=('analiz',), use_container_width=True)

# ==========================================
# 2. EKRAN: KREDİ RİSK TAHMİNİ 
# ==========================================
elif st.session_state.aktif_sayfa == 'tahmin':
    st.button("⬅️ Ana Menüye Dön", on_click=sayfa_degistir, args=('ana_menu',))
    st.write("---")
    
    st.title("Kredi Onay Tahmin Paneli")
    
    if model is None or scaler is None:
        st.error("Model (rf_model.pkl) veya Scaler (scaler.pkl) dosyaları klasörde bulunamadı!")
    else:
        with st.form("tahmin_formu"):
            c1, c2 = st.columns(2)
            
            with c1:
                age = st.number_input("Yaş", min_value=18, max_value=100, value=28)
                income = st.number_input("Yıllık Gelir ($)", min_value=0, max_value=10000000, value=55000)
                emp_len = st.number_input("Çalışma Süresi (Yıl)", min_value=0, max_value=70, value=4)
                amount = st.number_input("Kredi Miktarı ($)", min_value=0, max_value=5000000, value=8000)
                cred_hist = st.number_input("Kredi Geçmişi Uzunluğu (Yıl)", min_value=0, max_value=50, value=3)

            with c2:
                rate = st.number_input("Faiz Oranı (%)", min_value=0.0, max_value=100.0, value=11.0, format="%.2f")
                home = st.selectbox("Ev Durumu", sorted(df['person_home_ownership'].unique()))
                intent = st.selectbox("Kredi Amacı", sorted(df['loan_intent'].unique()))
                grade = st.selectbox("Risk Grubu (Grade)", sorted(df['loan_grade'].unique()))
                default = st.selectbox("Daha Önce Temerrüde Düştü mü?", sorted(df['cb_person_default_on_file'].unique()))
            
            submit = st.form_submit_button("Müşteri Riskini Analiz Et", use_container_width=True)
            
            if submit:
                input_row = pd.DataFrame([{
                    'person_age': age, 'person_income': income, 'person_emp_length': emp_len,
                    'loan_amnt': amount, 'loan_int_rate': rate, 'loan_percent_income': amount / income if income > 0 else 0,
                    'cb_person_cred_hist_length': cred_hist, 'person_home_ownership': home,
                    'loan_intent': intent, 'loan_grade': grade, 'cb_person_default_on_file': default
                }])
                
                user_encoded = pd.get_dummies(input_row)
                expected_columns = scaler.feature_names_in_
                user_encoded = user_encoded.reindex(columns=expected_columns, fill_value=0).astype(float)
                
                user_scaled = scaler.transform(user_encoded)
                
                risk_ihtimali = model.predict_proba(user_scaled)[0][1] 
                geri_odeme_ihtimali = 1 - risk_ihtimali 
                
                st.write("---")
                st.subheader("Yapay Zeka Analiz Sonucu")
                
                st.info(f"💡 **Model Öngörüsü:** Yapay zeka, müşteri verilerini baz alarak bu kişinin kredisini sorunsuz bir şekilde geri ödeme ihtimalini **%{geri_odeme_ihtimali*100:.1f}** olarak hesaplamıştır.")
                
                st.progress(int(geri_odeme_ihtimali * 100))
                
                st.caption(f"*(Müşterinin temerrüde düşme/ödememe riski: %{risk_ihtimali*100:.1f})*")
                st.write("⚠️ *Not: Nihai onay veya ret kararı, bankanın güncel risk politikalarına ve ekonomik göstergelere bağlı olarak banka yetkilisi tarafından verilmelidir.*")

# ==========================================
# 3. EKRAN: VERİ ANALİZİ VE GÖRSELLER
# ==========================================
elif st.session_state.aktif_sayfa == 'analiz':
    st.button("⬅️ Ana Menüye Dön", on_click=sayfa_degistir, args=('ana_menu',))
    st.write("---")
    
    st.title("Veri Seti Analiz Görselleri")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Ev Sahipliğine Göre Geri Ödeme")
        fig1, ax1 = plt.subplots()
        home_ownership = pd.crosstab(df['person_home_ownership'], df['loan_status'], normalize='index')
        home_ownership.plot(kind='bar', ax=ax1, color=['#2ecc71', '#e74c3c'])
        plt.xticks(rotation=0)
        st.pyplot(fig1)

    with col2:
        st.subheader("Kredi Amacına Göre Risk")
        fig2, ax2 = plt.subplots()
        intent = pd.crosstab(df['loan_intent'], df['loan_status'], normalize='index')
        intent.plot(kind='bar', ax=ax2, color=['#3498db', '#f1c40f'])
        st.pyplot(fig2)

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Risk Notu (Grade) Dağılımı")
        fig3, ax3 = plt.subplots()
        sns.countplot(x='loan_grade', hue='loan_status', data=df, palette='magma', order=['A', 'B', 'C', 'D', 'E', 'F', 'G'], ax=ax3)
        st.pyplot(fig3)

    with col4:
        st.subheader("Gelir/Kredi Oranı Yoğunluğu")
        df_paid = df[df['loan_status'] == 0].copy()
        df_paid['loan_percent_rounded'] = df_paid['loan_percent_income'].round(2)
        line_data = df_paid.groupby('loan_percent_rounded').size().reset_index(name='count')
        fig4, ax4 = plt.subplots()
        sns.lineplot(data=line_data, x='loan_percent_rounded', y='count', color='blue', ax=ax4)
        st.pyplot(fig4)

    st.write("---")
    st.subheader("Sayısal Değişkenlerin Korelasyonu")
    fig5, ax5 = plt.subplots(figsize=(10, 5))
    corr = df.select_dtypes(include=['int64', 'float64']).corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax5)
    st.pyplot(fig5)