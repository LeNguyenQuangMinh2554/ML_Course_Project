"""
🏥 NY Hospital Charges Prediction - GUI Application (Improved Version)
=======================================================================
Đề tài: Phân tích và Dự đoán Chi phí Nhập viện tại các Bệnh viện Bang New York (2009)

Môn: Machine Learning - Kỳ thi cuối kỳ
Hướng dẫn chạy:
    1. Cài đặt thư viện: pip install streamlit pandas numpy scikit-learn plotly seaborn
    2. Chạy ứng dụng: streamlit run gui_app.py

Cải tiến so với phiên bản cũ:
    - Thêm Feature Engineering (LOS_Group, Urgency_Level, Age_Risk)
    - Thêm xử lý Outliers bằng IQR
    - Thêm Data Cleaning (duplicates, missing)
    - Sử dụng Pipeline với Scaler như notebook
    - Thêm nhiều models và scalers hơn
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import warnings
warnings.filterwarnings("ignore")

# ========================== CẤU HÌNH TRANG ==========================
st.set_page_config(
    page_title="NY Hospital Charges Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========================== CUSTOM CSS ==========================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E3A8A;
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #E0F2FE 0%, #BAE6FD 100%);
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        font-size: 2rem;
        font-weight: bold;
        box-shadow: 0 8px 25px rgba(0,0,0,0.3);
        margin: 1rem 0;
    }
    .info-box {
        background: #F0F9FF;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #0EA5E9;
        margin: 1rem 0;
        color: #1a1a1a !important;
    }
    .section-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# ========================== FEATURE ENGINEERING ==========================
def categorize_los(los):
    """Phân nhóm số ngày nằm viện"""
    if los <= 2:
        return 'Short (1-2 days)'
    elif los <= 5:
        return 'Medium (3-5 days)'
    elif los <= 10:
        return 'Long (6-10 days)'
    else:
        return 'Extended (>10 days)'

def categorize_urgency(admission_type):
    """Phân nhóm mức độ khẩn cấp"""
    if admission_type in ['Emergency', 'Trauma']:
        return 'High Urgency'
    elif admission_type == 'Urgent':
        return 'Medium Urgency'
    else:
        return 'Low Urgency'

def categorize_age_risk(age_group):
    """Phân nhóm rủi ro theo tuổi"""
    if age_group in ['0 to 17', '70 or Older']:
        return 'High Risk Age'
    elif age_group in ['50 to 69']:
        return 'Medium Risk Age'
    else:
        return 'Low Risk Age'


# ========================== LOAD DATA ==========================
@st.cache_data
def load_and_clean_data():
    """Load và làm sạch dữ liệu theo đúng quy trình notebook"""
    try:
        df = pd.read_csv('NY Hospital Admissions - Dataset.csv')
        original_len = len(df)
        
        # 1. Xóa duplicates
        df = df.drop_duplicates()
        after_dup = len(df)
        
        # 2. Xóa missing values
        df = df.dropna()
        after_missing = len(df)
        
        # 3. Xử lý outliers bằng IQR
        Q1 = df['Total Charges'].quantile(0.25)
        Q3 = df['Total Charges'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df['Total Charges'] >= lower_bound) & (df['Total Charges'] <= upper_bound)]
        after_outliers = len(df)
        
        # 4. Feature Engineering
        df['LOS_Group'] = df['Length of Stay'].apply(categorize_los)
        df['Urgency_Level'] = df['Admission Type'].apply(categorize_urgency)
        df['Age_Risk'] = df['Age Group'].apply(categorize_age_risk)
        
        cleaning_info = {
            'original': original_len,
            'after_dup': after_dup,
            'after_missing': after_missing,
            'after_outliers': after_outliers,
            'duplicates_removed': original_len - after_dup,
            'missing_removed': after_dup - after_missing,
            'outliers_removed': after_missing - after_outliers
        }
        
        return df, cleaning_info
    except Exception as e:
        st.error(f"❌ Không thể load dữ liệu: {e}")
        return None, None


# ========================== TRAIN MODEL ==========================
@st.cache_resource
def train_models(df):
    """Train multiple models với Pipeline và Scaler như notebook"""
    
    if df is None or len(df) == 0:
        return None, None, None, None, None
    
    # Sample data để training nhanh hơn
    if len(df) > 100000:
        df_model = df.sample(n=100000, random_state=42).copy()
    else:
        df_model = df.copy()
    
    # Encode categorical variables
    le_dict = {}
    categorical_cols = ['Service Area', 'Name', 'Age Group', 'Gender', 'Race', 'Ethnicity', 
                        'Admission Type', 'LOS_Group', 'Urgency_Level', 'Age_Risk']
    
    for col in categorical_cols:
        if col in df_model.columns:
            le = LabelEncoder()
            df_model[col + '_encoded'] = le.fit_transform(df_model[col].astype(str))
            le_dict[col] = le
    
    # Create features
    feature_cols = [col + '_encoded' for col in categorical_cols if col in df_model.columns]
    if 'Length of Stay' in df_model.columns:
        feature_cols.append('Length of Stay')
    
    X = df_model[feature_cols].values.astype(np.float64)
    y = df_model['Total Charges'].values.astype(np.float64)
    
    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Models và Scalers (như notebook)
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'Ridge Regression': Ridge(alpha=1.0),
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(random_state=42),
        'Lasso': Lasso(alpha=1.0),
    }
    
    scalers = {
        'StandardScaler': StandardScaler(),
        'MinMaxScaler': MinMaxScaler(),
        'RobustScaler': RobustScaler()
    }
    
    results = []
    best_model = None
    best_scaler = None
    best_score = -np.inf
    best_name = ""
    best_scaler_name = ""
    
    for model_name, model in models.items():
        for scaler_name, scaler in scalers.items():
            try:
                # Scale features
                scaler_instance = type(scaler)()
                X_train_scaled = scaler_instance.fit_transform(X_train)
                X_test_scaled = scaler_instance.transform(X_test)
                
                # Train model
                model_instance = type(model)(**model.get_params())
                model_instance.fit(X_train_scaled, y_train)
                y_pred = model_instance.predict(X_test_scaled)
                
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae = mean_absolute_error(y_test, y_pred)
                
                results.append({
                    'Model': model_name,
                    'Scaler': scaler_name,
                    'R2': r2,
                    'RMSE': rmse,
                    'MAE': mae,
                    'model_obj': model_instance,
                    'scaler_obj': scaler_instance
                })
                
                if r2 > best_score:
                    best_score = r2
                    best_model = model_instance
                    best_scaler = scaler_instance
                    best_name = model_name
                    best_scaler_name = scaler_name
                    
            except Exception as e:
                continue
    
    return best_model, best_scaler, best_name, best_scaler_name, results, le_dict, feature_cols


def predict_charges(model, scaler, le_dict, feature_cols, input_data):
    """Dự đoán chi phí"""
    encoded_input = []
    categorical_cols = ['Service Area', 'Name', 'Age Group', 'Gender', 'Race', 'Ethnicity', 
                        'Admission Type', 'LOS_Group', 'Urgency_Level', 'Age_Risk']
    
    for col in categorical_cols:
        col_encoded = col + '_encoded'
        if col_encoded in feature_cols and col in le_dict:
            try:
                encoded_val = le_dict[col].transform([input_data[col]])[0]
            except:
                encoded_val = 0
            encoded_input.append(encoded_val)
    
    if 'Length of Stay' in feature_cols:
        encoded_input.append(input_data['Length of Stay'])
    
    try:
        input_scaled = scaler.transform([encoded_input])
        prediction = model.predict(input_scaled)[0]
        return max(0, prediction)
    except Exception as e:
        return 0


# ========================== MAIN APP ==========================
def main():
    st.markdown('<div class="main-header">🏥 Dự đoán Chi phí Nhập viện - Bệnh viện New York (2009)</div>', 
                unsafe_allow_html=True)
    
    # Load data
    df, cleaning_info = load_and_clean_data()
    if df is None:
        st.stop()
    
    # Train models
    with st.spinner("🔄 Đang huấn luyện models... (có thể mất 1-2 phút)"):
        result = train_models(df)
        if result[0] is None:
            st.error("❌ Không thể train model!")
            st.stop()
        best_model, best_scaler, best_name, best_scaler_name, results, le_dict, feature_cols = result
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 📋 Thông tin Dự án")
        st.markdown("""
        **Đề tài:** Phân tích, dự đoán chi phí nhập viện tại các bệnh viện bang New York (2009)
        
        **Môn học:** Machine Learning - Final Exam
        """)
        
        st.markdown("---")
        st.markdown("### 🏆 Model tốt nhất")
        st.success(f"**{best_name}**")
        st.info(f"Scaler: {best_scaler_name}")
        
        # Tìm R2 của best model
        best_r2 = next((r['R2'] for r in results if r['Model'] == best_name and r['Scaler'] == best_scaler_name), 0)
        st.metric("R² Score", f"{best_r2:.4f}")
        
        st.markdown("---")
        st.markdown("### 📊 Data Cleaning")
        st.write(f"📁 Original: {cleaning_info['original']:,}")
        st.write(f"🔄 Duplicates removed: {cleaning_info['duplicates_removed']:,}")
        st.write(f"❌ Missing removed: {cleaning_info['missing_removed']:,}")
        st.write(f"📉 Outliers removed: {cleaning_info['outliers_removed']:,}")
        st.write(f"✅ Final: {cleaning_info['after_outliers']:,}")
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔮 Dự đoán", 
        "📊 EDA", 
        "🤖 Models", 
        "📈 Thống kê",
        "🔧 Feature Engineering"
    ])
    
    # ==================== TAB 1: DỰ ĐOÁN ====================
    with tab1:
        st.markdown("### 🔮 Nhập thông tin để dự đoán chi phí nhập viện")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📍 Thông tin Bệnh viện")
            service_area = st.selectbox("Khu vực dịch vụ", options=sorted(df['Service Area'].unique()))
            
            # Lọc bệnh viện theo khu vực
            hospitals_in_area = df[df['Service Area'] == service_area]['Name'].dropna().unique()
            hospital_name = st.selectbox("Tên Bệnh viện 🏥", options=sorted(hospitals_in_area) if len(hospitals_in_area) > 0 else ['N/A'])
            
            # Lọc quận theo bệnh viện đã chọn
            counties = df[(df['Service Area'] == service_area) & (df['Name'] == hospital_name)]['County'].dropna().unique()
            county = st.selectbox("Quận/Hạt", options=sorted(counties) if len(counties) > 0 else ['N/A'])
        
        with col2:
            st.markdown("#### 👤 Thông tin Bệnh nhân")
            age_group = st.selectbox("Nhóm tuổi", options=['0 to 17', '18 to 29', '30 to 49', '50 to 69', '70 or Older'])
            gender = st.radio("Giới tính", options=['M', 'F'], horizontal=True)
            race = st.selectbox("Chủng tộc", options=df['Race'].dropna().unique())
            ethnicity = st.selectbox("Dân tộc", options=df['Ethnicity'].dropna().unique())
        
        st.markdown("---")
        col3, col4 = st.columns(2)
        
        with col3:
            admission_type = st.radio("Loại nhập viện", 
                                      options=['Elective', 'Emergency', 'Urgent', 'Trauma', 'Newborn'], 
                                      horizontal=True)
        with col4:
            length_of_stay = st.slider("Số ngày nằm viện", min_value=1, max_value=60, value=5)
        
        st.markdown("---")
        
        if st.button("🔮 DỰ ĐOÁN CHI PHÍ", type="primary", use_container_width=True):
            # Tính feature engineering
            los_group = categorize_los(length_of_stay)
            urgency_level = categorize_urgency(admission_type)
            age_risk = categorize_age_risk(age_group)
            
            input_data = {
                'Service Area': service_area,
                'Name': hospital_name,
                'Age Group': age_group,
                'Gender': gender,
                'Race': race,
                'Ethnicity': ethnicity,
                'Admission Type': admission_type,
                'Length of Stay': length_of_stay,
                'LOS_Group': los_group,
                'Urgency_Level': urgency_level,
                'Age_Risk': age_risk
            }
            
            prediction = predict_charges(best_model, best_scaler, le_dict, feature_cols, input_data)
            
            st.markdown(f'<div class="prediction-box">💰 Chi phí dự đoán: ${prediction:,.2f}</div>', 
                       unsafe_allow_html=True)
            
            col_a, col_b, col_c, col_d = st.columns(4)
            with col_a:
                st.metric("📅 Số ngày", f"{length_of_stay} ngày")
            with col_b:
                st.metric("💵 Chi phí/ngày", f"${prediction/length_of_stay:,.2f}")
            with col_c:
                category = "Thấp" if prediction < 15000 else "Trung bình" if prediction < 35000 else "Cao"
                st.metric("📊 Mức chi phí", category)
            with col_d:
                st.metric("⚠️ Mức khẩn cấp", urgency_level.split()[0])
            
            avg_charge = df['Total Charges'].mean()
            diff = prediction - avg_charge
            diff_percent = (diff / avg_charge) * 100
            
            st.markdown(f"""
            <div class="info-box">
                <strong>📈 So sánh với trung bình:</strong><br>
                Chi phí TB toàn bộ: <strong>${avg_charge:,.2f}</strong> | 
                Chênh lệch: <strong>${diff:+,.2f}</strong> ({diff_percent:+.1f}%)
            </div>
            """, unsafe_allow_html=True)
            
            # Thông tin feature engineering
            st.markdown("#### 🔧 Các đặc trưng được tạo tự động:")
            fe_col1, fe_col2, fe_col3 = st.columns(3)
            with fe_col1:
                st.info(f"📅 LOS Group: **{los_group}**")
            with fe_col2:
                st.info(f"⚡ Urgency: **{urgency_level}**")
            with fe_col3:
                st.info(f"👴 Age Risk: **{age_risk}**")
    
    # ==================== TAB 2: EDA ====================
    with tab2:
        st.markdown("### 📊 Exploratory Data Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Phân phối Chi phí (Total Charges)")
            fig1 = px.histogram(df, x='Total Charges', nbins=50, 
                               title='Phân phối Chi phí nhập viện',
                               color_discrete_sequence=['#3498db'])
            fig1.update_layout(xaxis_title='Chi phí ($)', yaxis_title='Số lượng')
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            st.markdown("#### Chi phí theo Loại Nhập viện")
            fig2 = px.box(df, x='Admission Type', y='Total Charges', 
                         color='Admission Type',
                         title='Chi phí theo Admission Type')
            st.plotly_chart(fig2, use_container_width=True)
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.markdown("#### Chi phí theo Nhóm tuổi")
            fig3 = px.box(df, x='Age Group', y='Total Charges', color='Age Group',
                         title='Chi phí theo Age Group')
            st.plotly_chart(fig3, use_container_width=True)
        
        with col4:
            st.markdown("#### Chi phí theo Khu vực")
            avg_by_area = df.groupby('Service Area')['Total Charges'].mean().sort_values(ascending=True)
            fig4 = px.bar(x=avg_by_area.values, y=avg_by_area.index, orientation='h',
                         title='Chi phí TB theo Service Area',
                         color=avg_by_area.values,
                         color_continuous_scale='Blues')
            fig4.update_layout(xaxis_title='Chi phí TB ($)', yaxis_title='Khu vực')
            st.plotly_chart(fig4, use_container_width=True)
        
        st.markdown("#### Mối quan hệ: Số ngày nằm viện vs Chi phí")
        sample_df = df.sample(n=min(5000, len(df)), random_state=42)
        fig5 = px.scatter(sample_df, x='Length of Stay', y='Total Charges', 
                         color='Admission Type', opacity=0.6,
                         title='Length of Stay vs Total Charges')
        fig5.update_layout(xaxis_title='Số ngày nằm viện', yaxis_title='Chi phí ($)')
        st.plotly_chart(fig5, use_container_width=True)
    
    # ==================== TAB 3: MODELS ====================
    with tab3:
        st.markdown("### 🤖 So sánh các Models")
        
        # Tạo DataFrame kết quả
        results_df = pd.DataFrame([
            {
                'Model': r['Model'],
                'Scaler': r['Scaler'],
                'R² Score': r['R2'],
                'RMSE ($)': r['RMSE'],
                'MAE ($)': r['MAE']
            }
            for r in results
        ]).sort_values('R² Score', ascending=False)
        
        # Format columns
        results_display = results_df.copy()
        results_display['R² Score'] = results_display['R² Score'].apply(lambda x: f"{x:.4f}")
        results_display['RMSE ($)'] = results_display['RMSE ($)'].apply(lambda x: f"${x:,.2f}")
        results_display['MAE ($)'] = results_display['MAE ($)'].apply(lambda x: f"${x:,.2f}")
        
        st.dataframe(results_display, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # R2 Score chart
            fig_r2 = px.bar(results_df.head(10), x='R² Score', y='Model', color='Scaler',
                           title='Top 10 Models theo R² Score',
                           orientation='h', barmode='group')
            st.plotly_chart(fig_r2, use_container_width=True)
        
        with col2:
            # RMSE chart
            fig_rmse = px.bar(results_df.head(10), x='RMSE ($)', y='Model', color='Scaler',
                             title='Top 10 Models theo RMSE',
                             orientation='h', barmode='group')
            st.plotly_chart(fig_rmse, use_container_width=True)
        
        st.success(f"🏆 **Best Model:** {best_name} + {best_scaler_name} (R² = {best_r2:.4f})")
    
    # ==================== TAB 4: THỐNG KÊ ====================
    with tab4:
        st.markdown("### 📈 Thống kê Dataset")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("📊 Tổng Records", f"{len(df):,}")
        with col2:
            st.metric("💰 Chi phí TB", f"${df['Total Charges'].mean():,.0f}")
        with col3:
            st.metric("📈 Chi phí Max", f"${df['Total Charges'].max():,.0f}")
        with col4:
            st.metric("📉 Chi phí Min", f"${df['Total Charges'].min():,.0f}")
        with col5:
            st.metric("🏥 Số Bệnh viện", f"{df['Name'].nunique():,}")
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Thống kê mô tả")
            st.dataframe(df.describe(), use_container_width=True)
        
        with col2:
            st.markdown("#### Top 10 Bệnh viện chi phí cao nhất")
            top_hospitals = df.groupby('Name')['Total Charges'].mean().sort_values(ascending=False).head(10)
            top_df = pd.DataFrame({
                'Bệnh viện': top_hospitals.index,
                'Chi phí TB ($)': top_hospitals.values
            })
            top_df['Chi phí TB ($)'] = top_df['Chi phí TB ($)'].apply(lambda x: f"${x:,.2f}")
            st.dataframe(top_df, use_container_width=True, hide_index=True)
    
    # ==================== TAB 5: FEATURE ENGINEERING ====================
    with tab5:
        st.markdown("### 🔧 Feature Engineering")
        
        st.markdown("""
        <div class="info-box">
            <strong>📝 Giải thích:</strong> Feature Engineering là quá trình tạo ra các biến mới từ dữ liệu gốc 
            để giúp model dự đoán chính xác hơn. Dựa trên phân tích EDA, chúng tôi đã tạo 3 biến mới:
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 📅 LOS_Group")
            st.markdown("""
            Phân nhóm số ngày nằm viện:
            - **Short**: 1-2 ngày
            - **Medium**: 3-5 ngày
            - **Long**: 6-10 ngày
            - **Extended**: >10 ngày
            """)
            fig1 = px.pie(df, names='LOS_Group', title='Phân bố LOS Group')
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            st.markdown("#### ⚡ Urgency_Level")
            st.markdown("""
            Mức độ khẩn cấp:
            - **High**: Emergency, Trauma
            - **Medium**: Urgent
            - **Low**: Elective, Newborn
            """)
            fig2 = px.pie(df, names='Urgency_Level', title='Phân bố Urgency Level')
            st.plotly_chart(fig2, use_container_width=True)
        
        with col3:
            st.markdown("#### 👴 Age_Risk")
            st.markdown("""
            Rủi ro theo độ tuổi:
            - **High Risk**: 0-17, 70+
            - **Medium Risk**: 50-69
            - **Low Risk**: 18-49
            """)
            fig3 = px.pie(df, names='Age_Risk', title='Phân bố Age Risk')
            st.plotly_chart(fig3, use_container_width=True)
        
        st.markdown("---")
        st.markdown("#### Chi phí theo các Feature mới")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            fig = px.box(df, x='LOS_Group', y='Total Charges', color='LOS_Group',
                        title='Chi phí theo LOS Group')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.box(df, x='Urgency_Level', y='Total Charges', color='Urgency_Level',
                        title='Chi phí theo Urgency Level')
            st.plotly_chart(fig, use_container_width=True)
        
        with col3:
            fig = px.box(df, x='Age_Risk', y='Total Charges', color='Age_Risk',
                        title='Chi phí theo Age Risk')
            st.plotly_chart(fig, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <center>
        🎓 <strong>Machine Learning - Final Exam Project</strong><br>
        Đề tài: Phân tích và Dự đoán Chi phí Nhập viện tại các Bệnh viện Bang New York (2009)
    </center>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
