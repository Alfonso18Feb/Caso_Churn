import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px

# Set page config
st.set_page_config(
    page_title="Churn Predictor Pro",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for Premium Look
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stApp {
        background: linear-gradient(135deg, #0e1117 0%, #161b22 100%);
        color: #e6edf3;
    }
    h1, h2, h3 {
        color: #58a6ff !important;
        font-family: 'Inter', sans-serif;
        font-weight: 800;
    }
    .stButton>button {
        background: linear-gradient(90deg, #1f6feb 0%, #388bfd 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(56, 139, 253, 0.4);
    }
    .prediction-card {
        background: #161b22;
        padding: 2rem;
        border-radius: 16px;
        border: 1px solid #30363d;
        backdrop-filter: blur(10px);
        margin-bottom: 2rem;
    }
    .stNumberInput, .stSelectbox, .stRadio {
        background: #0d1117 !important;
        border-radius: 8px !important;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_artifacts():
    try:
        return joblib.load('churn_model_artifacts.joblib')
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def preprocess_single_row(input_data, artifacts):
    training_columns = artifacts['training_columns']
    encoder = artifacts['encoder']
    high_card_cols = artifacts['high_card_cols']
    
    # Start with a base of zeros for all training columns
    df_base = pd.DataFrame(0, index=[0], columns=training_columns)
    
    # Fill in the numerical/categorical data provided
    for col, val in input_data.items():
        if col in high_card_cols and encoder:
            # Handle high-cardinality ordinal encoding
            val_encoded = encoder.transform(pd.DataFrame([[str(val)]], columns=[col]))[0][0]
            df_base.loc[0, col] = val_encoded
        elif col in training_columns:
            # Direct mapping for numerical or non-OHE categorical
            df_base.loc[0, col] = val
        else:
            # Check for One-Hot Encoded columns (e.g., GENERO_M, GENERO_F)
            ohe_col = f"{col}_{val}"
            if ohe_col in training_columns:
                df_base.loc[0, ohe_col] = 1
                
    return df_base.astype(float)

# App Header
st.title("🔮 Churn Analytics Pro")
st.markdown("### Predicting Customer Loyalty")

artifacts = load_artifacts()

if artifacts:
    # Sidebar Inputs
    st.sidebar.header("👤 Customer Features")
    
    with st.sidebar:
        with st.form("customer_form"):
            st.write("Enter Customer Details")
            
            # Key Inputs (Inferred from training features)
            edad = st.number_input("Age (Edad)", min_value=18, max_value=100, value=35)
            modelo = st.selectbox("Vehicle Model (Modelo)", ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"])
            genero = st.selectbox("Gender (GENERO)", ["M", "F"])
            origen = st.selectbox("Origin (Origen)", ["A", "B", "C", "D", "E"])
            motivo = st.selectbox("Sale Reason (MOTIVO_VENTA)", ["Particular", "Empresa"])
            queja = st.radio("Has Complaint (QUEJA)?", ["N", "Y"])
            
            # Numeric inputs
            pvp = st.number_input("PVP (€)", min_value=0.0, value=25000.0)
            margen = st.number_input("Margin (€)", value=5000.0)
            
            st.divider()
            st.write("Service History")
            mantenimiento = st.checkbox("Free Maintenance (MANTENIMIENTO_GRATUITO)?", value=False)
            fue_lead = st.checkbox("Was Lead (Fue_Lead)?", value=False)
            kw = st.number_input("Horsepower (Kw)", min_value=0.0, value=100.0)
            
            # Submit button
            submitted = st.form_submit_button("Predict Churn Risk")

    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["🎯 Prediction", "📈 Model Performance", "🔍 Feature Importance"])

    with tab1:
        col1, col2 = st.columns([1, 2])
        
        # We'll use a placeholder for the results in the sidebar-less view if not submitted
        with col2:
            if submitted:
                # Construct input data dictionary
                input_data = {
                    'Edad': edad,
                    'Modelo': modelo,
                    'GENERO': genero,
                    'Origen': origen,
                    'MOTIVO_VENTA': motivo,
                    'QUEJA': queja,
                    'PVP': pvp,
                    'Margen_eur': margen,
                    'Margen_eur_bruto': margen * 1.2,
                    'MANTENIMIENTO_GRATUITO': 1 if mantenimiento else 0,
                    'Fue_Lead': 1 if fue_lead else 0,
                    'Kw': kw
                }
                
                # Preprocess
                processed_input = preprocess_single_row(input_data, artifacts)
                
                # Predict
                prediction = artifacts['model'].predict(processed_input)[0]
                probability = artifacts['model'].predict_proba(processed_input)[0][1]
                probability = 1-probability # User custom inversion
                
                st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
                
                res_col1, res_col2 = st.columns(2)
                
                # Risk Categorization based on Probability
                if probability < 0.2:
                    risk_label = "LOW"
                    risk_color = "success"
                    risk_symbol = "✅"
                elif probability < 0.4:
                    risk_label = "MEDIUM LOW"
                    risk_color = "success"
                    risk_symbol = "🟡"
                elif probability < 0.6:
                    risk_label = "MEDIUM"
                    risk_color = "warning"
                    risk_symbol = "🟠"
                elif probability < 0.8:
                    risk_label = "MEDIUM HIGH"
                    risk_color = "error"
                    risk_symbol = "🔴"
                else:
                    risk_label = "HIGH"
                    risk_color = "error"
                    risk_symbol = "⚠️"

                with res_col1:
                    st.subheader("Results")
                    status_text = f"{risk_symbol} {risk_label} CHURN RISK"
                    if risk_color == "success":
                        st.success(status_text)
                    elif risk_color == "warning":
                        st.warning(status_text)
                    else:
                        st.error(status_text)
                    
                    st.metric("Probability", f"{probability:.2%}")
                
                with res_col2:
                    # Gauge Chart with 5 segments
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = probability * 100,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Risk Level Scale"},
                        gauge = {
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "#388bfd"},
                            'steps': [
                                {'range': [0, 20], 'color': "rgba(40, 167, 69, 0.3)"},
                                {'range': [20, 40], 'color': "rgba(173, 255, 47, 0.3)"},
                                {'range': [40, 60], 'color': "rgba(255, 193, 7, 0.3)"},
                                {'range': [60, 80], 'color': "rgba(255, 127, 80, 0.3)"},
                                {'range': [80, 100], 'color': "rgba(251, 114, 114, 0.3)"}
                            ],
                        }
                    ))
                    fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor="rgba(0,0,0,0)", font={'color': "#e6edf3"})
                    st.plotly_chart(fig, use_container_width=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info("👈 Enter customer details in the sidebar and click 'Predict Churn Risk' to see the analysis.")

    with tab2:
        st.header("📈 Model Performance Metrics")
        metrics = artifacts['metrics']
        
        m_col1, m_col2 = st.columns(2)
        
        with m_col1:
            # ROC Curve
            if 'fpr' in metrics and 'tpr' in metrics:
                fig_roc = go.Figure()
                fig_roc.add_trace(go.Scatter(x=metrics['fpr'], y=metrics['tpr'], name=f"ROC Curve (AUC={metrics['roc_auc']:.2f})", mode='lines', line=dict(color='#58a6ff', width=3)))
                fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Random", mode='lines', line=dict(dash='dash', color='gray')))
                fig_roc.update_layout(title="ROC Curve", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", height=400, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={'color': "#e6edf3"})
                st.plotly_chart(fig_roc, use_container_width=True)
            else:
                st.warning("ROC Curve data not found. Please re-run `train_final_model.py`.")

            # Confusion Matrix
            st.subheader("Confusion Matrix")
            cm = metrics['conf_matrix']
            fig_cm = px.imshow(cm, text_auto=True, labels=dict(x="Predicted", y="Actual", color="Count"), x=['No Churn', 'Churn'], y=['No Churn', 'Churn'], color_continuous_scale='Blues')
            fig_cm.update_layout(height=400, paper_bgcolor="rgba(0,0,0,0)", font={'color': "#e6edf3"})
            st.plotly_chart(fig_cm, use_container_width=True)

        with m_col2:
            # PR Curve
            if 'precision_curve' in metrics and 'recall_curve' in metrics:
                fig_pr = go.Figure()
                fig_pr.add_trace(go.Scatter(x=metrics['recall_curve'], y=metrics['precision_curve'], name=f"PR Curve (AUC={metrics['pr_auc']:.2f})", mode='lines', line=dict(color='#388bfd', width=3)))
                fig_pr.update_layout(title="Precision-Recall Curve", xaxis_title="Recall", yaxis_title="Precision", height=400, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={'color': "#e6edf3"})
                st.plotly_chart(fig_pr, use_container_width=True)
            else:
                st.warning("PR Curve data not found.")
            
            # Summary Metrics Card
            st.markdown(f"""
            <div style="background: #161b22; padding: 1.5rem; border-radius: 12px; border: 1px solid #30363d;">
                <h4>🏆 Model Performance Summary</h4>
                <p><b>Model Name:</b> {artifacts.get('name', 'N/A')}</p>
                <p><b>F1 Score:</b> {metrics['f1']:.4f}</p>
                <p><b>Accuracy:</b> {metrics['accuracy']:.4f}</p>
                <p><b>Precision:</b> {metrics['precision']:.4f}</p>
                <p><b>Recall:</b> {metrics['recall']:.4f}</p>
            </div>
            """, unsafe_allow_html=True)

    with tab3:
        st.header("🔍 Feature Importance")
        if hasattr(artifacts['model'], 'feature_importances_'):
            importances = artifacts['model'].feature_importances_
            feat_df = pd.DataFrame({
                'Feature': artifacts['feature_names'],
                'Importance': importances
            }).sort_values(by='Importance', ascending=False).head(15)
            
            fig_imp = px.bar(feat_df, x='Importance', y='Feature', orientation='h',
                            title="Top 15 Influential Factors",
                            color='Importance',
                            color_continuous_scale='Blues')
            fig_imp.update_layout(height=600, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={'color': "#e6edf3"})
            st.plotly_chart(fig_imp, use_container_width=True)

else:
    st.warning("Please run `train_final_model.py` first to generate the model artifacts.")

# Footer
st.divider()
st.markdown("BCustomer Intelligence Platform")
