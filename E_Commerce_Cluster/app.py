import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="E-Commerce Customer Clustering",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
    <style>
    .main-title {
        font-size: 3em;
        font-weight: bold;
        text-align: center;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Load models and data
@st.cache_resource
def load_models():
    kmeans_model = joblib.load('kmeans_model.joblib')
    scaler = joblib.load('scaler.joblib')
    best_k = joblib.load('best_k.joblib')
    return kmeans_model, scaler, best_k

@st.cache_data
def load_data():
    df = pd.read_csv('ecommerce_customers_LPA.csv')
    return df

# Load models and data
kmeans_model, scaler, best_k = load_models()
df = load_data()

# Scale original data for predictions
features = ['Age', 'Annual_Income_LPA', 'Spending_Score']
X = df[features].values
x_scaled = scaler.transform(X)

# Add cluster labels to original data
df['Cluster'] = kmeans_model.fit_predict(x_scaled)

# Define segment names
cluster_names = {
    0: "Balanced Customers",
    1: "Young Spenders",
    2: "Seniors • Low Spending",
    3: "High Income • High Spending"
}

df['Segment'] = df['Cluster'].map(cluster_names)

# Sidebar navigation
st.sidebar.title("🛍️ Navigation")
page = st.sidebar.radio("Select a page:", [
    "📊 Dashboard",
    "🔍 Cluster Analysis",
    "👤 Predict Customer Segment",
    "📈 Model Insights",
    "📋 Data Explorer"
])

# ==================== PAGE: DASHBOARD ====================
if page == "📊 Dashboard":
    st.markdown("<h1 class='main-title'>E-Commerce Customer Clustering Dashboard</h1>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Customers", len(df))
    with col2:
        st.metric("Number of Clusters", best_k)
    with col3:
        st.metric("Number of Features", len(features))
    with col4:
        st.metric("Largest Cluster Size", df['Cluster'].value_counts().max())
    
    st.divider()
    
    # Cluster Distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Cluster Distribution")
        cluster_counts = df['Cluster'].value_counts().sort_index()
        fig = go.Figure(data=[
            go.Bar(
                x=[cluster_names.get(i, f"Cluster {i}") for i in cluster_counts.index],
                y=cluster_counts.values,
                marker=dict(color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(cluster_counts)])
            )
        ])
        fig.update_layout(
            title="Customer Distribution Across Clusters",
            xaxis_title="Segment",
            yaxis_title="Number of Customers",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Cluster Composition")
        cluster_pct = (df['Cluster'].value_counts().sort_index() / len(df) * 100).round(1)
        fig = go.Figure(data=[
            go.Pie(
                labels=[cluster_names.get(i, f"Cluster {i}") for i in cluster_pct.index],
                values=cluster_pct.values,
                marker=dict(colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(cluster_pct)])
            )
        ])
        fig.update_layout(title="Percentage Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # 3D Scatter Plot
    st.subheader("3D Cluster Visualization")
    fig = px.scatter_3d(
        df,
        x='Age',
        y='Annual_Income_LPA',
        z='Spending_Score',
        color='Segment',
        hover_data=['Age', 'Annual_Income_LPA', 'Spending_Score'],
        color_discrete_sequence=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    )
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)

# ==================== PAGE: CLUSTER ANALYSIS ====================
elif page == "🔍 Cluster Analysis":
    st.title("🔍 Detailed Cluster Analysis")
    
    # Cluster Profile Table
    st.subheader("Cluster Profiles (Average Values)")
    profile = df.groupby('Cluster')[features].mean().round(2)
    profile.index = profile.index.map(lambda x: cluster_names.get(x, f"Cluster {x}"))
    st.dataframe(profile, use_container_width=True)
    
    st.divider()
    
    # Cluster Characteristics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Income vs Spending by Segment")
        fig = px.scatter(
            df,
            x='Annual_Income_LPA',
            y='Spending_Score',
            color='Segment',
            size='Age',
            hover_data=['Age', 'Annual_Income_LPA', 'Spending_Score'],
            color_discrete_sequence=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        )
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Age Distribution by Segment")
        fig = px.box(
            df,
            x='Segment',
            y='Age',
            color='Segment',
            color_discrete_sequence=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        )
        fig.update_layout(height=450, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # Segment Descriptions
    st.subheader("📌 Segment Insights")
    
    descriptions = {
        0: {
            "title": "Balanced Customers",
            "description": "Mid-range income and spending, likely core customers",
            "characteristics": "Moderate spending habits, stable customer base"
        },
        1: {
            "title": "Young Spenders",
            "description": "Younger age group with high spending tendencies",
            "characteristics": "High engagement, trend-conscious, brand-loyal potential"
        },
        2: {
            "title": "Seniors • Low Spending",
            "description": "Older customers with lower spending scores",
            "characteristics": "Price-sensitive, require personalized offers"
        },
        3: {
            "title": "High Income • High Spending",
            "description": "Premium customers with high income and spending",
            "characteristics": "VIP segment, luxury products, high lifetime value"
        }
    }
    
    for cluster_id in range(best_k):
        if cluster_id in descriptions:
            desc = descriptions[cluster_id]
            count = len(df[df['Cluster'] == cluster_id])
            pct = (count / len(df) * 100)
            
            st.markdown(f"""
            **{desc['title']}** ({count} customers, {pct:.1f}%)
            - {desc['description']}
            - Characteristics: {desc['characteristics']}
            """)

# ==================== PAGE: PREDICT CUSTOMER SEGMENT ====================
elif page == "👤 Predict Customer Segment":
    st.title("👤 Predict Customer Segment")
    
    st.info("Enter customer details to predict which segment they belong to")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
    
    with col2:
        income = st.number_input("Annual Income (LPA)", min_value=0.0, max_value=500.0, value=50.0)
    
    with col3:
        spending = st.number_input("Spending Score", min_value=0, max_value=100, value=50)
    
    if st.button("🔮 Predict Segment", use_container_width=True):
        # Scale the input
        user_data = np.array([[age, income, spending]])
        user_data_scaled = scaler.transform(user_data)
        
        # Predict cluster
        predicted_cluster = kmeans_model.predict(user_data_scaled)[0]
        segment_name = cluster_names.get(predicted_cluster, f"Cluster {predicted_cluster}")
        
        st.success(f"✅ Predicted Segment: **{segment_name}**")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Age", age)
        with col2:
            st.metric("Annual Income (LPA)", f"₹{income:.0f}L")
        with col3:
            st.metric("Spending Score", spending)
        
        st.divider()
        
        # Compare with cluster profile
        st.subheader("Comparison with Cluster Profile")
        
        cluster_profile = df[df['Cluster'] == predicted_cluster][features].mean()
        comparison_df = pd.DataFrame({
            'Metric': features,
            'Your Value': [age, income, spending],
            'Cluster Average': cluster_profile.values
        })
        
        st.dataframe(comparison_df, use_container_width=True)
        
        # Recommendations
        st.subheader("💡 Recommendations")
        recommendations = {
            0: [
                "Offer balanced promotions and discounts",
                "Focus on loyalty programs",
                "Regular engagement through email campaigns"
            ],
            1: [
                "Promote trending products and limited editions",
                "Use social media marketing",
                "Create exclusive youth-focused campaigns"
            ],
            2: [
                "Provide senior citizen discounts",
                "Focus on quality and reliability",
                "Simplify the purchasing process"
            ],
            3: [
                "Offer premium products and services",
                "Create VIP membership programs",
                "Provide personalized shopping experiences"
            ]
        }
        
        if predicted_cluster in recommendations:
            for i, rec in enumerate(recommendations[predicted_cluster], 1):
                st.write(f"{i}. {rec}")

# ==================== PAGE: MODEL INSIGHTS ====================
elif page == "📈 Model Insights":
    st.title("📈 Model Insights & Performance")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Model Type", "K-Means Clustering")
    with col2:
        st.metric("Optimal Clusters (k)", best_k)
    with col3:
        st.metric("Features Used", len(features))
    
    st.divider()
    
    st.subheader("Feature Statistics")
    
    stats_df = df[features].describe().round(2)
    st.dataframe(stats_df, use_container_width=True)
    
    st.divider()
    
    # Distribution plots
    st.subheader("Feature Distributions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        fig = px.histogram(df, x='Age', nbins=30, color_discrete_sequence=['#1f77b4'])
        fig.update_layout(title="Age Distribution", height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.histogram(df, x='Annual_Income_LPA', nbins=30, color_discrete_sequence=['#ff7f0e'])
        fig.update_layout(title="Income Distribution", height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        fig = px.histogram(df, x='Spending_Score', nbins=30, color_discrete_sequence=['#2ca02c'])
        fig.update_layout(title="Spending Score Distribution", height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # Correlation heatmap
    st.subheader("Feature Correlation Matrix")
    
    corr_matrix = df[features].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=features,
        y=features,
        colorscale='RdBu',
        zmid=0,
        text=corr_matrix.values.round(2),
        texttemplate='%{text}',
        textfont={"size": 12}
    ))
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

# ==================== PAGE: DATA EXPLORER ====================
elif page == "📋 Data Explorer":
    st.title("📋 Data Explorer")
    
    # Filter options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        selected_segments = st.multiselect(
            "Select Segments",
            options=df['Segment'].unique(),
            default=df['Segment'].unique()
        )
    
    with col2:
        min_age = st.number_input("Minimum Age", min_value=int(df['Age'].min()), max_value=int(df['Age'].max()), value=int(df['Age'].min()))
        max_age = st.number_input("Maximum Age", min_value=int(df['Age'].min()), max_value=int(df['Age'].max()), value=int(df['Age'].max()))
    
    with col3:
        min_income = st.number_input("Minimum Income", min_value=float(df['Annual_Income_LPA'].min()), max_value=float(df['Annual_Income_LPA'].max()), value=float(df['Annual_Income_LPA'].min()))
        max_income = st.number_input("Maximum Income", min_value=float(df['Annual_Income_LPA'].min()), max_value=float(df['Annual_Income_LPA'].max()), value=float(df['Annual_Income_LPA'].max()))
    
    # Apply filters
    filtered_df = df[
        (df['Segment'].isin(selected_segments)) &
        (df['Age'] >= min_age) & (df['Age'] <= max_age) &
        (df['Annual_Income_LPA'] >= min_income) & (df['Annual_Income_LPA'] <= max_income)
    ].copy()
    
    st.subheader(f"Showing {len(filtered_df)} of {len(df)} customers")
    
    # Display data
    st.dataframe(filtered_df, use_container_width=True)
    
    # Download option
    csv = filtered_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Filtered Data as CSV",
        data=csv,
        file_name="filtered_customers.csv",
        mime="text/csv"
    )
    
    st.divider()
    
    # Summary statistics
    st.subheader("Summary Statistics (Filtered Data)")
    st.dataframe(filtered_df[features].describe().round(2), use_container_width=True)

# Footer
st.divider()
st.markdown("""
    <div style='text-align: center; color: grey; font-size: 12px; margin-top: 50px;'>
    🛍️ E-Commerce Customer Clustering Analysis | Built with Streamlit
    </div>
    """, unsafe_allow_html=True)
