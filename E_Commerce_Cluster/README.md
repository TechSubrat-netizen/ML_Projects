# 🛍️ E-Commerce Customer Clustering - Streamlit Web App

A comprehensive web application for e-commerce customer segmentation using K-means clustering.

## 📋 Overview

This Streamlit app provides interactive visualization and analysis of customer segments based on:
- **Age**
- **Annual Income (LPA)**
- **Spending Score**

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the App
```bash
streamlit run app.py
```

The app will automatically open at `http://localhost:8501`

## 📊 Features

### 📊 Dashboard
- Overview metrics and KPIs
- Cluster distribution visualization
- 3D cluster scatter plot
- Real-time customer statistics

### 🔍 Cluster Analysis
- Detailed cluster profiles with average values
- Income vs. Spending visualization
- Age distribution by segment
- Segment insights and characteristics

### 👤 Predict Customer Segment
- Input customer details (Age, Income, Spending Score)
- Get instant segment prediction
- Compare with cluster profile
- Receive targeted recommendations

### 📈 Model Insights
- Feature statistics and distributions
- Correlation matrix heatmap
- Model performance details
- Technical model information

### 📋 Data Explorer
- Filter customers by segment and demographics
- Advanced filtering options
- Download filtered data as CSV
- Summary statistics

## 📁 Project Structure

```
E_Commerce_Cluster/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── kmeans_model.joblib            # Trained K-means model
├── scaler.joblib                  # StandardScaler for feature scaling
├── best_k.joblib                  # Optimal number of clusters
├── ecommerce_customers_LPA.csv    # Customer data
├── e_commerce_cluster.ipynb       # Original Jupyter notebook
└── README.md                       # This file
```

## 🎯 Customer Segments

### 1. **Balanced Customers**
   - Mid-range income and spending
   - Core customer base
   - Focus: Loyalty programs and regular engagement

### 2. **Young Spenders**
   - Younger age group with high spending
   - Trend-conscious and brand-loyal
   - Focus: Social media marketing and trending products

### 3. **Seniors • Low Spending**
   - Older customers with lower spending
   - Price-sensitive
   - Focus: Senior discounts and simplified experience

### 4. **High Income • High Spending**
   - Premium segment with high income
   - High lifetime value
   - Focus: VIP programs and luxury products

## 🛠️ Technical Details

- **Model**: K-Means Clustering
- **Features**: 3 (Age, Annual Income, Spending Score)
- **Clusters**: 4 (Optimal k determined by silhouette score)
- **Preprocessing**: StandardScaler normalization
- **Framework**: Streamlit + Plotly

## 📊 Data Specifications

- **Total Customers**: ~2,000
- **Features**: Age (18-100), Annual Income (0-500 LPA), Spending Score (0-100)
- **Format**: CSV

## 💾 Model Files

All models are pre-trained and ready to use:
- `kmeans_model.joblib` - The trained K-means clustering model
- `scaler.joblib` - Feature scaler for normalization
- `best_k.joblib` - The optimal number of clusters (k=4)

## 🎨 UI Features

- **Dark/Light Mode**: Automatic based on system settings
- **Responsive Design**: Works on desktop, tablet, and mobile
- **Interactive Charts**: Hover for details, zoom, and export
- **Real-time Filters**: Update visualizations instantly
- **Download Options**: Export filtered data as CSV

## 📱 Browser Support

- Chrome/Chromium ✅
- Firefox ✅
- Safari ✅
- Edge ✅

## 🔧 Customization

To modify segment names or descriptions, edit the `cluster_names` dictionary in `app.py`:

```python
cluster_names = {
    0: "Your Segment Name",
    1: "Another Name",
    2: "Third Segment",
    3: "Fourth Segment"
}
```

## 🚀 Deployment

### Deploy on Streamlit Cloud
1. Push to GitHub
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud)
3. Connect your repo and select this app
4. App will be live in seconds!

### Deploy on Heroku
1. Create `Procfile`: `web: streamlit run app.py`
2. Deploy as usual

## 📊 Performance Notes

- App loads models from joblib files (fast instantiation)
- Data cached after first load
- Interactive visualizations use Plotly (smooth performance)
- Suitable for 5,000+ customers

## 🐛 Troubleshooting

**"Module not found" error**
```bash
pip install -r requirements.txt
```

**App not opening**
- Check if Streamlit is running: `streamlit run app.py`
- Default URL: `http://localhost:8501`

**Slow performance**
- Clear cache: Press 'R' in Streamlit
- Reduce data size if needed

## 📧 Support

For issues or feature requests, check the notebook for analysis details or review the model performance metrics in the Model Insights section.

---

**Created with ❤️ using Streamlit | ML Model: K-Means Clustering**
