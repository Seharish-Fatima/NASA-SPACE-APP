import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from scipy import stats
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Set page config
st.set_page_config(
    page_title="RNA-Seq Analysis Dashboard",
    page_icon="🧬",
    layout="wide",
)

# Custom CSS with new interactive elements styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stPlotlyChart {
        background-color: #ffffff;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .highlight {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# Sample data
@st.cache_data
def load_data():
    data = {
        'Sample': ['BI1', 'BI2', 'BI3', 'BI4', 'BI5'],
        'RIN': [4.9, 5.6, 6.6, 5.5, 5.7],
        'rRNA_Contamination': [3.78, 3.02, 1.53, 0.60, 1.86],
        'Read_Depth': [54477267, 61115053, 63712483, 63108146, 56208952]
    }
    return pd.DataFrame(data)

df = load_data()

# New Statistical Analysis Functions
def calculate_statistics(data, column):
    stats_dict = {
        'Mean': np.mean(data[column]),
        'Median': np.median(data[column]),
        'Std Dev': np.std(data[column]),
        'CV (%)': (np.std(data[column]) / np.mean(data[column])) * 100,
        'Skewness': stats.skew(data[column]),
        'Kurtosis': stats.kurtosis(data[column]),
        'Z-score range': f"{min(stats.zscore(data[column])):.2f} to {max(stats.zscore(data[column])):.2f}"
    }
    return stats_dict

def detect_outliers(data, column, threshold=2):
    z_scores = np.abs(stats.zscore(data[column]))
    return data[z_scores > threshold]

# Title and Introduction with interactive elements
st.title("🧬 RNA-Seq Quality Metrics Dashboard")
with st.expander("ℹ️ About this Dashboard"):
    st.markdown("""
    This interactive dashboard provides comprehensive analysis of RNA sequencing quality metrics.
    
    **Key Features:**
    - Real-time quality metric visualization
    - Statistical analysis and outlier detection
    - Interactive threshold adjustment
    - Custom visualization options
    - Export capabilities
    """)

# Interactive Settings in Sidebar
with st.sidebar:
    st.header("Dashboard Settings")
    
    # Visualization Settings
    st.subheader("Visualization Options")
    color_theme = st.selectbox(
        "Color Theme",
        ["Default", "Viridis", "Plasma", "Inferno", "Magma"]
    )
    
    show_statistics = st.checkbox("Show Advanced Statistics", True)
    show_outliers = st.checkbox("Show Outlier Detection", True)
    
    # Quality Thresholds with real-time feedback
    st.subheader("Quality Thresholds")
    rin_threshold = st.slider(
        "RIN Score Threshold", 
        0.0, 10.0, 4.0,
        help="Minimum acceptable RIN score"
    )
    
    rrna_threshold = st.slider(
        "rRNA Contamination Threshold (%)", 
        0.0, 10.0, 5.0,
        help="Maximum acceptable rRNA contamination"
    )
    
    read_depth_threshold = st.slider(
        "Read Depth Threshold (M)", 
        0, 100, 50,
        help="Minimum acceptable read depth in millions"
    )

# Main Dashboard Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "Quality Metrics", 
    "Statistical Analysis",
    "Advanced Analytics",
    "Quality Report"
])

with tab1:
    # Interactive Quality Metrics Display
    st.header("Interactive Quality Metrics")
    
    # Metric Selection
    selected_metric = st.selectbox(
        "Select Primary Metric for Analysis",
        ["RIN", "rRNA_Contamination", "Read_Depth"]
    )
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Dynamic visualization based on selected metric
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=df['Sample'],
            y=df[selected_metric],
            name=selected_metric,
            text=df[selected_metric].round(2),
            textposition='auto',
        ))
        
        # Add threshold line
        if selected_metric == "RIN":
            threshold = rin_threshold
        elif selected_metric == "rRNA_Contamination":
            threshold = rrna_threshold
        else:
            threshold = read_depth_threshold * 1e6
            
        fig.add_hline(
            y=threshold,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Threshold: {threshold}"
        )
        
        fig.update_layout(title=f"{selected_metric} Analysis")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Real-time statistics for selected metric
        st.subheader("Quick Statistics")
        stats_dict = calculate_statistics(df, selected_metric)
        for stat, value in stats_dict.items():
            st.metric(stat, f"{value:.2f}" if isinstance(value, float) else value)

with tab2:
    st.header("Statistical Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Interactive Statistical Tests
        st.subheader("Statistical Tests")
        
        # Normality Test
        stat, p_value = stats.normaltest(df[selected_metric])
        st.write("Normality Test (D'Agostino's K^2 Test)")
        st.write(f"p-value: {p_value:.4f}")
        st.write("Interpretation:", 
                "Normal distribution" if p_value > 0.05 else "Non-normal distribution")
        
        # Outlier Detection with adjustable threshold
        z_score_threshold = st.slider(
            "Z-score threshold for outlier detection",
            1.0, 4.0, 2.0,
            help="Higher values are more conservative in outlier detection"
        )
        
        outliers = detect_outliers(df, selected_metric, z_score_threshold)
        if not outliers.empty:
            st.write("Detected Outliers:")
            st.dataframe(outliers)
        else:
            st.write("No outliers detected with current threshold")
    
    with col2:
        # Distribution Plot
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(
            x=df[selected_metric],
            nbinsx=10,
            name="Distribution"
        ))
        fig_dist.add_trace(go.Scatter(
            x=df[selected_metric],
            y=[0] * len(df),
            mode='markers',
            name="Data Points",
            marker=dict(size=10)
        ))
        fig_dist.update_layout(title="Distribution Analysis")
        st.plotly_chart(fig_dist, use_container_width=True)

with tab3:
    st.header("Advanced Analytics")
    
    # PCA Analysis
    st.subheader("Principal Component Analysis")
    
    # Prepare data for PCA
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[['RIN', 'rRNA_Contamination', 'Read_Depth']])
    pca = PCA()
    pca_result = pca.fit_transform(scaled_data)
    
    # Plot explained variance ratio
    fig_pca = go.Figure()
    fig_pca.add_trace(go.Bar(
        x=['PC1', 'PC2', 'PC3'],
        y=pca.explained_variance_ratio_,
        text=[f"{v:.1%}" for v in pca.explained_variance_ratio_],
        textposition='auto',
    ))
    fig_pca.update_layout(title="Explained Variance Ratio by Principal Components")
    st.plotly_chart(fig_pca, use_container_width=True)
    
    # Interactive Correlation Analysis
    st.subheader("Interactive Correlation Analysis")
    
    correlation_method = st.selectbox(
        "Select Correlation Method",
        ["Pearson", "Spearman"]
    )
    
    if correlation_method == "Pearson":
        corr_matrix = df[['RIN', 'rRNA_Contamination', 'Read_Depth']].corr(method='pearson')
    else:
        corr_matrix = df[['RIN', 'rRNA_Contamination', 'Read_Depth']].corr(method='spearman')
    
    fig_corr = px.imshow(
        corr_matrix,
        labels=dict(x="Metric", y="Metric", color="Correlation"),
        color_continuous_scale=color_theme.lower() if color_theme != "Default" else "RdBu"
    )
    st.plotly_chart(fig_corr, use_container_width=True)

with tab4:
    st.header("Quality Report Generator")
    
    # Interactive Report Settings
    report_settings = st.multiselect(
        "Select Report Components",
        ["Basic Statistics", "Outlier Analysis", "Distribution Analysis", "Quality Assessment"],
        default=["Basic Statistics", "Quality Assessment"]
    )
    
    if st.button("Generate Report"):
        st.markdown("### RNA-Seq Quality Report")
        st.markdown(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        if "Basic Statistics" in report_settings:
            st.subheader("Basic Statistics")
            for metric in ['RIN', 'rRNA_Contamination', 'Read_Depth']:
                st.write(f"\n**{metric} Statistics:**")
                stats_dict = calculate_statistics(df, metric)
                st.table(pd.DataFrame([stats_dict]).T)
        
        if "Outlier Analysis" in report_settings:
            st.subheader("Outlier Analysis")
            for metric in ['RIN', 'rRNA_Contamination', 'Read_Depth']:
                outliers = detect_outliers(df, metric)
                if not outliers.empty:
                    st.write(f"\n**Outliers detected in {metric}:**")
                    st.dataframe(outliers)
                else:
                    st.write(f"\nNo outliers detected in {metric}")
        
        if "Quality Assessment" in report_settings:
            st.subheader("Overall Quality Assessment")
            quality_summary = pd.DataFrame({
                'Metric': ['RIN', 'rRNA Contamination', 'Read Depth'],
                'Status': [
                    '✅ Pass' if df['RIN'].mean() >= rin_threshold else '❌ Fail',
                    '✅ Pass' if df['rRNA_Contamination'].mean() <= rrna_threshold else '❌ Fail',
                    '✅ Pass' if df['Read_Depth'].mean() >= read_depth_threshold * 1e6 else '❌ Fail'
                ],
                'Comment': [
                    f"Mean RIN: {df['RIN'].mean():.2f}",
                    f"Mean rRNA: {df['rRNA_Contamination'].mean():.2f}%",
                    f"Mean Depth: {df['Read_Depth'].mean()/1e6:.1f}M"
                ]
            })
            st.table(quality_summary)

# Export Options
st.header("Export Options")
col1, col2 = st.columns(2)

with col1:
    # Export raw data
    csv = df.to_csv(index=False)
    st.download_button(
        label="Download Raw Data (CSV)",
        data=csv,
        file_name="rna_seq_metrics.csv",
        mime="text/csv",
    )

with col2:
    # Export analysis results
    if st.button("Export Analysis Report"):
        # Create a comprehensive report
        report = f"""RNA-Seq Analysis Report
        Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
        
        Summary Statistics:
        {pd.DataFrame([calculate_statistics(df, selected_metric)]).T.to_string()}
        
        Quality Assessment:
        - RIN Score: {'Pass' if df['RIN'].mean() >= rin_threshold else 'Fail'}
        - rRNA Contamination: {'Pass' if df['rRNA_Contamination'].mean() <= rrna_threshold else 'Fail'}
        - Read Depth: {'Pass' if df['Read_Depth'].mean() >= read_depth_threshold * 1e6 else 'Fail'}
        """
        
        st.download_button(
            label="Download Analysis Report",
            data=report,
            file_name="rna_seq_analysis_report.txt",
            mime="text/plain",
        )