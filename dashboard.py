"""
Streamlit Dashboard for AI Model Monitoring
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import sqlite3
import os
from monitoring.advanced_monitoring import AdvancedMetricsCollector

# Page config
st.set_page_config(
    page_title="AI Model Monitoring Dashboard",
    page_icon="📊",
    layout="wide"
)

# Title
st.title("📊 AI Model Monitoring Dashboard")
st.markdown("Real-time monitoring of AI model performance, data drift, and system metrics")

# Initialize metrics collector
@st.cache_resource
def get_metrics_collector():
    return AdvancedMetricsCollector()

metrics_collector = get_metrics_collector()

# Sidebar controls
st.sidebar.header("Dashboard Controls")
time_range = st.sidebar.selectbox(
    "Time Range",
    ["Last Hour", "Last 6 Hours", "Last 24 Hours", "Last 7 Days"],
    index=2
)

refresh_rate = st.sidebar.slider("Auto-refresh (seconds)", 10, 300, 60)

# Convert time range to hours
time_ranges = {
    "Last Hour": 1,
    "Last 6 Hours": 6,
    "Last 24 Hours": 24,
    "Last 7 Days": 168
}
hours = time_ranges[time_range]

# Auto-refresh
if st.sidebar.button("🔄 Refresh Now"):
    st.rerun()

# Main dashboard content
col1, col2, col3, col4 = st.columns(4)

# Get metrics summary
try:
    summary = metrics_collector.get_metrics_summary(hours)
    
    # Key metrics cards
    with col1:
        if "total_request_ms" in summary["metrics"]:
            avg_latency = summary["metrics"]["total_request_ms"]["average"]
            st.metric(
                label="Avg Response Time",
                value=f"{avg_latency:.1f}ms",
                delta=f"{summary['metrics']['total_request_ms']['count']} requests"
            )
        else:
            st.metric("Avg Response Time", "N/A")
    
    with col2:
        if "retrieval_confidence" in summary["metrics"]:
            avg_confidence = summary["metrics"]["retrieval_confidence"]["average"]
            st.metric(
                label="Avg Confidence",
                value=f"{avg_confidence:.2f}",
                delta=f"{summary['metrics']['retrieval_confidence']['count']} queries"
            )
        else:
            st.metric("Avg Confidence", "N/A")
    
    with col3:
        if summary["model_performance"]:
            total_requests = sum(perf["requests"] for perf in summary["model_performance"].values())
            st.metric(
                label="Total Requests",
                value=f"{total_requests:,}",
                delta=f"{len(summary['model_performance'])} models"
            )
        else:
            st.metric("Total Requests", "0")
    
    with col4:
        if "retrieval_latency_ms" in summary["metrics"]:
            avg_retrieval = summary["metrics"]["retrieval_latency_ms"]["average"]
            st.metric(
                label="Avg Retrieval Time",
                value=f"{avg_retrieval:.1f}ms",
                delta=f"{summary['metrics']['retrieval_latency_ms']['count']} retrievals"
            )
        else:
            st.metric("Avg Retrieval Time", "N/A")

except Exception as e:
    st.error(f"Error loading metrics: {e}")

# Charts section
st.header("📈 Performance Trends")

# Create tabs for different views
tab1, tab2, tab3, tab4 = st.tabs(["Response Times", "Confidence Scores", "Model Performance", "Data Drift"])

with tab1:
    st.subheader("Response Time Trends")
    
    # Get metrics data for plotting
    try:
        conn = sqlite3.connect("./monitoring_metrics.db")
        cutoff_time = time.time() - (hours * 3600)
        
        # Response time metrics
        df_latency = pd.read_sql_query("""
            SELECT datetime(timestamp, 'unixepoch') as datetime, value
            FROM metrics 
            WHERE name IN ('total_request_ms', 'retrieval_latency_ms', 'summarization_latency_ms')
            AND timestamp > ?
            ORDER BY timestamp
        """, conn, params=(cutoff_time,))
        
        if not df_latency.empty:
            # Create line chart
            fig = px.line(df_latency, x='datetime', y='value', 
                         color='name', title='Response Time Trends',
                         labels={'value': 'Time (ms)', 'datetime': 'Time'})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No response time data available for the selected period")
        
        conn.close()
    except Exception as e:
        st.error(f"Error loading response time data: {e}")

with tab2:
    st.subheader("Confidence Score Trends")
    
    try:
        conn = sqlite3.connect("./monitoring_metrics.db")
        cutoff_time = time.time() - (hours * 3600)
        
        # Confidence metrics
        df_confidence = pd.read_sql_query("""
            SELECT datetime(timestamp, 'unixepoch') as datetime, value
            FROM metrics 
            WHERE name = 'retrieval_confidence'
            AND timestamp > ?
            ORDER BY timestamp
        """, conn, params=(cutoff_time,))
        
        if not df_confidence.empty:
            # Create histogram and line chart
            col1, col2 = st.columns(2)
            
            with col1:
                fig_hist = px.histogram(df_confidence, x='value', 
                                      title='Confidence Score Distribution',
                                      labels={'value': 'Confidence Score', 'count': 'Frequency'})
                st.plotly_chart(fig_hist, use_container_width=True)
            
            with col2:
                fig_line = px.line(df_confidence, x='datetime', y='value',
                                 title='Confidence Score Over Time',
                                 labels={'value': 'Confidence Score', 'datetime': 'Time'})
                st.plotly_chart(fig_line, use_container_width=True)
        else:
            st.info("No confidence data available for the selected period")
        
        conn.close()
    except Exception as e:
        st.error(f"Error loading confidence data: {e}")

with tab3:
    st.subheader("Model Performance Overview")
    
    try:
        conn = sqlite3.connect("./monitoring_metrics.db")
        cutoff_time = time.time() - (hours * 3600)
        
        # Model performance data
        df_performance = pd.read_sql_query("""
            SELECT datetime(timestamp, 'unixepoch') as datetime, 
                   model_name, accuracy, latency_ms, confidence, 
                   input_tokens, output_tokens, cost_usd
            FROM model_performance 
            WHERE timestamp > ?
            ORDER BY timestamp
        """, conn, params=(cutoff_time,))
        
        if not df_performance.empty:
            # Performance metrics by model
            model_summary = df_performance.groupby('model_name').agg({
                'accuracy': 'mean',
                'latency_ms': 'mean',
                'confidence': 'mean',
                'input_tokens': 'sum',
                'output_tokens': 'sum',
                'cost_usd': 'sum'
            }).round(3)
            
            st.dataframe(model_summary, use_container_width=True)
            
            # Performance trends
            fig_perf = px.scatter(df_performance, x='latency_ms', y='accuracy',
                                color='model_name', size='confidence',
                                title='Model Performance: Accuracy vs Latency',
                                labels={'latency_ms': 'Latency (ms)', 'accuracy': 'Accuracy'})
            st.plotly_chart(fig_perf, use_container_width=True)
        else:
            st.info("No model performance data available for the selected period")
        
        conn.close()
    except Exception as e:
        st.error(f"Error loading model performance data: {e}")

with tab4:
    st.subheader("Data Drift Detection")
    
    try:
        conn = sqlite3.connect("./monitoring_metrics.db")
        cutoff_time = time.time() - (hours * 3600)
        
        # Data drift data
        df_drift = pd.read_sql_query("""
            SELECT datetime(timestamp, 'unixepoch') as datetime,
                   feature_name, drift_score, p_value, threshold
            FROM data_drift 
            WHERE timestamp > ?
            ORDER BY timestamp DESC
        """, conn, params=(cutoff_time,))
        
        if not df_drift.empty:
            # Drift alerts
            drift_alerts = df_drift[df_drift['drift_score'] > df_drift['threshold']]
            
            if not drift_alerts.empty:
                st.warning(f"⚠️ {len(drift_alerts)} drift alerts detected!")
                st.dataframe(drift_alerts[['datetime', 'feature_name', 'drift_score', 'p_value']], 
                           use_container_width=True)
            else:
                st.success("✅ No significant data drift detected")
            
            # Drift score trends
            fig_drift = px.line(df_drift, x='datetime', y='drift_score',
                              color='feature_name', title='Data Drift Scores Over Time',
                              labels={'drift_score': 'Drift Score', 'datetime': 'Time'})
            st.plotly_chart(fig_drift, use_container_width=True)
        else:
            st.info("No data drift data available for the selected period")
        
        conn.close()
    except Exception as e:
        st.error(f"Error loading data drift data: {e}")

# System status
st.header("🔧 System Status")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Database Status")
    try:
        conn = sqlite3.connect("./monitoring_metrics.db")
        cursor = conn.cursor()
        
        # Get table sizes
        tables = ['metrics', 'model_performance', 'data_drift']
        for table in tables:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            st.metric(f"{table.title()} Records", f"{count:,}")
        
        conn.close()
        st.success("✅ Database connection healthy")
    except Exception as e:
        st.error(f"❌ Database error: {e}")

with col2:
    st.subheader("API Status")
    try:
        import requests
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            st.success("✅ Main API healthy")
        else:
            st.warning("⚠️ Main API issues")
    except:
        st.error("❌ Main API unreachable")
    
    try:
        import requests
        response = requests.get("http://localhost:8001/docs", timeout=5)
        if response.status_code == 200:
            st.success("✅ MCP Server healthy")
        else:
            st.warning("⚠️ MCP Server issues")
    except:
        st.error("❌ MCP Server unreachable")

# Auto-refresh
if st.sidebar.checkbox("Auto-refresh enabled"):
    time.sleep(refresh_rate)
    st.rerun()
