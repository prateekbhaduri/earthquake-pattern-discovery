import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Earthquake Pattern Discovery",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Global */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Background */
.stApp { background: #0a0e1a; color: #e2e8f0; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f1629 0%, #111827 100%);
    border-right: 1px solid #1e293b;
}
section[data-testid="stSidebar"] * { color: #cbd5e1 !important; }
section[data-testid="stSidebar"] .stSlider > div { color: #94a3b8 !important; }

/* Metric cards */
div[data-testid="metric-container"] {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 16px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.4);
}
div[data-testid="metric-container"] label { color: #94a3b8 !important; font-size: 0.8rem !important; }
div[data-testid="metric-container"] [data-testid="stMetricValue"] { color: #f8fafc !important; font-size: 1.8rem !important; font-weight: 700 !important; }
div[data-testid="metric-container"] [data-testid="stMetricDelta"] { color: #38bdf8 !important; }

/* Headers */
h1 { background: linear-gradient(135deg, #f97316, #ef4444, #a855f7); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 800 !important; }
h2 { color: #f1f5f9 !important; font-weight: 600 !important; }
h3 { color: #cbd5e1 !important; font-weight: 500 !important; }

/* Tabs */
button[data-baseweb="tab"] { background: #1e293b !important; color: #94a3b8 !important; border-radius: 8px 8px 0 0 !important; border: 1px solid #334155 !important; font-weight: 500 !important; }
button[data-baseweb="tab"][aria-selected="true"] { background: linear-gradient(135deg, #f97316, #ef4444) !important; color: white !important; border-color: transparent !important; }

/* Expander */
details { background: #1e293b; border: 1px solid #334155; border-radius: 10px; padding: 4px; }

/* Info / warning boxes */
div[data-testid="stAlert"] { border-radius: 10px !important; border: 1px solid #334155 !important; background: #1e293b !important; }

/* Select / Slider */
.stSelectbox div[data-baseweb="select"] > div { background: #1e293b !important; border-color: #334155 !important; color: #e2e8f0 !important; }
.stSlider .stSlider { accent-color: #f97316; }

/* Badge strip */
.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 99px;
    font-size: 0.72rem;
    font-weight: 600;
    margin: 2px;
}
.badge-red   { background: #7f1d1d; color: #fca5a5; border: 1px solid #991b1b; }
.badge-orange{ background: #7c2d12; color: #fdba74; border: 1px solid #9a3412; }
.badge-green { background: #14532d; color: #86efac; border: 1px solid #166534; }
.badge-blue  { background: #1e3a5f; color: #93c5fd; border: 1px solid #1d4ed8; }

/* Divider */
hr { border-color: #1e293b !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0f172a; }
::-webkit-scrollbar-thumb { background: #334155; border-radius: 99px; }

.hero-card {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border: 1px solid #334155;
    border-radius: 14px;
    padding: 20px 24px;
    margin-bottom: 12px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.5);
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("query.csv")
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df["year"]  = df["time"].dt.year
    df["month"] = df["time"].dt.month
    df["hour"]  = df["time"].dt.hour
    df["depth_cat"] = pd.cut(df["depth"],
                              bins=[-1, 70, 300, 700],
                              labels=["Shallow (<70km)", "Intermediate (70-300km)", "Deep (>300km)"])
    df["mag_cat"] = pd.cut(df["mag"],
                            bins=[4.5, 5, 6, 7, 10],
                            labels=["Moderate (4.5-5)", "Strong (5-6)", "Major (6-7)", "Great (7+)"])
    return df

df_full = load_data()

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR – GLOBAL FILTERS
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌍 Control Panel")
    st.markdown("---")

    st.markdown("### 📊 Data Filters")
    mag_range = st.slider("Magnitude Range", float(df_full.mag.min()), float(df_full.mag.max()),
                          (4.5, float(df_full.mag.max())), 0.1)
    depth_range = st.slider("Depth Range (km)", float(df_full.depth.min()), float(df_full.depth.max()),
                             (0.0, float(df_full.depth.max())), 1.0)
    year_range = st.slider("Year Range", int(df_full.year.min()), int(df_full.year.max()),
                            (int(df_full.year.min()), int(df_full.year.max())))

    st.markdown("---")
    st.markdown("### 🤖 Clustering Algorithm")
    algo = st.selectbox("Algorithm", ["DBSCAN (Density)", "K-Means (Centroid)"])

    if "DBSCAN" in algo:
        eps_val   = st.slider("ε (neighborhood radius)", 0.5, 10.0, 2.0, 0.5,
                               help="Max distance between points in a cluster")
        min_samp  = st.slider("Min Samples", 2, 30, 5,
                               help="Min points to form a dense region")
    else:
        n_clusters = st.slider("Number of Clusters (K)", 2, 15, 6)

    st.markdown("---")
    st.markdown("### 🗺️ Map Style")
    map_style = st.selectbox("Basemap", ["carto-darkmatter", "open-street-map", "satellite-streets"])
    marker_size_scale = st.slider("Marker Size Scale", 1, 10, 4)

    st.markdown("---")
    st.markdown('<p style="color:#475569;font-size:0.75rem;text-align:center;">Earthquake Pattern Discovery v1.0</p>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# FILTER DATA
# ─────────────────────────────────────────────────────────────────────────────
df = df_full[
    (df_full.mag   >= mag_range[0])  & (df_full.mag   <= mag_range[1]) &
    (df_full.depth >= depth_range[0])& (df_full.depth <= depth_range[1]) &
    (df_full.year  >= year_range[0]) & (df_full.year  <= year_range[1])
].copy()

# ─────────────────────────────────────────────────────────────────────────────
# CLUSTERING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def run_clustering(algo, df_hash, eps_val=2.0, min_samp=5, n_clusters=6):
    coords = df[["latitude", "longitude", "depth", "mag"]].dropna()
    scaler = StandardScaler()
    X = scaler.fit_transform(coords)

    if "DBSCAN" in algo:
        model = DBSCAN(eps=eps_val, min_samples=min_samp, metric="euclidean", n_jobs=-1)
    else:
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)

    labels = model.fit_predict(X)
    return coords.index, labels

idx, labels = run_clustering(algo, str(df.shape), eps_val if "DBSCAN" in algo else 2.0,
                              min_samp if "DBSCAN" in algo else 5,
                              n_clusters if "K-Means" in algo else 6)

df.loc[idx, "cluster"] = labels
df["cluster"] = df["cluster"].fillna(-2).astype(int)
df["cluster_label"] = df["cluster"].apply(lambda x: "Noise" if x == -1 else f"Cluster {x+1}" if x >= 0 else "Unclustered")

# Cluster stats
cluster_stats = df[df["cluster"] >= 0].groupby("cluster").agg(
    count=("mag", "count"),
    avg_mag=("mag", "mean"),
    max_mag=("mag", "max"),
    avg_depth=("depth", "mean"),
    center_lat=("latitude", "mean"),
    center_lon=("longitude", "mean"),
).reset_index()
cluster_stats["risk_score"] = (cluster_stats["avg_mag"] * 0.5 +
                                cluster_stats["max_mag"] * 0.3 +
                                (1 / (cluster_stats["avg_depth"] + 1)) * 20 +
                                np.log1p(cluster_stats["count"]) * 0.5).round(2)
cluster_stats = cluster_stats.sort_values("risk_score", ascending=False)

n_clusters_found = (df["cluster"] >= 0).sum()
n_noise          = (df["cluster"] == -1).sum()
n_valid          = len(df)

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("# 🌋 Earthquake Pattern Discovery")
st.markdown("**Spatiotemporal Clustering · Seismic Zone Detection · Risk Analysis**")
st.markdown("---")

# KPI Row
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("📊 Earthquakes", f"{n_valid:,}", f"of {len(df_full):,} total")
k2.metric("🔴 Clusters Found", int(df[df.cluster>=0]["cluster"].nunique()))
k3.metric("⚡ Max Magnitude", f"{df.mag.max():.1f}")
k4.metric("🌊 Avg Depth", f"{df.depth.mean():.1f} km")
k5.metric("📅 Date Span", f"{df.time.dt.year.min()}–{df.time.dt.year.max()}")

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🗺️  Seismic Map",
    "📡  Cluster Analysis",
    "⚠️  Risk Zones",
    "📈  Temporal Patterns",
    "🔬  Deep Dive",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 – SEISMIC MAP
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    col_l, col_r = st.columns([3, 1])
    with col_r:
        st.markdown("### 🎛️ Map Options")
        color_by = st.selectbox("Color by", ["Cluster", "Magnitude", "Depth", "Depth Category"])
        show_noise = st.checkbox("Show noise/unclustered", True)
        size_by_mag = st.checkbox("Size by magnitude", True)

    plot_df = df.copy() if show_noise else df[df.cluster >= 0].copy()

    color_col = {
        "Cluster": "cluster_label",
        "Magnitude": "mag",
        "Depth": "depth",
        "Depth Category": "depth_cat",
    }[color_by]

    size_col = df["mag"].apply(lambda m: (m - 4) ** 2 * marker_size_scale) if size_by_mag else marker_size_scale

    if color_by == "Cluster":
        n_c = plot_df["cluster_label"].nunique()
        palette = px.colors.qualitative.Bold + px.colors.qualitative.Vivid
        color_seq = palette[:n_c]
        fig_map = px.scatter_mapbox(
            plot_df, lat="latitude", lon="longitude",
            color="cluster_label",
            size=plot_df["mag"].apply(lambda m: max((m - 4) ** 2 * marker_size_scale, 1)),
            size_max=22,
            hover_data={"mag": True, "depth": True, "place": True, "time": True,
                        "cluster_label": True, "latitude": False, "longitude": False},
            color_discrete_sequence=color_seq,
            mapbox_style=map_style,
            title="🌍 Global Earthquake Cluster Map",
            zoom=1, height=620,
        )
    elif color_by == "Magnitude":
        fig_map = px.scatter_mapbox(
            plot_df, lat="latitude", lon="longitude",
            color="mag", color_continuous_scale="Inferno",
            size=plot_df["mag"].apply(lambda m: max((m - 4) ** 2 * marker_size_scale, 1)),
            size_max=22,
            hover_data={"mag": True, "depth": True, "place": True},
            mapbox_style=map_style,
            title="🌍 Earthquake Map (colored by Magnitude)",
            zoom=1, height=620,
        )
    else:
        fig_map = px.scatter_mapbox(
            plot_df, lat="latitude", lon="longitude",
            color=color_col,
            size=plot_df["mag"].apply(lambda m: max((m - 4) ** 2 * marker_size_scale, 1)),
            size_max=22,
            hover_data={"mag": True, "depth": True, "place": True},
            mapbox_style=map_style,
            title=f"🌍 Earthquake Map (colored by {color_by})",
            zoom=1, height=620,
        )

    fig_map.update_layout(
        paper_bgcolor="#0a0e1a", plot_bgcolor="#0a0e1a",
        font_color="#e2e8f0",
        legend=dict(bgcolor="#1e293b", bordercolor="#334155"),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    with col_l:
        st.plotly_chart(fig_map, use_container_width=True)

    # Depth cross-section
    st.markdown("### 🔍 Latitude vs Depth Cross-Section")
    fig_cross = px.scatter(
        plot_df.sample(min(3000, len(plot_df))), x="latitude", y="depth",
        color="cluster_label" if color_by == "Cluster" else color_col,
        size="mag", size_max=14,
        color_continuous_scale="Inferno" if color_by in ["Magnitude", "Depth"] else None,
        title="Seismic Depth Profile (Latitude Cross-Section)",
        labels={"depth": "Depth (km)", "latitude": "Latitude"},
        height=320,
    )
    fig_cross.update_yaxes(autorange="reversed")
    fig_cross.update_layout(
        paper_bgcolor="#111827", plot_bgcolor="#1e293b",
        font_color="#e2e8f0", margin=dict(t=40, b=20),
    )
    st.plotly_chart(fig_cross, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 – CLUSTER ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 📡 Cluster Summary Table")
    if len(cluster_stats) > 0:
        display_stats = cluster_stats.copy()
        display_stats["Cluster"] = display_stats["cluster"].apply(lambda x: f"Cluster {x+1}")
        display_stats = display_stats[["Cluster", "count", "avg_mag", "max_mag", "avg_depth",
                                        "center_lat", "center_lon", "risk_score"]]
        display_stats.columns = ["Cluster", "Events", "Avg Mag", "Max Mag",
                                   "Avg Depth (km)", "Center Lat", "Center Lon", "Risk Score"]
        display_stats = display_stats.round(3)
        st.dataframe(
            display_stats.style
              .background_gradient(subset=["Risk Score"], cmap="Reds")
              .background_gradient(subset=["Max Mag"], cmap="Oranges")
              .background_gradient(subset=["Events"], cmap="Blues"),
            use_container_width=True, height=350,
        )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 📦 Events per Cluster")
        fig_bar = px.bar(
            cluster_stats.head(15),
            x=cluster_stats.head(15)["cluster"].apply(lambda x: f"C{x+1}"),
            y="count", color="avg_mag",
            color_continuous_scale="Inferno",
            labels={"x": "Cluster", "count": "# Events", "avg_mag": "Avg Mag"},
            height=320,
        )
        fig_bar.update_layout(paper_bgcolor="#111827", plot_bgcolor="#1e293b",
                               font_color="#e2e8f0", showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)

    with c2:
        st.markdown("#### 🎯 Avg Magnitude vs Avg Depth")
        fig_scat = px.scatter(
            cluster_stats,
            x="avg_depth", y="avg_mag",
            size="count", color="risk_score",
            text=cluster_stats["cluster"].apply(lambda x: f"C{x+1}"),
            color_continuous_scale="RdYlGn_r",
            labels={"avg_depth": "Avg Depth (km)", "avg_mag": "Avg Magnitude"},
            height=320,
        )
        fig_scat.update_traces(textposition="top center")
        fig_scat.update_layout(paper_bgcolor="#111827", plot_bgcolor="#1e293b", font_color="#e2e8f0")
        st.plotly_chart(fig_scat, use_container_width=True)

    # PCA Visualization
    st.markdown("#### 🔬 PCA – Cluster Structure in 2D Feature Space")
    pca_df = df[["latitude", "longitude", "depth", "mag", "cluster_label"]].dropna()
    if len(pca_df) > 100:
        scaler = StandardScaler()
        X_pca = scaler.fit_transform(pca_df[["latitude", "longitude", "depth", "mag"]])
        pca = PCA(n_components=2, random_state=42)
        pcs = pca.fit_transform(X_pca)
        pca_df = pca_df.copy()
        pca_df["PC1"] = pcs[:, 0]
        pca_df["PC2"] = pcs[:, 1]
        sample = pca_df.sample(min(4000, len(pca_df)), random_state=42)
        fig_pca = px.scatter(
            sample, x="PC1", y="PC2", color="cluster_label",
            title=f"PCA (Explained variance: {pca.explained_variance_ratio_.sum()*100:.1f}%)",
            height=380,
            color_discrete_sequence=px.colors.qualitative.Bold + px.colors.qualitative.Vivid,
        )
        fig_pca.update_traces(marker=dict(size=4, opacity=0.7))
        fig_pca.update_layout(paper_bgcolor="#111827", plot_bgcolor="#1e293b",
                               font_color="#e2e8f0", legend=dict(bgcolor="#1e293b"))
        st.plotly_chart(fig_pca, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 – RISK ZONES
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### ⚠️ High-Risk Zone Identification")
    st.info("Risk Score = weighted combination of avg magnitude, max magnitude, shallow depth factor, and event density.")

    if len(cluster_stats) > 0:
        top_n = st.slider("Show top N high-risk clusters", 3, min(15, len(cluster_stats)), 5)
        top_risk = cluster_stats.head(top_n)

        cols = st.columns(min(top_n, 5))
        for i, (_, row) in enumerate(top_risk.iterrows()):
            if i < len(cols):
                with cols[i % len(cols)]:
                    risk = row["risk_score"]
                    level = "🔴 CRITICAL" if risk > 8 else "🟠 HIGH" if risk > 6 else "🟡 MODERATE"
                    st.markdown(f"""
                    <div class="hero-card" style="text-align:center">
                        <div style="font-size:1.1rem;font-weight:700;color:#f1f5f9">Cluster {int(row['cluster'])+1}</div>
                        <div style="font-size:0.8rem;color:#94a3b8;margin:4px 0">{level}</div>
                        <div style="font-size:2rem;font-weight:800;background:linear-gradient(135deg,#f97316,#ef4444);-webkit-background-clip:text;-webkit-text-fill-color:transparent">{risk:.2f}</div>
                        <div style="font-size:0.72rem;color:#64748b">Risk Score</div>
                        <hr style="border-color:#334155;margin:8px 0">
                        <div style="font-size:0.8rem;color:#94a3b8">📍 {row['count']:,} events</div>
                        <div style="font-size:0.8rem;color:#94a3b8">⚡ Max M{row['max_mag']:.1f}</div>
                        <div style="font-size:0.8rem;color:#94a3b8">🌊 Avg {row['avg_depth']:.0f} km depth</div>
                    </div>""", unsafe_allow_html=True)

    st.markdown("---")
    col_a, col_b = st.columns([2, 1])
    with col_a:
        st.markdown("#### 🗺️ Risk Heatmap")
        fig_heat = px.density_mapbox(
            df, lat="latitude", lon="longitude", z="mag",
            radius=15, zoom=1, height=500,
            mapbox_style=map_style,
            color_continuous_scale="YlOrRd",
            title="Seismic Energy Density (Heatmap)",
        )
        fig_heat.update_layout(paper_bgcolor="#0a0e1a", font_color="#e2e8f0", margin=dict(t=40))
        st.plotly_chart(fig_heat, use_container_width=True)

    with col_b:
        st.markdown("#### 📊 Risk Score Distribution")
        if len(cluster_stats) > 0:
            fig_risk = px.bar(
                cluster_stats.head(15),
                x=cluster_stats.head(15)["cluster"].apply(lambda x: f"C{x+1}"),
                y="risk_score",
                color="risk_score",
                color_continuous_scale="RdYlGn_r",
                labels={"x": "Cluster", "risk_score": "Risk Score"},
                height=300,
            )
            fig_risk.add_hline(y=cluster_stats["risk_score"].mean(), line_dash="dash",
                                line_color="#38bdf8", annotation_text="Mean Risk")
            fig_risk.update_layout(paper_bgcolor="#111827", plot_bgcolor="#1e293b",
                                    font_color="#e2e8f0", showlegend=False,
                                    margin=dict(t=20))
            st.plotly_chart(fig_risk, use_container_width=True)

        st.markdown("#### 🌊 Depth Category Breakdown")
        dep_counts = df["depth_cat"].value_counts().reset_index()
        dep_counts.columns = ["Depth Category", "Count"]
        fig_dep = px.pie(dep_counts, names="Depth Category", values="Count",
                          color_discrete_sequence=["#f97316", "#3b82f6", "#a855f7"],
                          height=250)
        fig_dep.update_layout(paper_bgcolor="#111827", font_color="#e2e8f0",
                               legend=dict(bgcolor="#1e293b"), margin=dict(t=10))
        st.plotly_chart(fig_dep, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 – TEMPORAL PATTERNS
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### 📈 Temporal Earthquake Patterns")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 📅 Monthly Event Count")
        monthly = df.groupby(df["time"].dt.to_period("M")).size().reset_index()
        monthly.columns = ["month", "count"]
        monthly["month"] = monthly["month"].astype(str)
        fig_monthly = px.area(monthly, x="month", y="count",
                               color_discrete_sequence=["#f97316"],
                               height=280, title="Events per Month")
        fig_monthly.update_layout(paper_bgcolor="#111827", plot_bgcolor="#1e293b",
                                   font_color="#e2e8f0", margin=dict(t=40))
        st.plotly_chart(fig_monthly, use_container_width=True)

    with c2:
        st.markdown("#### ⚡ Magnitude Distribution over Time")
        fig_box = px.box(df, x="year", y="mag", color="year",
                          color_discrete_sequence=px.colors.sequential.Plasma,
                          height=280, title="Magnitude Box by Year")
        fig_box.update_layout(paper_bgcolor="#111827", plot_bgcolor="#1e293b",
                               font_color="#e2e8f0", showlegend=False, margin=dict(t=40))
        st.plotly_chart(fig_box, use_container_width=True)

    # Hourly pattern
    st.markdown("#### 🕐 Hourly Distribution (UTC)")
    hourly = df.groupby("hour").agg(count=("mag", "count"), avg_mag=("mag", "mean")).reset_index()
    fig_hourly = make_subplots(specs=[[{"secondary_y": True}]])
    fig_hourly.add_trace(go.Bar(x=hourly["hour"], y=hourly["count"],
                                 name="Event Count", marker_color="#3b82f6"), secondary_y=False)
    fig_hourly.add_trace(go.Scatter(x=hourly["hour"], y=hourly["avg_mag"],
                                     name="Avg Magnitude", line=dict(color="#f97316", width=2),
                                     mode="lines+markers"), secondary_y=True)
    fig_hourly.update_layout(paper_bgcolor="#111827", plot_bgcolor="#1e293b",
                              font_color="#e2e8f0", height=300,
                              title="Events and Avg Magnitude by Hour (UTC)",
                              legend=dict(bgcolor="#1e293b"))
    fig_hourly.update_xaxes(title_text="Hour (UTC)")
    fig_hourly.update_yaxes(title_text="Event Count", secondary_y=False)
    fig_hourly.update_yaxes(title_text="Avg Magnitude", secondary_y=True)
    st.plotly_chart(fig_hourly, use_container_width=True)

    # Aftershock detection
    st.markdown("#### 🔴 Aftershock Cluster Detection")
    st.caption("Earthquakes within 3 days and 1° lat/lon of a M6+ event are flagged as potential aftershocks.")
    big_quakes = df[df["mag"] >= 6.0].sort_values("time")
    df["is_aftershock"] = False
    for _, bq in big_quakes.iterrows():
        mask = (
            (df["time"] > bq["time"]) &
            (df["time"] <= bq["time"] + pd.Timedelta(days=3)) &
            (abs(df["latitude"] - bq["latitude"]) <= 1) &
            (abs(df["longitude"] - bq["longitude"]) <= 1) &
            (df["mag"] < bq["mag"])
        )
        df.loc[mask, "is_aftershock"] = True

    n_as = df["is_aftershock"].sum()
    col_x, col_y, col_z = st.columns(3)
    col_x.metric("M6+ Mainshocks", len(big_quakes))
    col_y.metric("Probable Aftershocks", int(n_as))
    col_z.metric("Aftershock Rate", f"{n_as/max(len(df),1)*100:.1f}%")

    as_map = px.scatter_mapbox(
        df.sample(min(5000, len(df))), lat="latitude", lon="longitude",
        color="is_aftershock",
        color_discrete_map={True: "#ef4444", False: "#3b82f6"},
        size="mag", size_max=16,
        mapbox_style=map_style, zoom=1, height=420,
        title="🔴 Aftershock Detection Map (red = probable aftershock)",
        hover_data=["mag", "depth", "place"],
    )
    as_map.update_layout(paper_bgcolor="#0a0e1a", font_color="#e2e8f0",
                          legend=dict(bgcolor="#1e293b"), margin=dict(t=40))
    st.plotly_chart(as_map, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 – DEEP DIVE
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("### 🔬 Deep Dive Analytics")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 📊 Magnitude Frequency (Gutenberg-Richter)")
        mag_bins = np.arange(df["mag"].min(), df["mag"].max() + 0.5, 0.5)
        mag_hist = df["mag"].value_counts(bins=mag_bins, sort=False).sort_index()
        cumulative = mag_hist.sum() - mag_hist.cumsum()
        fig_gr = go.Figure()
        fig_gr.add_trace(go.Bar(x=[str(b) for b in mag_hist.index], y=mag_hist.values,
                                 name="Frequency", marker_color="#3b82f6"))
        fig_gr.add_trace(go.Scatter(x=[str(b) for b in cumulative.index], y=cumulative.values,
                                     name="Cumulative (≥M)", mode="lines+markers",
                                     line=dict(color="#f97316", width=2)))
        fig_gr.update_layout(paper_bgcolor="#111827", plot_bgcolor="#1e293b", font_color="#e2e8f0",
                              title="Gutenberg-Richter Magnitude-Frequency", height=300,
                              legend=dict(bgcolor="#1e293b"))
        st.plotly_chart(fig_gr, use_container_width=True)

    with c2:
        st.markdown("#### 🌊 3D Earthquake Cloud")
        sample3d = df.sample(min(2000, len(df)), random_state=42)
        fig_3d = px.scatter_3d(
            sample3d, x="longitude", y="latitude", z="depth",
            color="mag", size="mag", size_max=10,
            color_continuous_scale="Inferno",
            labels={"depth": "Depth (km)"},
            title="3D Spatial Distribution",
            height=360,
        )
        fig_3d.update_layout(paper_bgcolor="#111827", font_color="#e2e8f0",
                              scene=dict(bgcolor="#111827",
                                         xaxis=dict(backgroundcolor="#1e293b"),
                                         yaxis=dict(backgroundcolor="#1e293b"),
                                         zaxis=dict(backgroundcolor="#1e293b", autorange="reversed")))
        st.plotly_chart(fig_3d, use_container_width=True)

    # Correlation heatmap
    st.markdown("#### 🔗 Feature Correlation Matrix")
    num_cols = ["mag", "depth", "nst", "gap", "dmin", "rms", "horizontalError", "depthError"]
    corr_df = df[num_cols].dropna().corr()
    fig_corr = px.imshow(corr_df, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                          title="Correlation between Seismic Features", height=380)
    fig_corr.update_layout(paper_bgcolor="#111827", font_color="#e2e8f0")
    st.plotly_chart(fig_corr, use_container_width=True)

    # Magnitude by net/source
    st.markdown("#### 🏢 Seismic Network Analysis")
    net_stats = df.groupby("net").agg(count=("mag", "count"), avg_mag=("mag", "mean"),
                                       max_mag=("mag", "max")).reset_index()
    net_stats = net_stats.sort_values("count", ascending=False).head(10)
    fig_net = px.bar(net_stats, x="net", y="count", color="avg_mag",
                      color_continuous_scale="Viridis",
                      title="Events per Seismic Network (top 10)",
                      labels={"net": "Network", "count": "Events", "avg_mag": "Avg Mag"},
                      height=280)
    fig_net.update_layout(paper_bgcolor="#111827", plot_bgcolor="#1e293b",
                           font_color="#e2e8f0")
    st.plotly_chart(fig_net, use_container_width=True)

    # Raw data explorer
    st.markdown("---")
    with st.expander("📋 Raw Data Explorer", expanded=False):
        cluster_filter = st.multiselect("Filter by cluster", options=df["cluster_label"].unique(),
                                         default=list(df["cluster_label"].unique())[:3])
        show_cols = ["time", "place", "mag", "depth", "latitude", "longitude",
                      "cluster_label", "is_aftershock", "net", "magType"]
        filtered = df[df["cluster_label"].isin(cluster_filter)][show_cols]
        st.dataframe(filtered.head(500), use_container_width=True, height=300)
        st.download_button("⬇️ Download Filtered CSV", filtered.to_csv(index=False),
                            "earthquake_clusters.csv", "text/csv")

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<p style="text-align:center;color:#475569;font-size:0.8rem">'
    '🌍 Earthquake Pattern Discovery · Built with Streamlit, Plotly, scikit-learn · '
    'Data: USGS Earthquake Catalog'
    '</p>',
    unsafe_allow_html=True
)