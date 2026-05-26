"""
app.py — Intern Performance Prediction · Streamlit App
Run:  streamlit run app.py

Architecture:
  - Pre-trained model (pretrained_model.pkl) loads instantly — no wait on upload
  - User CSV is used only for scoring/flagging, NOT retraining
  - Pages: Overview · Risk Flagging · Predict Intern · Export
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle, io, warnings
warnings.filterwarnings("ignore")

from pipeline import (
    engineer_features, flag_interns, predict_single,
    ALL_FEATURES, FEATURE_COLS, ENGINEERED_COLS, assign_risk,
)

# ── Page config ─────────────────────────────────────────────
st.set_page_config(
    page_title="Intern Performance ML",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

COLORS = {
    "Excelling": "#1D9E75",
    "On Track":  "#EF9F27",
    "At Risk":   "#E24B4A",
    "Critical":  "#C0392B",
}
TIER_BG = {
    "Excelling": "#EAF7F1",
    "On Track":  "#FEF9EE",
    "At Risk":   "#FEF0EF",
    "Critical":  "#FADADD",
}
BG = "#F5F7FA"
FONT = dict(family="Inter, Segoe UI, sans-serif", color="#1A202C")

def chart_layout(title="", height=320, **kwargs):
    """Shared Plotly layout — bold dark title, crisp axes, white card."""
    base = dict(
        paper_bgcolor="white", plot_bgcolor="white",
        height=height, font=FONT,
        margin=dict(t=56 if title else 24, b=32, l=12, r=12),
        xaxis=dict(showgrid=True, gridcolor="#EDF2F7", gridwidth=1,
                   linecolor="#CBD5E0",
                   tickfont=dict(size=13, color="#1A202C", family="Inter, Segoe UI, sans-serif"),
                   title_font=dict(size=13, color="#2D3748", family="Inter, Segoe UI, sans-serif")),
        yaxis=dict(showgrid=True, gridcolor="#EDF2F7", gridwidth=1,
                   linecolor="#CBD5E0",
                   tickfont=dict(size=13, color="#1A202C", family="Inter, Segoe UI, sans-serif"),
                   title_font=dict(size=13, color="#2D3748", family="Inter, Segoe UI, sans-serif")),
    )
    if title:
        base["title"] = dict(
            text=f"<b>{title}</b>",
            font=dict(size=15, color="#0F1B2D",
                      family="Inter, Segoe UI, sans-serif"),
            x=0.01, xanchor="left",
        )
    base.update(kwargs)
    return base

# ── CSS ─────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stAppViewContainer"]  { background: #F5F7FA; }
[data-testid="stSidebar"]           { background: #0F1B2D; padding-top: 1.5rem; }
[data-testid="stSidebar"] *         { color: #CBD5E0 !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3        { color: #FFFFFF !important; }
[data-testid="stSidebar"] .stRadio label span { color: #CBD5E0 !important; }
[data-testid="stSidebar"] hr        { border-color: #2D3748; }

/* Cards */
.kpi-card {
    background: white; border-radius: 14px; padding: 1.1rem 1.3rem;
    border: 1px solid #E2E8F0; text-align: center; height: 100%;
}
.kpi-val  { font-size: 2rem; font-weight: 700; line-height: 1.1; }
.kpi-lbl  { font-size: 12px; color: #718096; margin-top: 4px; font-weight: 500; }

/* Section headings */
.sec-head {
    font-size: 14px; font-weight: 700; letter-spacing: .03em;
    text-transform: uppercase; color: #1A202C; margin-bottom: .6rem;
    border-left: 3px solid #3B82F6; padding-left: 8px;
}

/* Tier badge */
.tier-badge {
    display: inline-block; padding: 4px 14px; border-radius: 20px;
    font-size: 12px; font-weight: 600;
}

/* Prediction score circle */
.score-ring {
    border-radius: 50%; width: 130px; height: 130px;
    display: flex; flex-direction: column; align-items: center;
    justify-content: center; margin: 0 auto 1rem;
    border: 6px solid; font-size: 36px; font-weight: 800; line-height: 1;
}

/* Force slider labels to be fully visible */
div[data-testid="stSlider"] label p {
    font-size: 13px !important;
    font-weight: 600 !important;
    color: #1A202C !important;
    opacity: 1 !important;
}
div[data-testid="stSlider"] [data-testid="stMarkdownContainer"] p {
    font-size: 13px !important;
    color: #1A202C !important;
}
/* Slider value display */
div[data-testid="stSlider"] [data-testid="stTickBarMin"],
div[data-testid="stSlider"] [data-testid="stTickBarMax"] {
    font-size: 11px !important;
    color: #4A5568 !important;
}

div[data-testid="stMetric"] {
    background: white; border-radius: 12px; padding: .8rem 1rem;
    border: 1px solid #E2E8F0;
}
</style>
""", unsafe_allow_html=True)


# ── Load pre-trained model (instant) ────────────────────────
@st.cache_resource
def load_pretrained():
    with open("pretrained_model.pkl", "rb") as f:
        return pickle.load(f)

try:
    bundle     = load_pretrained()
    best_model = bundle["best_model"]
    best_name  = bundle["best_name"]
    results    = bundle["results"]
    fi         = bundle["fi"]
    X_test     = bundle["X_test"]
    y_test     = bundle["y_test"]
except FileNotFoundError:
    st.error("❌ `pretrained_model.pkl` not found. Run `python pretrain.py` first.")
    st.stop()


# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎓 Intern ML")
    st.markdown("<p style='font-size:12px;color:#718096'>Powered by XGBoost + Random Forest</p>", unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("### 📂 Upload Intern Dataset")
    uploaded = st.file_uploader("CSV file", type=["csv"],
                                 help="Upload your intern CSV — predictions run instantly on the pre-trained model.")

    st.markdown("---")
    page = st.radio("Navigate", [
        "📊  Overview",
        "🚨  Risk Flagging",
        "🔮  Predict Intern",
        "📥  Export",
    ])
    st.markdown("---")

    r2  = results[best_name]["r2"]
    mae = results[best_name]["mae"]
    st.markdown(f"""
    <div style='font-size:12px;color:#718096;line-height:1.9'>
    <b style='color:#A0AEC0'>Best Model</b><br>{best_name}<br>
    <b style='color:#A0AEC0'>R²</b><br>{r2}<br>
    <b style='color:#A0AEC0'>MAE</b><br>{mae} pts<br>
    <b style='color:#A0AEC0'>Trained on</b><br>6,000 interns
    </div>
    """, unsafe_allow_html=True)


# ── Landing if no file uploaded ──────────────────────────────
if uploaded is None:
    st.markdown("""
    <div style='max-width:640px;margin:5rem auto;text-align:center'>
        <div style='font-size:72px'>🎓</div>
        <h1 style='font-size:2rem;margin:.8rem 0 .4rem'>Intern Performance Predictor</h1>
        <p style='color:#718096;font-size:15px;line-height:1.7'>
            Upload your intern CSV in the sidebar to instantly score every intern,
            flag at-risk cases, and drill into individual predictions —
            no waiting, model is pre-trained and ready.
        </p>
        <div style='background:white;border:1px solid #E2E8F0;border-radius:16px;
                    padding:1.5rem 2rem;margin-top:2rem;text-align:left'>
            <div style='font-size:13px;font-weight:600;color:#2D3748;margin-bottom:.5rem'>
                Required CSV columns
            </div>
            <div style='font-size:12px;color:#718096;line-height:2;font-family:monospace'>
                intern_id · intern_name · task_completion_rate<br>
                avg_task_completion_time · attendance_percentage<br>
                feedback_rating · deadlines_missed · github_commits<br>
                communication_score · learning_speed · meeting_participation<br>
                work_hours_per_day · stress_level · final_performance_score<br>
                performance_category
            </div>
        </div>
        <div style='margin-top:1.5rem;display:flex;gap:12px;justify-content:center;flex-wrap:wrap'>
            <div style='background:#EAF7F1;color:#1D9E75;padding:6px 16px;border-radius:20px;font-size:12px;font-weight:600'>⚡ Instant results</div>
            <div style='background:#EBF4FF;color:#3B82F6;padding:6px 16px;border-radius:20px;font-size:12px;font-weight:600'>🔍 SHAP explanations</div>
            <div style='background:#FEF9EE;color:#D97706;padding:6px 16px;border-radius:20px;font-size:12px;font-weight:600'>🚨 Risk flagging</div>
            <div style='background:#FEF0EF;color:#E24B4A;padding:6px 16px;border-radius:20px;font-size:12px;font-weight:600'>📥 CSV export</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ── Load uploaded CSV and score it ──────────────────────────
@st.cache_data(show_spinner=False)
def load_and_score(file_bytes):
    df      = pd.read_csv(io.BytesIO(file_bytes))
    flagged = flag_interns(df, best_model, best_name)
    df_eng  = engineer_features(df)
    return df, flagged, df_eng

file_bytes = uploaded.read()
with st.spinner("Scoring interns…"):
    df, flagged, df_eng = load_and_score(file_bytes)

tier_counts = flagged["risk_tier"].value_counts()
n = len(df)


# ════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ════════════════════════════════════════════════════════════
if page == "📊  Overview":
    st.markdown(f"## 📊 Overview  <span style='font-size:14px;color:#718096;font-weight:400'>— {uploaded.name} · {n:,} interns</span>", unsafe_allow_html=True)

    # ── KPI row
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    avg  = df["final_performance_score"].mean()
    hi   = df["final_performance_score"].max()
    lo   = df["final_performance_score"].min()

    def kpi(col, val, label, color="#2D3748"):
        col.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-val' style='color:{color}'>{val}</div>
            <div class='kpi-lbl'>{label}</div>
        </div>""", unsafe_allow_html=True)

    kpi(c1, f"{n:,}",                     "Total Interns")
    kpi(c2, f"{avg:.1f}",                 "Avg Score",          "#3B82F6")
    kpi(c3, tier_counts.get("Excelling",0),"🟢 Excelling",      COLORS["Excelling"])
    kpi(c4, tier_counts.get("On Track",0), "🟡 On Track",       COLORS["On Track"])
    kpi(c5, tier_counts.get("At Risk",0),  "🟠 At Risk",        COLORS["At Risk"])
    kpi(c6, tier_counts.get("Critical",0), "🔴 Critical",       COLORS["Critical"])

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Charts row 1
    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(df, x="final_performance_score", nbins=40,
                           color_discrete_sequence=["#3B82F6"],
                           labels={"final_performance_score":"Score"})
        for thr, lbl, col_ in [(35,"Critical","#C0392B"),(45,"At Risk","#E24B4A"),(65,"Excelling","#1D9E75")]:
            fig.add_vline(
                x=thr, line_dash="dash", line_color=col_, line_width=2,
                annotation_text=f"  <b>{lbl}</b>  ",
                annotation_font=dict(size=13, color="white",
                                     family="Inter, Segoe UI, sans-serif"),
                annotation_bgcolor=col_,
                annotation_bordercolor=col_,
                annotation_borderwidth=1,
                annotation_borderpad=4,
                annotation_position="top right",
            )
        fig.update_layout(**chart_layout("Score Distribution", height=320, showlegend=False))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        td = flagged["risk_tier"].value_counts().reset_index()
        td.columns = ["Tier","Count"]
        fig2 = px.pie(td, names="Tier", values="Count",
                      color="Tier", color_discrete_map=COLORS, hole=0.48)
        fig2.update_traces(
            textposition="outside",
            textinfo="percent+label",
            textfont=dict(size=14, color="#0F1B2D",
                          family="Inter, Segoe UI, sans-serif"),
            pull=[0.04, 0.04, 0.04, 0.04],
            marker=dict(line=dict(color="white", width=2)),
        )
        fig2.update_layout(**chart_layout("Risk Tier Split", height=320, showlegend=False))
        st.plotly_chart(fig2, use_container_width=True)

    # ── Charts row 2
    col3, col4 = st.columns(2)

    with col3:
        fi_df = fi.head(12).reset_index()
        fi_df.columns = ["Feature","Importance"]
        fi_df["Feature"] = fi_df["Feature"].str.replace("_"," ").str.title()
        fig3 = px.bar(fi_df, x="Importance", y="Feature", orientation="h",
                      color="Importance", color_continuous_scale=["#93C5FD","#1D4ED8"],
                      text=fi_df["Importance"].map(lambda v: f"{v:.3f}"))
        fig3.update_traces(textposition="outside",
                           textfont=dict(size=12, color="#1A202C"))
        fig3.update_layout(**chart_layout("Feature Importance (Pre-trained Model)", height=380,
                                          coloraxis_showscale=False,
                                          yaxis=dict(autorange="reversed",
                                                     tickfont=dict(size=13, color="#0F1B2D",
                                                                   family="Inter, Segoe UI, sans-serif"))))
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        corr = (df[FEATURE_COLS + ["final_performance_score"]]
                .corr()["final_performance_score"]
                .drop("final_performance_score")
                .sort_values())
        corr_df = corr.reset_index()
        corr_df.columns = ["Feature","r"]
        corr_df["Feature"] = corr_df["Feature"].str.replace("_"," ").str.title()
        fig4 = px.bar(corr_df, x="r", y="Feature", orientation="h",
                      color="r", color_continuous_scale=["#E24B4A","#eee","#1D9E75"],
                      color_continuous_midpoint=0,
                      text=corr_df["r"].map(lambda v: f"{v:+.2f}"))
        fig4.update_traces(textposition="outside",
                           textfont=dict(size=12, color="#1A202C"))
        fig4.update_layout(**chart_layout("Correlations with Performance Score", height=380,
                                          coloraxis_showscale=False,
                                          yaxis=dict(tickfont=dict(size=13, color="#0F1B2D",
                                                                   family="Inter, Segoe UI, sans-serif"))))
        st.plotly_chart(fig4, use_container_width=True)

    # ── Engineered features
    sel = st.selectbox("Select composite feature", ENGINEERED_COLS,
                       format_func=lambda x: x.replace("_"," ").title())
    cat_colors = {"Excellent":"#1D9E75","Average":"#EF9F27","Needs Improvement":"#E24B4A"}
    fig5 = go.Figure()
    for cat, color in cat_colors.items():
        sub = df_eng[df_eng["performance_category"]==cat]
        if sub.empty: continue
        fig5.add_trace(go.Scatter(x=sub[sel], y=sub["final_performance_score"],
                                  mode="markers", name=cat, opacity=0.4,
                                  marker=dict(color=color, size=4)))
        x_,y_ = sub[sel].values, sub["final_performance_score"].values
        mask = ~(np.isnan(x_)|np.isnan(y_))
        if mask.sum()>1:
            m,b = np.polyfit(x_[mask],y_[mask],1)
            xl = np.linspace(x_[mask].min(),x_[mask].max(),100)
            fig5.add_trace(go.Scatter(x=xl,y=m*xl+b,mode="lines",
                                      showlegend=False,
                                      line=dict(color=color,width=2,dash="dot")))
    fig5.update_layout(**chart_layout(
        f"{sel.replace('_',' ').title()} vs Performance Score", height=320,
        xaxis=dict(title=sel.replace("_"," ").title(),
                   tickfont=dict(size=11,color="#4A5568"),
                   title_font=dict(size=12,color="#2D3748"),
                   showgrid=True, gridcolor="#EDF2F7"),
        yaxis=dict(title="Performance Score",
                   tickfont=dict(size=11,color="#4A5568"),
                   title_font=dict(size=12,color="#2D3748"),
                   showgrid=True, gridcolor="#EDF2F7"),
        legend=dict(orientation="h", y=-0.22,
                    font=dict(size=12, color="#1A202C")),
    ))
    st.plotly_chart(fig5, use_container_width=True, key="scatter_eng_feature")


# ════════════════════════════════════════════════════════════
# PAGE 2 — RISK FLAGGING
# ════════════════════════════════════════════════════════════
elif page == "🚨  Risk Flagging":
    st.markdown("## 🚨 Risk Flagging", unsafe_allow_html=True)

    # KPI strip
    c1,c2,c3,c4 = st.columns(4)
    for col, tier, emoji in [
        (c1,"Critical","🔴"),(c2,"At Risk","🟠"),
        (c3,"On Track","🟡"),(c4,"Excelling","🟢")
    ]:
        cnt = tier_counts.get(tier,0)
        pct = cnt/n*100
        col.markdown(f"""
        <div class='kpi-card' style='border-top:4px solid {COLORS[tier]}'>
            <div class='kpi-val' style='color:{COLORS[tier]}'>{cnt}</div>
            <div class='kpi-lbl'>{emoji} {tier} &nbsp;·&nbsp; {pct:.1f}%</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Filters row
    fcol1, fcol2, fcol3 = st.columns([2,2,2])
    tier_filter = fcol1.multiselect("Risk tier",
        ["Critical","At Risk","On Track","Excelling"],
        default=["Critical","At Risk"])
    search = fcol2.text_input("Search intern name", placeholder="Type a name…")
    min_score, max_score = fcol3.slider("Predicted score range", 0, 100, (0,100))

    show = flagged[flagged["risk_tier"].isin(tier_filter)].copy()
    if search:
        show = show[show["intern_name"].str.contains(search, case=False, na=False)]
    show = show[(show["predicted_score"]>=min_score)&(show["predicted_score"]<=max_score)]

    # ── Color the risk_tier column
    def style_tier(val):
        c = {"Critical":"#FADADD","At Risk":"#FFE5CC",
             "On Track":"#FFFACD","Excelling":"#D4EDDA"}.get(val,"")
        return f"background-color:{c};font-weight:600"

    cols_show = ["intern_name","predicted_score","actual_score","risk_tier",
                 "deadlines_missed","stress_level","attendance_percentage",
                 "feedback_rating","alert_reasons"]

    display = show[cols_show].rename(columns={
        "intern_name":"Name","predicted_score":"Pred Score",
        "actual_score":"Actual","risk_tier":"Tier",
        "deadlines_missed":"Missed","stress_level":"Stress",
        "attendance_percentage":"Attendance","feedback_rating":"Feedback",
        "alert_reasons":"Alerts",
    })

    st.dataframe(
        display.style.map(style_tier, subset=["Tier"]),
        use_container_width=True, height=420,
    )
    st.caption(f"Showing **{len(show):,}** of {n:,} interns")

    # ── Alert breakdown + score box side by side
    a1, a2 = st.columns(2)
    with a1:
        st.markdown("<div class='sec-head'>Alert Reason Frequency</div>", unsafe_allow_html=True)
        alerts = show["alert_reasons"].str.split(" | ").explode()
        alerts = alerts[alerts!="—"].value_counts().reset_index()
        alerts.columns=["Reason","Count"]
        if not alerts.empty:
            fig = px.bar(alerts, x="Count", y="Reason", orientation="h",
                         color_discrete_sequence=["#E24B4A"])
            fig.update_layout(**chart_layout("Alert Reason Frequency",
                                             height=max(240, len(alerts)*50),
                                             yaxis=dict(autorange="reversed",
                                                        tickfont=dict(size=12,color="#1A202C"))))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No alerts in selected group.")

    with a2:
        st.markdown("<div class='sec-head'>Score by Tier</div>", unsafe_allow_html=True)
        fig2 = px.box(flagged, x="risk_tier", y="predicted_score",
                      color="risk_tier", color_discrete_map=COLORS,
                      category_orders={"risk_tier":["Critical","At Risk","On Track","Excelling"]})
        fig2.update_layout(**chart_layout("Predicted Score by Tier", height=320,
                                          showlegend=False,
                                          xaxis=dict(tickfont=dict(size=12,color="#1A202C"))))
        st.plotly_chart(fig2, use_container_width=True)


# ════════════════════════════════════════════════════════════
# PAGE 3 — PREDICT INTERN
# ════════════════════════════════════════════════════════════
elif page == "🔮  Predict Intern":
    st.markdown("## 🔮 Predict a New Intern")
    st.caption(f"Model: **{best_name}** · Scores are generated instantly from the pre-trained model.")

    left, right = st.columns([1.2, 1])

    with left:
        with st.form("pred_form"):
            st.markdown("#### Intern Details")
            name = st.text_input("Intern name", "New Intern")

            st.markdown("""
            <div style='background:#EBF4FF;border-radius:8px;padding:.5rem .75rem;
                        margin:.5rem 0;font-size:13px;font-weight:700;color:#1D4ED8'>
                📋 Task Metrics
            </div>""", unsafe_allow_html=True)
            r1c1, r1c2 = st.columns(2)
            tcr  = r1c1.slider("Task completion rate (%)",     40, 100, 75,  key="s_tcr")
            act  = r1c1.slider("Avg completion time (hrs)",    1.0, 15.0, 7.0, 0.5, key="s_act")
            dm   = r1c1.slider("Deadlines missed",             0, 10, 3,     key="s_dm")
            gc   = r1c2.slider("GitHub commits",               0, 300, 120,  key="s_gc")
            fr   = r1c2.slider("Feedback rating (1–5)",        1.0, 5.0, 3.0, 0.1, key="s_fr")
            att  = r1c2.slider("Attendance (%)",               50, 100, 80,  key="s_att")

            st.markdown("""
            <div style='background:#F0FDF4;border-radius:8px;padding:.5rem .75rem;
                        margin:.5rem 0;font-size:13px;font-weight:700;color:#166534'>
                💬 Engagement & Wellbeing
            </div>""", unsafe_allow_html=True)
            r2c1, r2c2 = st.columns(2)
            cs   = r2c1.slider("Communication score (1–10)",   1, 10, 6,    key="s_cs")
            ls   = r2c1.slider("Learning speed (1–10)",        1, 10, 6,    key="s_ls")
            mp   = r2c2.slider("Meeting participation (1–10)", 1, 10, 6,    key="s_mp")
            wh   = r2c2.slider("Work hours per day",           2.0, 12.0, 7.0, 0.5, key="s_wh")
            sl   = r2c2.slider("Stress level (1–10)",          1, 10, 5,    key="s_sl")
            submitted = st.form_submit_button("🔮 Predict Performance", use_container_width=True)

    with right:
        if submitted:
            res   = predict_single(best_model, {
                "task_completion_rate":tcr,"avg_task_completion_time":act,
                "attendance_percentage":att,"feedback_rating":fr,
                "deadlines_missed":dm,"github_commits":gc,
                "communication_score":cs,"learning_speed":ls,
                "meeting_participation":mp,"work_hours_per_day":wh,
                "stress_level":sl,
            })
            score = res["score"]
            tier  = res["risk_tier"]
            eng   = res["engineered"]
            tc    = COLORS[tier]
            tbg   = TIER_BG[tier]

            # Score circle
            st.markdown(f"""
            <div style='background:white;border-radius:18px;border:1px solid #E2E8F0;
                        padding:1.8rem 1.5rem;text-align:center'>
                <div style='font-size:13px;font-weight:600;color:#718096;margin-bottom:.8rem'>
                    {name}
                </div>
                <div style='width:130px;height:130px;border-radius:50%;border:7px solid {tc};
                            display:flex;flex-direction:column;align-items:center;
                            justify-content:center;margin:0 auto .8rem;background:{tbg}'>
                    <div style='font-size:38px;font-weight:800;color:{tc};line-height:1'>{score}</div>
                    <div style='font-size:11px;color:#718096'>/ 100</div>
                </div>
                <span style='background:{tbg};color:{tc};padding:5px 18px;
                             border-radius:20px;font-weight:700;font-size:13px'>{tier}</span>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Composite scores
            st.markdown("<div class='sec-head'>Composite Scores</div>", unsafe_allow_html=True)
            for lbl, val in eng.items():
                bar_c = COLORS["At Risk"] if lbl=="burnout_risk" else COLORS["Excelling"]
                disp  = lbl.replace("_"," ").title()
                st.markdown(f"""
                <div style='margin-bottom:10px'>
                    <div style='display:flex;justify-content:space-between;
                                font-size:12px;margin-bottom:3px'>
                        <span style='color:#4A5568'>{disp}</span>
                        <b>{val}</b>
                    </div>
                    <div style='background:#E2E8F0;border-radius:6px;height:9px'>
                        <div style='width:{min(val,100)}%;background:{bar_c};
                                    height:100%;border-radius:6px'></div>
                    </div>
                </div>""", unsafe_allow_html=True)

            # Recommendation
            rec = {
                "Excelling":"🌟 High performer — assign stretch projects and leadership opportunities.",
                "On Track": "📈 Progressing — monitor deadlines, increase engagement.",
                "At Risk":  "⚠️ Needs support — schedule bi-weekly check-ins, reduce stress load.",
                "Critical": "🚨 Urgent — immediate 1-on-1 with manager required.",
            }
            st.info(rec[tier])
        else:
            st.markdown("""
            <div style='background:white;border-radius:18px;border:1px dashed #CBD5E0;
                        padding:3rem;text-align:center;color:#A0AEC0'>
                <div style='font-size:40px'>🔮</div>
                <div style='margin-top:.5rem;font-size:14px'>
                    Adjust the sliders and click <b>Predict</b>
                </div>
            </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# PAGE 4 — EXPORT
# ════════════════════════════════════════════════════════════
elif page == "📥  Export":
    st.markdown("## 📥 Export Results")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='sec-head'>Download Reports</div>", unsafe_allow_html=True)

        def dl_btn(label, data_df, fname):
            csv = data_df.to_csv(index=False).encode("utf-8")
            st.download_button(label, data=csv, file_name=fname,
                               mime="text/csv", use_container_width=True)

        dl_btn("⬇️ Full risk report (all interns)",        flagged,                                  "intern_risk_report.csv")
        dl_btn("⬇️ Critical interns only",                 flagged[flagged["risk_tier"]=="Critical"], "critical_interns.csv")
        dl_btn("⬇️ At-risk interns (Critical + At Risk)",  flagged[flagged["risk_tier"].isin(["Critical","At Risk"])], "at_risk_interns.csv")
        dl_btn("⬇️ Excelling interns",                     flagged[flagged["risk_tier"]=="Excelling"],"excelling_interns.csv")

    with col2:
        st.markdown("<div class='sec-head'>Model Summary</div>", unsafe_allow_html=True)
        m = results[best_name]
        st.json({
            "best_model":    best_name,
            "r2":            m["r2"],
            "mae":           m["mae"],
            "rmse":          m["rmse"],
            "cv_r2":         m["cv_r2"],
            "features_used": ALL_FEATURES,
            "trained_on":    "6,000 interns",
        })

        st.markdown("<div class='sec-head' style='margin-top:1rem'>All Model Results</div>", unsafe_allow_html=True)
        rows = [{"Model":k,"MAE":v["mae"],"RMSE":v["rmse"],
                 "R²":v["r2"],"CV R²":v.get("cv_r2","—")}
                for k,v in results.items()]
        st.dataframe(pd.DataFrame(rows).set_index("Model"), use_container_width=True)