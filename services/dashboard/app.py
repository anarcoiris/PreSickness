"""
Dashboard Clínico - EM Predictor
Interfaz para neurólogos y personal clínico.

Responsable: Agent Frontend (Droid)

Funcionalidades:
- Vista general de pacientes y alertas
- Detalle de riesgo por paciente
- Timeline de predicciones
- Gestión de alertas
"""
import os
from datetime import datetime, timedelta
from typing import Optional

import httpx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="EM Predictor - Dashboard Clínico",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# URLs de servicios
API_URL = os.getenv("API_URL", "http://localhost:8000")
INFERENCE_URL = os.getenv("INFERENCE_URL", "http://localhost:8001")
ALERT_URL = os.getenv("ALERT_URL", "http://localhost:8002")
DB_DSN = os.getenv("DB_DSN", "postgresql://emuser:changeme@localhost/empredictor")

# Umbrales
THRESHOLD_WARNING = 0.35
THRESHOLD_CRITICAL = 0.55


# ══════════════════════════════════════════════════════════════════════════════
# ESTILOS
# ══════════════════════════════════════════════════════════════════════════════

st.markdown(
    """
<style>
    /* Tema oscuro médico */
    .stApp {
        background: linear-gradient(135deg, #0a1628 0%, #1a2744 100%);
    }
    
    /* Cards */
    .metric-card {
        background: rgba(26, 39, 68, 0.8);
        border: 1px solid rgba(99, 179, 237, 0.3);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 0.5rem 0;
    }
    
    .metric-card.critical {
        border-color: #e53e3e;
        box-shadow: 0 0 20px rgba(229, 62, 62, 0.2);
    }
    
    .metric-card.warning {
        border-color: #dd6b20;
        box-shadow: 0 0 20px rgba(221, 107, 32, 0.2);
    }
    
    .metric-card.ok {
        border-color: #38a169;
    }
    
    /* Títulos */
    h1, h2, h3 {
        color: #e2e8f0 !important;
    }
    
    /* Alertas */
    .alert-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.875rem;
        font-weight: 600;
    }
    
    .alert-critical {
        background: #e53e3e;
        color: white;
    }
    
    .alert-warning {
        background: #dd6b20;
        color: white;
    }
    
    .alert-info {
        background: #3182ce;
        color: white;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: rgba(10, 22, 40, 0.95);
    }
    
    /* Ocultar footer streamlit */
    footer {visibility: hidden;}
</style>
""",
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════════════════════════════════════
# FUNCIONES DE DATOS
# ══════════════════════════════════════════════════════════════════════════════


@st.cache_data(ttl=60)
def fetch_alerts(status: Optional[str] = None, limit: int = 50) -> pd.DataFrame:
    """Obtiene alertas del servicio."""
    try:
        params = {"limit": limit}
        if status:
            params["status"] = status
        response = httpx.get(f"{ALERT_URL}/v1/alerts", params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        if not data:
            return pd.DataFrame()
        return pd.DataFrame(data)
    except Exception as e:
        st.error(f"Error al obtener alertas: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=60)
def fetch_alert_stats() -> dict:
    """Obtiene estadísticas de alertas."""
    try:
        response = httpx.get(f"{ALERT_URL}/v1/stats", timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception:
        return {
            "total_pending": 0,
            "total_acknowledged": 0,
            "total_resolved": 0,
            "by_level": {"info": 0, "warning": 0, "critical": 0},
            "avg_response_time_hours": None,
        }


@st.cache_data(ttl=300)
def fetch_patients_summary() -> pd.DataFrame:
    """Obtiene resumen de pacientes (mock para prototipo)."""
    # En producción: consultar DB directamente
    # Para prototipo: datos sintéticos
    import numpy as np

    np.random.seed(42)
    n_patients = 15

    data = {
        "user_id_hash": [f"patient_{i:03d}_{'x'*56}" for i in range(n_patients)],
        "display_id": [f"P-{i+1:03d}" for i in range(n_patients)],
        "last_prediction": pd.date_range(
            end=datetime.now(), periods=n_patients, freq="H"
        ),
        "risk_score": np.random.beta(2, 5, n_patients),
        "trend_7d": np.random.uniform(-0.1, 0.1, n_patients),
        "days_since_last_brote": np.random.randint(30, 365, n_patients),
        "adherence_pct": np.random.uniform(0.6, 1.0, n_patients),
    }

    df = pd.DataFrame(data)
    df["risk_level"] = df["risk_score"].apply(
        lambda x: "critical" if x >= THRESHOLD_CRITICAL else ("warning" if x >= THRESHOLD_WARNING else "ok")
    )
    return df


def get_risk_color(score: float) -> str:
    """Devuelve color según nivel de riesgo."""
    if score >= THRESHOLD_CRITICAL:
        return "#e53e3e"
    elif score >= THRESHOLD_WARNING:
        return "#dd6b20"
    else:
        return "#38a169"


def get_risk_label(score: float) -> str:
    """Devuelve etiqueta según nivel de riesgo."""
    if score >= THRESHOLD_CRITICAL:
        return "🔴 Crítico"
    elif score >= THRESHOLD_WARNING:
        return "🟠 Moderado"
    else:
        return "🟢 Bajo"


# ══════════════════════════════════════════════════════════════════════════════
# COMPONENTES UI
# ══════════════════════════════════════════════════════════════════════════════


def render_metric_card(title: str, value: str, subtitle: str = "", level: str = "ok"):
    """Renderiza una tarjeta de métrica."""
    st.markdown(
        f"""
        <div class="metric-card {level}">
            <div style="color: #a0aec0; font-size: 0.875rem; margin-bottom: 0.5rem;">{title}</div>
            <div style="color: #e2e8f0; font-size: 2rem; font-weight: 700;">{value}</div>
            <div style="color: #718096; font-size: 0.75rem; margin-top: 0.25rem;">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_patient_row(patient: pd.Series):
    """Renderiza una fila de paciente."""
    col1, col2, col3, col4, col5 = st.columns([2, 2, 2, 2, 2])

    with col1:
        st.markdown(f"**{patient['display_id']}**")

    with col2:
        risk_pct = int(patient["risk_score"] * 100)
        st.markdown(
            f"<span style='color: {get_risk_color(patient['risk_score'])}'>"
            f"{get_risk_label(patient['risk_score'])} ({risk_pct}%)</span>",
            unsafe_allow_html=True,
        )

    with col3:
        trend = patient["trend_7d"]
        arrow = "↑" if trend > 0 else "↓" if trend < 0 else "→"
        color = "#e53e3e" if trend > 0.05 else "#38a169" if trend < -0.05 else "#a0aec0"
        st.markdown(
            f"<span style='color: {color}'>{arrow} {abs(trend)*100:.1f}%</span>",
            unsafe_allow_html=True,
        )

    with col4:
        st.write(f"{patient['days_since_last_brote']} días")

    with col5:
        adherence = patient["adherence_pct"] * 100
        st.progress(patient["adherence_pct"], text=f"{adherence:.0f}%")


# ══════════════════════════════════════════════════════════════════════════════
# PÁGINAS
# ══════════════════════════════════════════════════════════════════════════════


def page_overview():
    """Página principal: vista general."""
    st.title("🧠 EM Predictor - Panel de Control")
    st.markdown("---")

    # Métricas principales
    stats = fetch_alert_stats()
    patients = fetch_patients_summary()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        critical_count = len(patients[patients["risk_level"] == "critical"])
        render_metric_card(
            "Pacientes Alto Riesgo",
            str(critical_count),
            "Requieren atención",
            "critical" if critical_count > 0 else "ok",
        )

    with col2:
        render_metric_card(
            "Alertas Pendientes",
            str(stats["total_pending"]),
            "Sin reconocer",
            "warning" if stats["total_pending"] > 0 else "ok",
        )

    with col3:
        render_metric_card(
            "Pacientes Activos",
            str(len(patients)),
            "En seguimiento",
        )

    with col4:
        avg_time = stats.get("avg_response_time_hours")
        time_str = f"{avg_time:.1f}h" if avg_time else "N/A"
        render_metric_card(
            "Tiempo Respuesta",
            time_str,
            "Promedio últimos 30 días",
        )

    st.markdown("---")

    # Gráfico de distribución de riesgo
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📊 Distribución de Riesgo")

        fig = px.histogram(
            patients,
            x="risk_score",
            nbins=20,
            color_discrete_sequence=["#63b3ed"],
            labels={"risk_score": "Probabilidad de Brote", "count": "Pacientes"},
        )
        fig.add_vline(
            x=THRESHOLD_WARNING, line_dash="dash", line_color="#dd6b20", annotation_text="Warning"
        )
        fig.add_vline(
            x=THRESHOLD_CRITICAL, line_dash="dash", line_color="#e53e3e", annotation_text="Crítico"
        )
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0",
            xaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("📈 Por Nivel")

        risk_counts = patients["risk_level"].value_counts()
        fig = go.Figure(
            data=[
                go.Pie(
                    labels=["Bajo", "Moderado", "Crítico"],
                    values=[
                        risk_counts.get("ok", 0),
                        risk_counts.get("warning", 0),
                        risk_counts.get("critical", 0),
                    ],
                    marker_colors=["#38a169", "#dd6b20", "#e53e3e"],
                    hole=0.6,
                )
            ]
        )
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0",
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.2),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Lista de pacientes de alto riesgo
    st.subheader("⚠️ Pacientes que Requieren Atención")

    high_risk = patients[patients["risk_score"] >= THRESHOLD_WARNING].sort_values(
        "risk_score", ascending=False
    )

    if len(high_risk) == 0:
        st.success("✅ No hay pacientes en riesgo elevado actualmente.")
    else:
        # Header
        col1, col2, col3, col4, col5 = st.columns([2, 2, 2, 2, 2])
        col1.markdown("**Paciente**")
        col2.markdown("**Riesgo**")
        col3.markdown("**Tendencia 7d**")
        col4.markdown("**Último Brote**")
        col5.markdown("**Adherencia**")

        for _, patient in high_risk.head(10).iterrows():
            render_patient_row(patient)


def page_alerts():
    """Página de gestión de alertas."""
    st.title("🔔 Gestión de Alertas")
    st.markdown("---")

    # Filtros
    col1, col2, col3 = st.columns(3)
    with col1:
        status_filter = st.selectbox(
            "Estado",
            ["Todas", "Pendientes", "Reconocidas"],
            index=1,
        )
    with col2:
        level_filter = st.selectbox(
            "Nivel",
            ["Todos", "Crítico", "Warning", "Info"],
        )
    with col3:
        if st.button("🔄 Actualizar"):
            st.cache_data.clear()

    # Mapear filtros
    status_map = {"Todas": None, "Pendientes": "pending", "Reconocidas": "acknowledged"}
    level_map = {"Todos": None, "Crítico": "critical", "Warning": "warning", "Info": "info"}

    alerts_df = fetch_alerts(
        status=status_map.get(status_filter),
        limit=100,
    )

    if level_filter != "Todos" and not alerts_df.empty:
        alerts_df = alerts_df[alerts_df["alert_level"] == level_map[level_filter]]

    st.markdown("---")

    if alerts_df.empty:
        st.info("No hay alertas que coincidan con los filtros.")
    else:
        for _, alert in alerts_df.iterrows():
            with st.container():
                col1, col2, col3, col4 = st.columns([1, 3, 2, 2])

                with col1:
                    level = alert.get("alert_level", "info")
                    if level == "critical":
                        st.markdown("🔴 **CRÍTICO**")
                    elif level == "warning":
                        st.markdown("🟠 **WARNING**")
                    else:
                        st.markdown("🔵 **INFO**")

                with col2:
                    user_short = alert.get("user_id_hash", "")[:12] + "..."
                    prob = alert.get("relapse_probability")
                    prob_str = f" ({int(prob*100)}%)" if prob else ""
                    st.markdown(f"**Paciente:** {user_short}{prob_str}")
                    st.caption(f"ID: {alert.get('id')} | Tipo: {alert.get('alert_type')}")

                with col3:
                    triggered = alert.get("triggered_at", "")
                    if triggered:
                        try:
                            dt = datetime.fromisoformat(triggered.replace("Z", "+00:00"))
                            st.write(dt.strftime("%d/%m/%Y %H:%M"))
                        except:
                            st.write(triggered)

                with col4:
                    status = alert.get("status", "pending")
                    if status == "pending":
                        if st.button("✓ Reconocer", key=f"ack_{alert.get('id')}"):
                            try:
                                response = httpx.post(
                                    f"{ALERT_URL}/v1/alerts/{alert.get('id')}/acknowledge",
                                    json={"acknowledged_by": "dashboard_user"},
                                    timeout=10,
                                )
                                if response.status_code == 200:
                                    st.success("Alerta reconocida")
                                    st.cache_data.clear()
                                    st.rerun()
                            except Exception as e:
                                st.error(f"Error: {e}")
                    else:
                        st.write(f"✅ {alert.get('acknowledged_by', 'N/A')}")

                st.markdown("---")


def page_patient_detail():
    """Página de detalle de paciente."""
    st.title("👤 Detalle de Paciente")
    st.markdown("---")

    patients = fetch_patients_summary()
    patient_options = patients["display_id"].tolist()

    selected = st.selectbox("Seleccionar paciente", patient_options)

    if selected:
        patient = patients[patients["display_id"] == selected].iloc[0]

        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("Información")
            st.metric("ID", patient["display_id"])
            st.metric("Riesgo Actual", f"{int(patient['risk_score']*100)}%")
            st.metric("Días desde último brote", patient["days_since_last_brote"])
            st.metric("Adherencia", f"{int(patient['adherence_pct']*100)}%")

        with col2:
            st.subheader("📈 Evolución de Riesgo (Simulado)")

            # Datos sintéticos de evolución
            import numpy as np

            dates = pd.date_range(end=datetime.now(), periods=30, freq="D")
            base_risk = patient["risk_score"]
            noise = np.random.normal(0, 0.05, 30)
            trend = np.linspace(-0.1, 0, 30)
            risk_history = np.clip(base_risk + noise + trend, 0, 1)

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=risk_history,
                    mode="lines+markers",
                    name="Riesgo",
                    line=dict(color="#63b3ed", width=2),
                    marker=dict(size=4),
                )
            )
            fig.add_hline(y=THRESHOLD_WARNING, line_dash="dash", line_color="#dd6b20")
            fig.add_hline(y=THRESHOLD_CRITICAL, line_dash="dash", line_color="#e53e3e")
            fig.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#e2e8f0",
                xaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.1)", range=[0, 1]),
                height=300,
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.subheader("🔔 Alertas Recientes")
        st.info("Funcionalidad conectada al servicio de alertas en producción.")


# ══════════════════════════════════════════════════════════════════════════════
# NAVEGACIÓN
# ══════════════════════════════════════════════════════════════════════════════


def main():
    # Sidebar
    with st.sidebar:
        st.image(
            "https://via.placeholder.com/200x60/1a2744/63b3ed?text=EM+Predictor",
            use_column_width=True,
        )
        st.markdown("---")

        page = st.radio(
            "Navegación",
            ["📊 Vista General", "🔔 Alertas", "👤 Detalle Paciente"],
            label_visibility="collapsed",
        )

        st.markdown("---")
        st.caption("v0.1.0 - Prototipo")
        st.caption(f"Última actualización: {datetime.now().strftime('%H:%M')}")

    # Renderizar página
    if page == "📊 Vista General":
        page_overview()
    elif page == "🔔 Alertas":
        page_alerts()
    elif page == "👤 Detalle Paciente":
        page_patient_detail()


if __name__ == "__main__":
    main()

