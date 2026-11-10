"""
Modulo de Estilos y Sistema de Diseno Clinico - NEURO-CALC
Sistema de diseno hospitalario sin sombras difusas ni tonos neon.
"""

CLINICAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

/* Base tipografica y canvas */
html, body, [class*="css"], [class*="st-"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    color: #0F172A;
}

/* Preservar iconos nativos de Streamlit */
span[data-testid="stIconMaterial"], .material-icons, .material-symbols-rounded {
    font-family: "Material Symbols Rounded", "Material Icons" !important;
}

.stApp {
    background-color: #F4F6FA !important;
}

section[data-testid="stSidebar"] {
    background-color: #FFFFFF !important;
    border-right: 1px solid #E2E8F0 !important;
}

.main .block-container {
    padding-top: 1.25rem !important;
    padding-bottom: 2.5rem !important;
    max-width: 1360px !important;
}

/* Textos y titulos */
p, span, label, li {
    color: #475569;
    font-size: 0.90rem;
    line-height: 1.5;
}

h1, h2, h3, h4, h5, h6 {
    color: #0F172A !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em;
}

strong, b {
    color: #0F172A !important;
    font-weight: 600;
}

/* Ocultar chrome no esencial preservando el boton para reabrir barra lateral */
#MainMenu,
footer,
[data-testid="stMainMenu"],
[data-testid="stAppDeployButton"],
.stDeployButton,
[data-testid="stToolbarActions"],
div[data-testid="stDecoration"] {
    display: none !important;
}

header[data-testid="stHeader"] {
    background: transparent !important;
    pointer-events: none;
}

header[data-testid="stHeader"] * {
    pointer-events: auto;
}

div[data-testid="stToolbar"] {
    visibility: visible !important;
    background: transparent !important;
}

[data-testid="stExpandSidebarButton"],
[data-testid="stSidebarCollapsedControl"] {
    visibility: visible !important;
    display: flex !important;
    opacity: 1 !important;
    background-color: #FFFFFF !important;
    border: 1px solid #CBD5E1 !important;
    border-radius: 6px !important;
    color: #0F172A !important;
    box-shadow: none !important;
    z-index: 999999 !important;
}

/* Botones primarios solidos (Cobalto medico, sin sombras difusas) */
button[kind="primary"],
button[kind="primaryFormSubmit"],
button[data-testid="baseButton-primary"],
button[data-testid="stBaseButton-primary"],
button[data-testid="stBaseButton-primaryFormSubmit"],
.stButton > button[kind="primary"],
div[data-testid="stButton"] > button[kind="primary"],
div[data-testid="stFormSubmitButton"] > button {
    background-color: #2563EB !important;
    border: 1px solid #1D4ED8 !important;
    border-radius: 6px !important;
    box-shadow: none !important;
    color: #FFFFFF !important;
    min-height: 38px !important;
}

button[kind="primary"] *,
button[kind="primaryFormSubmit"] *,
button[data-testid="baseButton-primary"] *,
button[data-testid="stBaseButton-primary"] *,
button[data-testid="stBaseButton-primaryFormSubmit"] *,
div[data-testid="stButton"] > button[kind="primary"] *,
div[data-testid="stFormSubmitButton"] > button * {
    color: #FFFFFF !important;
    font-weight: 600 !important;
    font-size: 0.86rem !important;
    text-shadow: none !important;
}

button[kind="primary"]:hover,
button[kind="primaryFormSubmit"]:hover,
button[data-testid="stBaseButton-primary"]:hover,
div[data-testid="stButton"] > button[kind="primary"]:hover,
div[data-testid="stFormSubmitButton"] > button:hover {
    background-color: #1D4ED8 !important;
    border-color: #1E40AF !important;
}

/* Botones secundarios (Superficie limpia, borde solido, cero sombras difusas) */
button[kind="secondary"],
button[data-testid="baseButton-secondary"],
button[data-testid="stBaseButton-secondary"],
div[data-testid="stButton"] > button[kind="secondary"],
div[data-testid="stPopover"] > button {
    background-color: #FFFFFF !important;
    border: 1px solid #CBD5E1 !important;
    border-radius: 6px !important;
    box-shadow: none !important;
    color: #334155 !important;
    min-height: 38px !important;
}

button[kind="secondary"] *,
button[data-testid="baseButton-secondary"] *,
button[data-testid="stBaseButton-secondary"] *,
div[data-testid="stButton"] > button[kind="secondary"] *,
div[data-testid="stPopover"] > button * {
    color: #334155 !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
}

button[kind="secondary"]:hover,
button[data-testid="stBaseButton-secondary"]:hover,
div[data-testid="stButton"] > button[kind="secondary"]:hover,
div[data-testid="stPopover"] > button:hover {
    background-color: #F8FAFC !important;
    border-color: #94A3B8 !important;
}

button[kind="secondary"]:hover *,
div[data-testid="stPopover"] > button:hover * {
    color: #0F172A !important;
}

/* Jerarquia de encabezados de pagina */
.page-header-wrap {
    margin: 12px 0 18px 0;
}
.page-eyebrow {
    font-size: 0.75rem;
    font-weight: 700;
    color: #64748B !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 4px;
}
.page-title {
    margin: 0 0 4px 0 !important;
    font-size: 1.85rem !important;
    font-weight: 700 !important;
    color: #0F172A !important;
    line-height: 1.2 !important;
    letter-spacing: -0.025em !important;
}
.page-subtitle {
    font-size: 0.92rem;
    color: #475569 !important;
    margin: 0;
    line-height: 1.45;
}

.section-heading {
    font-size: 1.08rem;
    font-weight: 700;
    color: #0F172A !important;
    margin: 0 0 12px 0;
}

/* Superficies y contenedores unificados (solidos, bordes 1px, sin sombras) */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 8px !important;
    padding: 16px 20px !important;
    margin-bottom: 12px !important;
    box-shadow: none !important;
}

/* Tarjetas KPI operativas y alertas clinicas */
.kpi-card {
    background-color: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 8px;
    padding: 16px 18px;
    margin-bottom: 12px;
}
.kpi-card-alert {
    background-color: #FEF2F2;
    border: 1px solid #FECACA;
    border-left: 4px solid #DC2626;
    border-radius: 8px;
    padding: 16px 18px;
    margin-bottom: 12px;
}
.kpi-label {
    font-size: 0.78rem;
    font-weight: 600;
    color: #475569 !important;
    margin-bottom: 6px;
}
.kpi-label-alert {
    font-size: 0.78rem;
    font-weight: 600;
    color: #991B1B !important;
    margin-bottom: 6px;
}
.kpi-value {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 1.65rem;
    font-weight: 700;
    color: #0F172A !important;
    line-height: 1.15;
}
.kpi-value-alert {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 1.65rem;
    font-weight: 700;
    color: #B91C1C !important;
    line-height: 1.15;
}
.kpi-unit {
    font-size: 0.85rem;
    font-weight: 500;
    color: #64748B !important;
    margin-left: 4px;
}
.kpi-context {
    font-size: 0.76rem;
    color: #64748B !important;
    margin-top: 6px;
}
.kpi-context-alert {
    font-size: 0.76rem;
    color: #7F1D1D !important;
    margin-top: 6px;
}

/* Pills de estado medico (tonos sobrios no-neon, texto completo sin truncar) */
.status-pill {
    display: inline-block;
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    padding: 3px 10px !important;
    border-radius: 999px !important;
    white-space: nowrap !important;
    letter-spacing: 0.01em;
}
.pill-critical {
    background-color: #FEF2F2 !important;
    color: #991B1B !important;
    border: 1px solid #FECACA !important;
}
.pill-warning {
    background-color: #FFFBEB !important;
    color: #92400E !important;
    border: 1px solid #FDE68A !important;
}
.pill-stable {
    background-color: #ECFDF5 !important;
    color: #065F46 !important;
    border: 1px solid #A7F3D0 !important;
}
.pill-info {
    background-color: #EFF6FF !important;
    color: #1E40AF !important;
    border: 1px solid #BFDBFE !important;
}

/* Compatibilidad para tarjetas legacy */
.clinical-card {
    background-color: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 8px;
    padding: 14px 16px;
    margin-bottom: 10px;
}
.clinical-card-header {
    font-size: 0.76rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: #475569 !important;
    margin-bottom: 6px;
}
.clinical-card-value {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 1.45rem;
    font-weight: 700;
    color: #0F172A !important;
    line-height: 1.2;
}
.clinical-card-unit {
    font-size: 0.84rem;
    color: #64748B !important;
    margin-left: 4px;
    font-weight: 500;
}
.clinical-card-badge {
    display: inline-block;
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    padding: 2px 8px !important;
    border-radius: 4px !important;
    margin-top: 6px;
}
.badge-stable { background-color: #ECFDF5 !important; color: #065F46 !important; border: 1px solid #A7F3D0 !important; }
.badge-warning { background-color: #FFFBEB !important; color: #92400E !important; border: 1px solid #FDE68A !important; }
.badge-cyan { background-color: #EFF6FF !important; color: #1E40AF !important; border: 1px solid #BFDBFE !important; }
.badge-critical { background-color: #FEF2F2 !important; color: #991B1B !important; border: 1px solid #FECACA !important; }

/* Campos de entrada y selectores limpios */
div[data-baseweb="input"] > div,
div[data-baseweb="select"] > div {
    background-color: #F8FAFC !important;
    border-color: #E2E8F0 !important;
    border-radius: 6px !important;
}
div[data-baseweb="select"] * {
    color: #0F172A !important;
}

/* Pestanas (Tabs) con jerarquia sobria */
.stTabs [data-baseweb="tab-list"] {
    gap: 12px;
    border-bottom: 1px solid #E2E8F0 !important;
    margin-bottom: 14px;
}
.stTabs [data-baseweb="tab"] {
    height: 40px;
    font-size: 0.88rem !important;
    font-weight: 500 !important;
    color: #64748B !important;
    background-color: transparent !important;
    padding: 0 12px !important;
}
.stTabs [aria-selected="true"] {
    color: #2563EB !important;
    font-weight: 600 !important;
    border-bottom: 2px solid #2563EB !important;
}

/* Tablas estaticas */
div[data-testid="stTable"] table {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.82rem !important;
    background-color: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important;
    width: 100% !important;
}
div[data-testid="stTable"] th {
    background-color: #F8FAFC !important;
    color: #334155 !important;
    font-weight: 600 !important;
    font-size: 0.74rem !important;
    padding: 10px 12px !important;
    border-bottom: 1px solid #E2E8F0 !important;
}
div[data-testid="stTable"] td {
    color: #0F172A !important;
    padding: 10px 12px !important;
    border-bottom: 1px solid #F1F5F9 !important;
}

/* KaTeX alto contraste */
.katex, .katex *, .katex-display * {
    color: #0F172A !important;
}

code {
    background-color: #F1F5F9 !important;
    color: #0F172A !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 4px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.82em !important;
    padding: 2px 6px !important;
}

div[data-testid="column"] {
    padding: 0 6px !important;
}

.stSlider {
    padding-top: 0px !important;
    padding-bottom: 0px !important;
}
</style>
"""
