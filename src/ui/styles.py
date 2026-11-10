"""
Modulo de Estilos y Sistema de Diseno Clinico - NEURO-CALC
Sistema: Estacion de Telemetria y Volumetria Tomografica

Tokens visuales de alta legibilidad:
- Canvas Base: #F8F9FA
- Superficies / Cards: #FFFFFF con borde 1px #EAECF0
- Tipografia Primaria: #101828 | Secundaria: #475467
- Acento Primario: #2563EB
"""

CLINICAL_CSS = """
<style>
/* Importacion de fuentes tecnicas y analiticas */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* Reset y configuracion base */
html, body, [class*="css"], [class*="st-"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    color: #101828;
}

/* Preservar fuente de iconos nativos de Streamlit */
span[data-testid="stIconMaterial"], .material-icons, .material-symbols-rounded {
    font-family: "Material Symbols Rounded", "Material Icons" !important;
}

/* Jerarquia de texto Braintrust */
p, span, label, li {
    color: #475467;
    font-size: 0.88rem;
    line-height: 1.5;
}

h1, h2, h3, h4, h5, h6 {
    color: #101828 !important;
    font-weight: 600 !important;
    letter-spacing: -0.01em;
}

strong, b {
    color: #101828 !important;
    font-weight: 600;
}

/* ==========================================================================
   SISTEMA DE BOTONES (BRAINTRUST / MOBBIN) - ALTO CONTRASTE WCAG AAA
   ========================================================================== */
/* Botones primarios (Cobalt Blue #2563EB): Texto BLANCO nitido en cualquier elemento */
button[kind="primary"],
button[kind="primaryFormSubmit"],
button[data-testid="baseButton-primary"],
button[data-testid="stBaseButton-primary"],
button[data-testid="stBaseButton-primaryFormSubmit"],
.stButton > button[kind="primary"],
.stButton > button[data-testid="baseButton-primary"],
div[data-testid="stButton"] > button[kind="primary"],
div[data-testid="stFormSubmitButton"] > button,
div[data-testid="stFormSubmitButton"] > button[kind="primaryFormSubmit"] {
    background-color: #2563EB !important;
    border: 1px solid #1D4ED8 !important;
    border-radius: 6px !important;
    box-shadow: 0 1px 2px rgba(16, 24, 40, 0.05) !important;
    color: #FFFFFF !important;
}

button[kind="primary"] *,
button[kind="primaryFormSubmit"] *,
button[data-testid="baseButton-primary"] *,
button[data-testid="stBaseButton-primary"] *,
button[data-testid="stBaseButton-primaryFormSubmit"] *,
.stButton > button[kind="primary"] *,
.stButton > button[data-testid="baseButton-primary"] *,
div[data-testid="stButton"] > button[kind="primary"] *,
div[data-testid="stFormSubmitButton"] > button *,
button[kind="primary"] p,
button[data-testid="baseButton-primary"] p,
button[kind="primary"] span,
button[data-testid="baseButton-primary"] span,
button[kind="primary"] div,
button[data-testid="baseButton-primary"] div {
    color: #FFFFFF !important;
    font-weight: 600 !important;
    font-size: 0.86rem !important;
    letter-spacing: 0.01em !important;
    text-shadow: 0 1px 2px rgba(0, 0, 0, 0.25) !important;
}

button[kind="primary"]:hover,
button[kind="primaryFormSubmit"]:hover,
button[data-testid="baseButton-primary"]:hover,
button[data-testid="stBaseButton-primary"]:hover,
div[data-testid="stButton"] > button[kind="primary"]:hover,
div[data-testid="stFormSubmitButton"] > button:hover {
    background-color: #1D4ED8 !important;
    border-color: #1E40AF !important;
}

button[kind="primary"]:hover *,
button[kind="primaryFormSubmit"]:hover *,
button[data-testid="baseButton-primary"]:hover *,
button[data-testid="stBaseButton-primary"]:hover *,
div[data-testid="stFormSubmitButton"] > button:hover * {
    color: #FFFFFF !important;
}

button[kind="primary"]:active,
button[data-testid="baseButton-primary"]:active,
div[data-testid="stFormSubmitButton"] > button:active {
    background-color: #1E40AF !important;
}

/* Botones secundarios (Base blanca, borde solido, texto charcoal) */
button[kind="secondary"],
button[data-testid="baseButton-secondary"],
button[data-testid="stBaseButton-secondary"],
.stButton > button[kind="secondary"],
.stButton > button[data-testid="baseButton-secondary"],
div[data-testid="stButton"] > button[kind="secondary"] {
    background-color: #FFFFFF !important;
    border: 1px solid #D0D5DD !important;
    border-radius: 6px !important;
    box-shadow: 0 1px 2px rgba(16, 24, 40, 0.05) !important;
    color: #344054 !important;
}

button[kind="secondary"] *,
button[data-testid="baseButton-secondary"] *,
button[data-testid="stBaseButton-secondary"] *,
.stButton > button[kind="secondary"] *,
.stButton > button[data-testid="baseButton-secondary"] *,
div[data-testid="stButton"] > button[kind="secondary"] *,
button[kind="secondary"] p,
button[data-testid="baseButton-secondary"] p,
button[kind="secondary"] span,
button[data-testid="baseButton-secondary"] span {
    color: #344054 !important;
    font-weight: 600 !important;
    font-size: 0.86rem !important;
}

button[kind="secondary"]:hover,
button[data-testid="baseButton-secondary"]:hover,
button[data-testid="stBaseButton-secondary"]:hover,
div[data-testid="stButton"] > button[kind="secondary"]:hover {
    background-color: #F9FAFB !important;
    border-color: #98A2B3 !important;
}

button[kind="secondary"]:hover * {
    color: #101828 !important;
}

/* Soporte matematico KaTeX de alto contraste */
.katex, .katex *, .katex-display * {
    color: #101828 !important;
}
.katex-display {
    padding: 12px 0 !important;
    margin: 8px 0 !important;
}

/* Cajas de valores evaluados segun DESING.md */
code {
    background-color: #F2F4F7 !important;
    color: #101828 !important;
    border: 1px solid #EAECF0 !important;
    border-radius: 4px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.82em !important;
    padding: 3px 8px !important;
}

/* Ocultar elementos decorativos por defecto sin ocultar el boton para expandir la barra lateral */
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
    border: 1px solid #D0D5DD !important;
    border-radius: 6px !important;
    color: #101828 !important;
    box-shadow: 0 1px 2px rgba(16, 24, 40, 0.08) !important;
    z-index: 999999 !important;
}

/* Fondo principal y barra lateral clara */
.stApp {
    background-color: #F8F9FA !important;
}
section[data-testid="stSidebar"] {
    background-color: #FFFFFF !important;
    border-right: 1px solid #EAECF0 !important;
}

/* Contenedores de tarjeta (Cards Braintrust) */
.clinical-card {
    background-color: #FFFFFF;
    border: 1px solid #EAECF0;
    border-radius: 6px;
    padding: 10px 14px;
    margin-bottom: 8px;
}

.clinical-card-header {
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #475467 !important;
    margin-bottom: 8px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.clinical-card-value {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 1.45rem;
    font-weight: 700;
    color: #101828 !important;
    line-height: 1.2;
}

.clinical-card-unit {
    font-size: 0.85rem;
    color: #475467 !important;
    margin-left: 4px;
    font-weight: 500;
}

/* Sistema de Badges Tecnicos */
.clinical-card-badge {
    display: inline-block;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.70rem !important;
    font-weight: 600 !important;
    padding: 2px 8px !important;
    border-radius: 4px !important;
    margin-top: 6px;
    letter-spacing: 0.04em;
}

.clinical-card-badge * {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.70rem !important;
    font-weight: 600 !important;
}

.badge-stable {
    background-color: #ECFDF3 !important;
    color: #027A48 !important;
    border: 1px solid #A6F4C5 !important;
}
.badge-stable * {
    color: #027A48 !important;
}

.badge-warning {
    background-color: #FFFAEB !important;
    color: #B54708 !important;
    border: 1px solid #FEDF89 !important;
}
.badge-warning * {
    color: #B54708 !important;
}

.badge-cyan {
    background-color: #EFF8FF !important;
    color: #175CD3 !important;
    border: 1px solid #B2DDFF !important;
}
.badge-cyan * {
    color: #175CD3 !important;
}

.badge-critical {
    background-color: #FEF3F2 !important;
    color: #B42318 !important;
    border: 1px solid #FECDCA !important;
}
.badge-critical * {
    color: #B42318 !important;
}

/* Encabezado de la estacion */
.workstation-header {
    background-color: #FFFFFF;
    border: 1px solid #EAECF0;
    border-radius: 6px;
    padding: 8px 14px;
    margin-bottom: 8px;
}
.workstation-title {
    font-size: 1.05rem;
    font-weight: 700;
    letter-spacing: -0.01em;
    color: #101828 !important;
    margin: 0;
}
.workstation-subtitle {
    font-size: 0.76rem;
    color: #475467 !important;
    margin-top: 2px;
}

/* Tablas HTML estaticas (st.table) */
div[data-testid="stTable"] table {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.82rem !important;
    background-color: #FFFFFF !important;
    border: 1px solid #EAECF0 !important;
    border-radius: 6px !important;
    width: 100% !important;
    border-collapse: collapse !important;
}
div[data-testid="stTable"] th {
    background-color: #F9FAFB !important;
    color: #344054 !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    font-size: 0.75rem !important;
    padding: 10px 14px !important;
    border-bottom: 1px solid #EAECF0 !important;
}
div[data-testid="stTable"] td {
    color: #101828 !important;
    padding: 10px 14px !important;
    border-bottom: 1px solid #F2F4F7 !important;
}

/* Personalizacion del File Uploader en barra lateral */
div[data-testid="stFileUploader"] section {
    background-color: #F9FAFB !important;
    border: 1px dashed #D0D5DD !important;
    border-radius: 8px !important;
    padding: 14px !important;
}
div[data-testid="stFileUploader"] section * {
    color: #475467 !important;
}
div[data-testid="stFileUploader"] button {
    background-color: #FFFFFF !important;
    color: #344054 !important;
    border: 1px solid #D0D5DD !important;
    border-radius: 6px !important;
}

/* Pestañas (Tabs) */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    border-bottom: 1px solid #EAECF0 !important;
}
.stTabs [data-baseweb="tab"] {
    height: 40px;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    color: #475467 !important;
    background-color: transparent !important;
    border-radius: 0px !important;
    padding: 0 16px !important;
}
.stTabs [aria-selected="true"] {
    color: #2563EB !important;
    font-weight: 600 !important;
    border-bottom: 2px solid #2563EB !important;
}

/* Selectbox */
div[data-baseweb="select"] {
    background-color: #FFFFFF !important;
    border: 1px solid #D0D5DD !important;
    border-radius: 6px !important;
}
div[data-baseweb="select"] * {
    color: #101828 !important;
}

/* Cuadros informativos st.info */
div[data-testid="stAlert"] {
    background-color: #EFF8FF !important;
    border: 1px solid #B2DDFF !important;
    border-radius: 6px !important;
    color: #175CD3 !important;
}
div[data-testid="stAlert"] * {
    color: #175CD3 !important;
}

/* Slider de inspeccion de corte */
div[data-testid="stSlider"] label {
    font-weight: 600 !important;
    color: #101828 !important;
}

/* Jerarquia visual de layout, contenedores y espaciado */
.main .block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 2.5rem !important;
    max-width: 1420px !important;
}

.page-header {
    margin-bottom: 16px;
}
.page-eyebrow {
    font-size: 0.70rem;
    font-weight: 700;
    color: #2563EB !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
.page-title {
    margin: 2px 0 4px 0 !important;
    font-size: 1.48rem !important;
    font-weight: 700 !important;
    color: #101828 !important;
    line-height: 1.25 !important;
}
.page-subtitle {
    font-size: 0.85rem;
    color: #475467 !important;
    margin: 0;
}

.section-label {
    font-size: 0.72rem;
    font-weight: 700;
    color: #475467 !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin: 18px 0 8px 0;
    padding-bottom: 4px;
    border-bottom: 1px solid #EAECF0;
}

/* Tarjetas unificadas con st.container(border=True) */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: #FFFFFF !important;
    border: 1px solid #EAECF0 !important;
    border-radius: 8px !important;
    padding: 12px 16px !important;
    margin-bottom: 10px !important;
}

div[data-testid="column"] {
    padding: 0 5px !important;
}
.stSlider {
    padding-top: 0px !important;
    padding-bottom: 0px !important;
}

</style>
"""
