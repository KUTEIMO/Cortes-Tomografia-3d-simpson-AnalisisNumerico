# NEURO-CALC | Estacion de Telemetria Radiologica y Volumetria Tomografica 3D

Plataforma computacional para el procesamiento de series axiales de resonancia magnetica (RMN FLAIR/T1) y tomografia computarizada, enfocada en segmentacion espectral multicanal, reconstruccion volumetrica 3D por *lofting* e integracion numerica de cuarto orden mediante **Cuadratura Compuesta de Simpson 1/3 y 3/8**.

---

## Arquitectura del Sistema

* **Motor de Integracion Numerica (`src/engine/`):**
  * **Simpson 1/3 Compuesto ($\mathcal{O}(h^4)$):** Integracion sobre ternas equiespaciadas de cortes transversales.
  * **Particion Hibrida Simpson 3/8:** Tratamiento automatico de series con paridad impar de subintervalos conservando convergencia cuártica global $\mathcal{O}(h^4)$ sin degradar al Trapecio.
  * **Metodos Comparativos:** Evaluacion simultanea contra Regla del Trapecio ($\mathcal{O}(h^2)$), Conteo de Voxeles / Riemann ($\mathcal{O}(h)$), Trazadores Cubicos Naturales $C^2$, Extrapolacion de Richardson y Tabla de Romberg.
  * **Evaluacion de Error:** Acotacion de error de truncamiento longitudinal via diferencias finitas centrales de 5 puntos y estimadores de incertidumbre planar y PVE.
* **Vision Artificial y Segmentacion (`src/vision/`):**
  * Extraccion espectral multicanal, supresion de grasa retro-orbitaria, confinamiento parenquimatoso geodesico y poligonizacion subpixel de contornos (Douglas-Peucker).
* **Telemetria DICOM e Integridad (`src/telemetry/`):**
  * Desempaquetado de metadatos fisicos ($h$, pixel spacing) con verificacion ciclica CRC32 corte por corte.
* **Interfaz Clinica PACS (`src/ui/`, `src/app.py`):**
  * Directorio de historias clinicas y control de acceso RBAC con auditoria inmutable.
  * Admision e ingesta segura de paquetes `.zip` (con validacion de rutas y extensiones).
  * Visor biaxial sincronizado 3D / Axial 2D con reproductor Cine-Loop e informes exportables.
  * Puerta de autenticacion con verificacion SHA-256 (`st.secrets["ACCESS_KEY"]` compatible con Streamlit Cloud).

---

## Estructura del Repositorio

```text
├── .streamlit/
│   └── config.toml                 # Configuracion de tema y servidor Streamlit
├── data/
│   ├── banco_casos_reales_zip/     # Paquetes clinicos TCGA-LGG comprimidos (.zip)
│   ├── database/                   # Registro JSON de casos y bitacora CRC32
│   └── kaggle_cases/               # Series axiales de resonancia magnetica reales
├── src/
│   ├── database/                   # Gestor CRUD de pacientes, roles RBAC y auditoria
│   ├── engine/                     # Simpson 1/3, 3/8, Romberg, Splines y Error
│   ├── telemetry/                  # Paquetes telemetricos y lectura DICOM
│   ├── ui/                         # Vistas modulares, renderizado 3D Plotly y CSS
│   ├── vision/                     # Segmentacion morfometrica y metricas Dice
│   └── app.py                      # Punto de entrada principal y enrutador
├── tests/                          # Suite de pruebas unitarias
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Instalacion y Ejecucion Local

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Ejecutar suite de pruebas unitarias
python -m unittest discover -s tests

# 3. Iniciar estacion web
streamlit run src/app.py
```

---

## Despliegue en Streamlit Community Cloud

1. Subir el repositorio a GitHub.
2. En [share.streamlit.io](https://share.streamlit.io), crear una nueva aplicacion apuntando a `src/app.py`.
3. *(Opcional)* En *App Settings -> Secrets*, configurar una clave personalizada:
   ```toml
   ACCESS_KEY = "tu_clave_privada"
   ```
