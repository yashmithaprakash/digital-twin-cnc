# DIGITAL TWIN FOR MANUFACTURING PROCESSES
### Project ID: PRJ_422
**Presidency University — School of Computer Science and Engineering**  
**Course:** Mini Project (CSE7102)

---

## 📌 Project Overview

A software-based **Cyber-Physical Digital Twin** for a high-speed CNC milling manufacturing process. Developed in **Python** using **Streamlit**, **Plotly**, **Scikit-learn**, **Pandas**, **NumPy**, and **SQLite**.

The system integrates real manufacturing sensor telemetry from the benchmark **PHM Society 2010 CNC Milling Dataset**, providing real-time condition monitoring, cutting tool flank wear prediction, explainable Remaining Useful Life (RUL) estimation, multivariate Isolation Forest anomaly detection, dynamic 3D/SVG virtual CNC representation, and physics-grounded "What-if" scenario simulation.

```
┌────────────────────────────────────────────────────────────────────────┐
│                      PHYSICAL CNC MILLING PROCESS                      │
│   Röders Tech RFM 760 │ Spindle: 10,400 RPM │ Feed: 1,555 mm/min       │
│   Workpiece: Inconel 718 │ Cutter: 6mm 3-Flute Ball Nose Carbide       │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                       MULTI-SENSOR TELEMETRY                           │
│   • Kistler 3-Axis Dynamometer: Cutting Force (Fx, Fy, Fz, Fres)       │
│   • Piezoelectric Accelerometers: Tri-axial Vibration (Vx, Vy, Vz, RMS)│
│   • Acoustic Emission Sensor: High-frequency stress waves (AE-RMS)     │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                     DATA PREPROCESSING & FEATURES                      │
│   • Butterworth zero-phase filtering, baseline detrending, windowing   │
│   • Time-Domain: RMS, Peak, Crest Factor, Kurtosis, Skewness           │
│   • Frequency-Domain: FFT power spectrum, tooth passing frequency     │
│   • Machining Physics: Radial-to-feed force ratio, total energy index  │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                           DIGITAL TWIN CORE                            │
│   • Virtual CNC State Synchronization (Normal, Warning, Critical)      │
│   • Scikit-learn Tool Wear Model (RandomForest: MAE=1.12µm, R²=0.998) │
│   • Isolation Forest Anomaly Detector with Explainable Sensor Attribution│
│   • Explainable RUL Trajectory Prediction Engine (ISO 8688)           │
│   • What-If Manufacturing Simulation (Taylor Tool Life + Kienzle Force)│
│   • Prescriptive Maintenance Advisory & SQLite Audit Trail Logger      │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│               STREAMLIT INDUSTRIAL IoT DASHBOARD UI                    │
│   • 9 Dedicated Pages │ Interactive Plotly Charts │ Dark Theme         │
│   • Real PHM 2010 Data & High-Fidelity Synthetic Simulation Fallback   │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 🎯 Key Objectives & Capabilities

1. **Condition Monitoring:** Real-time visibility into cutting forces, vibrations, and acoustic emission across all axis directions.
2. **Tool Wear Prediction:** Scikit-learn Random Forest regression model accurately predicting cutting tool flank wear ($VB$ in $\mu\text{m}$).
3. **Explainable RUL Estimation:** Multi-stage degradation tracking calculating remaining machining cuts, operational minutes, and 80% confidence intervals.
4. **Anomaly Detection:** Unsupervised Scikit-learn `IsolationForest` detecting anomalous chatter, chipping, or thermal breakdown with exact sensor attribution.
5. **Virtual CNC Machine State:** Dynamic vector-rendered CNC milling representation reflecting spindle speed, feed rate, depth of cut, coolant status, and operating mode.
6. **What-if Scenario Simulation:** Dual-condition comparison (`CURRENT CONDITION` vs `SIMULATED CONDITION`) testing alternative feed rates, spindle speeds, depths of cut, and materials.
7. **Prescriptive Maintenance Decision Support:** Automatic SOP generation, downtime estimations, and local SQLite audit logging.

---

## 📊 Dataset: PHM Society 2010 CNC Milling Benchmark

The system integrates the genuine **PHM Society 2010 CNC Milling Dataset**:

| Parameter | Specification |
| :--- | :--- |
| **Machine Tool** | Röders Tech RFM 760 High-Speed Machining Center |
| **Workpiece Material** | Inconel 718 Superalloy (High Hardness & Thermal Resistance) |
| **Cutting Tool** | $6\,\text{mm}$ Diameter, 3-Flute Ball Nose Tungsten Carbide Cutter |
| **Spindle Speed** | $10,400\,\text{RPM}$ (Nominal) |
| **Table Feed Rate** | $1,555\,\text{mm/min}$ ($0.05\,\text{mm/tooth}$) |
| **Depth of Cut** | Radial: $0.125\,\text{mm}$, Axial: $1.5\,\text{mm}$ |
| **Cutting Environment** | High-speed dry milling with air blow |
| **Sampling Rate** | $50\,\text{kHz}$ across all 7 sensor channels |
| **Training Cutters** | `c1` (315 cuts), `c4` (315 cuts), `c6` (315 cuts) = **945 total cuts** |
| **Wear Measurements** | Flute 1, Flute 2, Flute 3 optical wear ($VB_{max}$ in $\mu\text{m}$) |

> **Fallback Architecture:** If real dataset files are moved or custom simulations are desired, the application seamlessly toggles to a physics-grounded synthetic digital twin.

---

## 🔬 Machine Learning Performance Metrics

Trained using **80% Train / 20% Test** cross-validation separation on real PHM 2010 cuts:

| Metric | Measured Value | Industrial Benchmark |
| :--- | :--- | :--- |
| **Mean Absolute Error (MAE)** | **$1.123\,\mu\text{m}$** | $< 15\,\mu\text{m}$ |
| **Root Mean Squared Error (RMSE)** | **$1.709\,\mu\text{m}$** | $< 20\,\mu\text{m}$ |
| **Coefficient of Determination ($R^2$)** | **$0.9984$** | $> 0.90$ |
| **Training Samples** | 756 milling passes | — |
| **Testing Samples** | 189 milling passes | — |

*Top Predictive Features:* Resultant cutting energy index (70.3%), AE rolling energy (13.4%), high-frequency AE RMS (6.5%), and X-axis vibration (6.0%).

---

## 🧭 Dashboard Pages & Modules

The web dashboard is organized into 9 modules accessible via the sidebar:

1. **📊 Dashboard Home:** Overall machine status, live sensor KPI cards, tool health gauges, and critical anomaly banners.
2. **📡 Sensor Telemetry:** Interactive Plotly waveforms for cutting force (Fx, Fy, Fz), vibration (Vx, Vy, Vz), and acoustic emission (AE).
3. **🔬 Tool Health Analytics:** Wear progression curves, ISO 8688 warning/critical thresholds, model evaluation badges ($MAE, RMSE, R^2$), and feature importance breakdown.
4. **🚨 Anomaly Detection:** Isolation Forest scoring, timeline anomaly flags, multi-sensor comparison charts, and sensor contribution breakdown panel.
5. **⏳ RUL Estimation:** Step-by-step explainable RUL calculation, remaining cut count, machining minutes, and uncertainty bounds.
6. **🌐 3D Digital Twin:** Dynamic SVG CNC milling machine state visualization synchronizing with spindle speed, feed rate, coolant, and tool wear.
7. **🧪 What-if Simulation:** Interactive parameter sliders testing custom machining conditions against current operation with Taylor tool life predictions.
8. **📋 Maintenance Advisory:** Prescriptive maintenance actions, Standard Operating Procedure (SOP) checklists, and SQLite service logging.
9. **📁 Dataset Management:** Complete schema inspector, cutter selector (`c1`, `c4`, `c6`, or `All`), ML retraining button, and CSV uploader.

---

## 🚀 Quickstart Instructions

### 1. Requirements
- Python 3.9, 3.10, 3.11, or 3.12
- Windows, macOS, or Linux

### 2. Setup Virtual Environment
```bash
# Clone the repository
git clone https://github.com/yashmithaprakash/digital-twin-cnc.git
cd digital-twin-cnc

# Create and activate virtual environment
python -m venv .venv

# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Retrain Models (Optional)
```bash
python models/model_training.py
```

### 4. Run the Streamlit Application
```bash
streamlit run app.py
```
Open **`http://localhost:8501`** in any web browser.

---

## 📁 Repository Directory Structure

```
digital-twin-cnc/
│
├── app.py                      # Main Streamlit Industrial Dashboard application
├── requirements.txt            # Python package dependencies
├── README.md                   # Project documentation & academic specifications
├── .gitignore                  # Git ignore rules
├── test_pages.py               # Automated test suite for all 9 dashboard pages
├── test_verify.py              # Automated test suite for core ML & physics logic
│
├── data/                       # Dataset storage
│   ├── phm2010_milling_full.csv     # Unified authentic PHM 2010 telemetry (945 cuts)
│   ├── phm2010_milling_features.csv # Statistical features dataset
│   ├── c1_wear.csv                  # Cutter 1 ground truth wear
│   ├── c4_wear.csv                  # Cutter 4 ground truth wear
│   ├── c6_wear.csv                  # Cutter 6 ground truth wear
│   ├── sample_cnc_data.csv          # High-fidelity synthetic fallback dataset
│   └── digital_twin_cnc.db          # Local SQLite telemetry and maintenance database
│
├── models/                     # Trained ML models and scripts
│   ├── model_training.py            # Model training & evaluation pipeline
│   ├── tool_wear_model.joblib       # Fitted Random Forest regression model
│   └── anomaly_detector.joblib      # Fitted Isolation Forest anomaly model
│
├── src/                        # Core algorithmic modules
│   ├── data_loader.py               # Auto-detection, parsing, SQLite management
│   ├── preprocessing.py            # Butterworth filtering, smoothing, normalization
│   ├── feature_engineering.py       # Time/frequency domain feature extraction
│   ├── tool_wear_model.py           # Scikit-learn regression model wrapper
│   ├── anomaly_detection.py         # Isolation Forest & explainable attribution
│   ├── rul_prediction.py            # Explainable RUL estimation engine
│   ├── digital_twin_view.py         # Dynamic SVG CNC milling machine state visualizer
│   ├── simulation.py                # What-if scenario simulation engine
│   └── recommendations.py           # Prescriptive maintenance decision engine
│
└── utils/                      # Configurations and constants
    └── config.py                    # Physical parameters, thresholds, and color palettes
```

---

## 👥 Authors & Academic Attribution

- **Student:** Yashmithha Prakash
- **Project ID:** PRJ_422
- **Institution:** Presidency University, Bengaluru
- **Department:** School of Computer Science and Engineering
- **Program:** Mini Project (CSE7102) Review-1
- **Supervisor:** Ms. Megha Karil, Department of Computer Science and Engineering
