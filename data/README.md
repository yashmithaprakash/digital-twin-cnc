# CNC Sensor Datasets & Telemetry Storage (PRJ_422)

This directory houses the milling process datasets and the local SQLite audit database used by the Digital Twin application.

## 1. Automatic Fallback & Realistic Synthetic Data
If real experimental data has not been provided, the application automatically synthesizes a high-fidelity dataset (`sample_cnc_data.csv`) adhering strictly to the physics of high-speed end-milling of Inconel 718.

### Data Attributes:
- `cut_id`: Sequential milling pass identifier (1 to 315 cuts).
- `spindle_speed_rpm`: Nominal spindle rotation speed (~10,400 RPM).
- `feed_rate_mm_min`: Workpiece feed velocity (~1,555 mm/min).
- `depth_of_cut_mm`: Axial depth of cut (~1.5 mm).
- `coolant`: Coolant delivery mode (`Flood`, `Mist`, `Dry`).
- `force_x`, `force_y`, `force_z`: 3-axis dynamic cutting forces (N) measured via dynamometer.
- `force_resultant`: $F_{res} = \sqrt{F_x^2 + F_y^2 + F_z^2}$.
- `vibration_x`, `vibration_y`, `vibration_z`: 3-axis accelerometer signals (g).
- `vibration_rms`: Root-mean-square vibration level across all axes.
- `ae_rms`: Acoustic Emission root-mean-square amplitude (V).
- `flank_wear_um`: Ground-truth tool flank wear (VB in micrometers, µm).
- `tool_health_pct`: Standardized health score (100% brand new down to 0% at replacement threshold).
- `rul_cuts`: Ground-truth remaining milling cuts until replacement criterion ($VB \ge 195\mu m$).
- `is_anomaly`: Flag indicating synthetic injection of workpiece hard inclusions, chatter resonance, or micro-chipping.
- `anomaly_type`: Descriptive diagnosis of detected irregularity.

---

## 2. PHM Society 2010 CNC Milling Dataset Compatibility
You can upload genuine experimental CSV files from the PHM 2010 Challenge. The application automatically maps:
- `smcAC`, `smcDC` or `force_x`, `force_y`, `force_z` $\rightarrow$ Force channels
- `vib_table`, `vib_spindle` or `vibration_x`, `vibration_y` $\rightarrow$ Vibration channels
- `AE_table`, `AE_spindle` or `ae_rms` $\rightarrow$ Acoustic Emission
- `flank_wear` or `VB` $\rightarrow$ Flank Wear (µm)

---

## 3. SQLite Storage (`digital_twin_cnc.db`)
Stores:
- `telemetry_logs`: Streaming sensor and prediction records.
- `maintenance_logs`: Service history, technician actions, and tool exchanges.
- `anomaly_alerts`: Historical record of detected anomaly events.
