# Sample Data

This folder contains **one complete example dataset group** for learning and testing HydroSpecFit.

The three Excel files in this folder belong to the **same experiment** and should be used together in the HydroSpecFit workflow.

---

## Experimental context

This example dataset was obtained from **Zn deposition on an Au-coated quartz crystal**.

The electrolyte parameters for this example are based on measurements in:

**2 M ZnSO4 + 0.5 mass% PEG400**

These settings are provided as the recommended starting parameters for reproducing the example workflow in HydroSpecFit.

---

## Included files

### 1. `quartz_in_air_example.xlsx`
Reference quartz signal measured in air.

### 2. `qcmd_example.xlsx`
Raw multiharmonic QCM-D time-series data from the same experiment.

### 3. `echem_example.xlsx`
Electrochemical time-series data corresponding to the same experiment.

---

## How these files are used together

These three files represent one complete experimental dataset group:

- `quartz_in_air_example.xlsx` provides the air reference signal
- `qcmd_example.xlsx` provides the QCM-D measurement data
- `echem_example.xlsx` provides the electrochemical data for synchronization, theoretical frequency calculation, and cycle detection

In HydroSpecFit, they should be loaded together as one analysis set.

---

## File format requirements

### 1. Quartz in air file

**Example file:** `quartz_in_air_example.xlsx`

This file provides the stabilized quartz reference signal in air.

#### Required structure
- Excel format: `.xlsx` or `.xls`
- One worksheet
- One row of stabilized reference values
- Columns follow the style:

```text
f3_1 (Hz), D3_1 (ppm), f5_1 (Hz), D5_1 (ppm), ...
````

#### Example columns

```text
f3_1 (Hz)
D3_1 (ppm)
f5_1 (Hz)
D5_1 (ppm)
f7_1 (Hz)
D7_1 (ppm)
f9_1 (Hz)
D9_1 (ppm)
f11_1 (Hz)
D11_1 (ppm)
```

#### Notes

* Frequency values are in **Hz**
* Dissipation values are in **ppm**
* The software reads the **last valid value** in each reference column
* Odd harmonics should be provided consistently

---

### 2. QCM-D data file

**Example file:** `qcmd_example.xlsx`

This file contains the raw multiharmonic QCM-D measurement data.

#### Required structure

* Excel format: `.xlsx` or `.xls`
* One worksheet
* Time-series rows
* The first column is time
* Frequency and dissipation columns are provided for each harmonic

#### Example columns

```text
time
f3
D3
f5
D5
f7
D7
f9
D9
f11
D11
```

#### Column meaning

* `time`: time in seconds
* `f3`, `f5`, `f7`, ...: resonance frequency at each harmonic
* `D3`, `D5`, `D7`, ...: dissipation at each harmonic

#### Notes

* Time should increase sequentially
* Frequency values should be in **Hz**
* Dissipation values should be in **ppm**
* Harmonics should be paired correctly, for example `f3` with `D3`, `f5` with `D5`
* The software automatically detects odd harmonics from the frequency column names

---

### 3. Electrochemical data file

**Example file:** `echem_example.xlsx`

This file contains the electrochemical data corresponding to the same experiment.

#### Required structure

* Excel format: `.xlsx` or `.xls`
* One worksheet
* Time-series rows

#### Example columns

```text
time/s
(Q-Qo)/mC
<Ewe/V>
I/mA
```

#### Column meaning

* `time/s`: time in seconds
* `(Q-Qo)/mC`: charge relative to the initial value, in mC
* `<Ewe/V>`: working electrode potential, in V
* `I/mA`: current, in mA

#### Notes

* `time/s` is used as the electrochemical time axis
* `(Q-Qo)/mC` is used for theoretical frequency calculation
* `<Ewe/V>` is used for cycle detection and plotting
* `I/mA` is included as experimental current information

---

## Recommended loading order in HydroSpecFit

Load the three files in the following order:

1. `qcmd_example.xlsx` as **QCM-D Data**
2. `echem_example.xlsx` as **EChem Data**
3. `quartz_in_air_example.xlsx` as **Quartz in Air**

Then continue with parameter checking, time synchronization if needed, and Auto-Cycles Optimization.

---

## Recommended parameters for this example

The following parameter values are recommended for reproducing this example dataset in HydroSpecFit.

| Parameter             | Recommended value | Description                                                      |
| --------------------- | ----------------: | ---------------------------------------------------------------- |
| Quartz Density        |       2.648 g/cm³ | Density of quartz                                                |
| Quartz Viscosity      |       2.947e10 Pa | Quartz mechanical parameter used in the model                    |
| Sensitivity           |    56.5 Hz·cm²/ug | Sauerbrey sensitivity factor                                     |
| Molecular Weight      |          65 g/mol | Molecular weight of Zn                                           |
| Number of Electrons   |                 2 | Electron transfer number for Zn²⁺/Zn                             |
| Active Area           |         0.785 cm² | Electrode active area                                            |
| Liquid Density        |      1.3228 g/cm³ | Electrolyte density for 2 M ZnSO4 + 0.5 mass% PEG400             |
| Ref. Liquid Viscosity |       0.0033 Pa·s | Reference electrolyte viscosity for 2 M ZnSO4 + 0.5 mass% PEG400 |
| Coverage (θ)          |               1.0 | Surface coverage factor                                          |

### Notes on the recommended parameters

* These values are intended as **starting parameters** for this example dataset
* The electrochemical parameters correspond to **Zn deposition**
* The electrolyte density and viscosity values correspond to **2 M ZnSO4 + 0.5 mass% PEG400**
* The substrate in this example is an **Au-coated quartz crystal**

---

## Important remarks

* These files are provided as **one complete example dataset group**
* They are intended for workflow demonstration and learning
* Users are encouraged to follow the same column naming style and overall file structure when preparing their own datasets
* If different naming conventions are used, HydroSpecFit may fail to detect the required columns correctly
* For best compatibility, keep the formatting close to these example files

---

