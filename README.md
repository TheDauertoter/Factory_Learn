# UI Course DLBDSMTP01 – Project: From Model to Production

## Task 1: Anomaly Detection in an IoT Setting

---

## Python Files



### `Stream_and_Service`

Runs the complete IoT anomaly detection pipeline.

**Features:**
- Simulates IoT sensor data as a continuous data stream
- Sends incoming sensor data to the prediction service
- Performs real-time anomaly detection using the trained model
- Continuously outputs predictions until stopped

**Usage:**

```bash
python Stream_and_Servic.py
```

Stop the application at any time using:

```text
Ctrl + C
```

### `Evaluation.py`

Runs 25 Performance Tests for the HST Model and outputs Average Metrics

**Usage:**

```bash
python Evaluation.py
```

---

## Requirements

Install the required Python packages before running the project:

```bash
pip install -r requirements.txt
```

---

