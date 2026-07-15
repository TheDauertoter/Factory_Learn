# UI Course DLBDSMTP01 – Project: From Model to Production

## Task 1: Anomaly Detection in an IoT Setting
**Focus:** Stream Processing

This repository contains the implementation of an anomaly detection pipeline for an IoT scenario using an **Isolation Forest** model. It includes both a standalone machine learning evaluation and a stream-processing simulation with a prediction service.

---

## Repository Structure

### `pure_model.py`

Standalone implementation for training and evaluating the machine learning model.

**Features:**
- Trains an **Isolation Forest** on **300 training samples**.
- Evaluates the trained model on **600 labeled test samples**.
- Prints common evaluation metrics (e.g., accuracy, precision, recall, F1-score).

**Usage:**

```bash
python pure_model.py
```

---

### `run_all.py`

Runs the complete IoT anomaly detection pipeline.

**Features:**
- Simulates IoT sensor data as a continuous data stream.
- Sends incoming sensor data to the prediction service.
- Performs real-time anomaly detection using the trained model.
- Continuously outputs predictions until stopped.

**Usage:**

```bash
python run_all.py
```

Stop the application at any time using:

```text
Ctrl + C
```

---

## Requirements

Install the required Python packages before running the project:

```bash
pip install -r requirements.txt
```

---

## Project Overview

The project demonstrates the transition from a standalone machine learning model to a production-like streaming application by combining:

- Real-time sensor data simulation
- Stream processing
- Isolation Forest anomaly detection
- Prediction as a service

This setup illustrates how machine learning models can be integrated into an IoT environment for continuous anomaly detection.
