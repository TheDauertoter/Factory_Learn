from contextlib import asynccontextmanager
from collections import deque
from threading import Lock

from fastapi import FastAPI
from pydantic import BaseModel
from pyod.models.iforest import IForest

import numpy as np


# ==================================================
# CONFIG
# ==================================================

FEATURES = [
    "temperature",
    "humidity",
    "noise_level"
]

INITIAL_TRAIN_SIZE = 3000
WINDOW_SIZE = 3000
DRIFT_CHECK_INTERVAL = 300

DRIFT_THRESHOLD = 0.10  # 10%

model = IForest(
    contamination=0.05,
    random_state=42
)

model_lock = Lock()

baseline_stats = {}

normal_buffer = deque(maxlen=WINDOW_SIZE)

normal_count = 0
anomaly_count = 0


# ==================================================
# REQUEST MODEL
# ==================================================

class SensorData(BaseModel):
    temperature: float
    humidity: float
    noise_level: float


# ==================================================
# TRAINING DATA
# ==================================================

def generate_normal_training_data(size=INITIAL_TRAIN_SIZE):

    return np.column_stack([
        np.random.normal(70, 2, size),
        np.random.normal(60, 3, size),
        np.random.normal(30, 2, size)
    ])


# ==================================================
# STATISTICS
# ==================================================

def calculate_stats(X):

    stats = {}

    for i, feature in enumerate(FEATURES):

        stats[feature] = {
            "mean": float(np.mean(X[:, i])),
            "std": float(np.std(X[:, i]))
        }

    return stats


def drift_detected(new_stats):

    for feature in FEATURES:

        old_mean = baseline_stats[feature]["mean"]
        old_std = baseline_stats[feature]["std"]

        new_mean = new_stats[feature]["mean"]
        new_std = new_stats[feature]["std"]

        mean_change = abs(new_mean - old_mean) / abs(old_mean)

        std_change = (
            abs(new_std - old_std)
            / max(old_std, 1e-6)
        )

        if mean_change > DRIFT_THRESHOLD:

            print(
                f"[DRIFT] {feature} mean "
                f"changed {mean_change*100:.2f}%"
            )

            return True

        if std_change > DRIFT_THRESHOLD:

            print(
                f"[DRIFT] {feature} std "
                f"changed {std_change*100:.2f}%"
            )

            return True

    return False


# ==================================================
# TRAINING
# ==================================================

def train_model(X):

    global baseline_stats

    with model_lock:
        model.fit(X)

    baseline_stats = calculate_stats(X)

    print("\n=== MODEL TRAINED ===")

    for feature in FEATURES:

        print(
            f"{feature}: "
            f"mean={baseline_stats[feature]['mean']:.2f} "
            f"std={baseline_stats[feature]['std']:.2f}"
        )

    print("=====================\n")


def initial_training():

    X = generate_normal_training_data()

    train_model(X)


# ==================================================
# RETRAINING
# ==================================================

def check_for_retraining():

    if len(normal_buffer) < WINDOW_SIZE:
        return

    X = np.array(normal_buffer)

    new_stats = calculate_stats(X)

    print(
        f"\nDrift check "
        f"(normal={normal_count}, "
        f"anomaly={anomaly_count})"
    )

    if drift_detected(new_stats):

        print("Retraining model...")

        train_model(X)

    else:

        print("No significant drift detected.\n")


# ==================================================
# PREDICTION
# ==================================================

def predict_one(data):

    global normal_count
    global anomaly_count

    x = np.array([
        [
            data["temperature"],
            data["humidity"],
            data["noise_level"]
        ]
    ])

    with model_lock:
        pred = model.predict(x)[0]

    # PYOD:
    # 0 = normal
    # 1 = anomaly

    if pred == 0:

        normal_count += 1

        normal_buffer.append([
            data["temperature"],
            data["humidity"],
            data["noise_level"]
        ])

        if normal_count % DRIFT_CHECK_INTERVAL == 0:

            check_for_retraining()

        return "NORMAL"

    anomaly_count += 1

    return "ANOMALY"


# ==================================================
# FASTAPI LIFESPAN
# ==================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    print("Training initial model...")

    initial_training()

    print("API ready.")

    yield

    print("Shutdown.")


app = FastAPI(
    title="Anomaly Detection API",
    lifespan=lifespan
)


# ==================================================
# ENDPOINTS
# ==================================================

@app.post("/predict")
def predict(data: SensorData):

    prediction = predict_one(
        data.model_dump()
    )

    return {
        "prediction": prediction
    }


@app.get("/stats")
def stats():

    return {
        "normal_count": normal_count,
        "anomaly_count": anomaly_count,
        "buffer_size": len(normal_buffer),
        "baseline_stats": baseline_stats
    }


# ==================================================
# RUN
# ==================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    )