import random
import threading
import time
from contextlib import asynccontextmanager

import numpy as np
import requests
import uvicorn
from fastapi import FastAPI
from pyod.models.iforest import IForest


# model definition
features = ["temperature", "humidity", "noise_level"]

model = IForest(
    contamination=0.05,
    random_state=42
)


# generate normal sensor data
def normal_sample():
    return {
        "temperature": random.normalvariate(70, 2),
        "humidity": random.normalvariate(60, 3),
        "noise_level": random.normalvariate(30, 2),
    }


# train model with 300 normal data points
def train_model():

    X = [
        list(normal_sample().values())
        for _ in range(300)
    ]

    model.fit(np.array(X))

    print("Model trained - Starting API to Start Stream")


# prediction
def predict_one(data: dict):

    x = np.array([
        [data[f] for f in features]
    ])

    pred = model.predict(x)[0]

    label = "ANOMALY" if pred == 1 else "NORMAL"

    return label


# stream sensor data to /predict
def stream_loop():

    # wait until FastAPI is fully started
    time.sleep(2)

    while True:

        # generate sensor data
        data = normal_sample()

        try:

            r = requests.post(
                "http://127.0.0.1:8000/predict",
                json=data,
                timeout=1,
            )

            result = r.json()

            print(
                f"T={data['temperature']:.1f} "
                f"H={data['humidity']:.1f} "
                f"N={data['noise_level']:.1f} "
                f"-> {result['prediction']}"
            )

        except requests.RequestException as e:
            print("Request failed:", e)

        time.sleep(0.5)


# lifespan - manages FastAPI from start to end
@asynccontextmanager
async def lifespan(app: FastAPI):

    train_model()

    thread = threading.Thread(
        target=stream_loop,
        daemon=True,
    )

    thread.start()

    print("Stream started")

    yield

    print("Shutting down")


# FastAPI
app = FastAPI(lifespan=lifespan)


# Model as a Service
@app.post("/predict")
def predict(data: dict):

    label = predict_one(data)

    return {
        "prediction": label
    }


# run all, CTRL+C to stop
if __name__ == "__main__":

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        log_level="warning"
    )