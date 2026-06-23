from contextlib import asynccontextmanager
from fastapi import FastAPI
from pyod.models.iforest import IForest

import numpy as np
import random
import threading
import time
import requests
import uvicorn


# model definition
features = ["temperature", "humidity", "noise_level"]
model = IForest(contamination=0.05, random_state=42)


# data normal
def normal_sample():
    return {
        "temperature": random.normalvariate(70, 2),
        "humidity": random.normalvariate(60, 3),
        "noise_level": random.normalvariate(30, 2),
    }

# data anomaly
def anomaly_sample():
    return {
        "temperature": random.normalvariate(85, 2),
        "humidity": random.normalvariate(45, 3),
        "noise_level": random.normalvariate(55, 2),
    }

# train model with 300 normal data points
def train_model():
    X = [list(normal_sample().values()) for _ in range(300)]

    model.fit(np.array(X))

    print("Model trained - Starting API to Start Stream")


#prediction
def predict_one(data: dict):

    x = np.array([[data[f] for f in features]])

    pred = model.predict(x)[0]

    label = "ANOMALY" if pred == 1 else "NORMAL"

    
    return label

#stream sensor data to /predict

def stream_loop():

    # wait until FastAPI is fully started
    time.sleep(2)

    while True:

        data = (
            anomaly_sample()
            if random.random() < 0.1
            else normal_sample()
        )

        try:

            r = requests.post(
                "http://127.0.0.1:8000/predict",
                json=data,
                timeout=2,
            )

            result = r.json()

            print(
                f"T={data['temperature']:.1f} "
                f"H={data['humidity']:.1f} "
                f"N={data['noise_level']:.1f} "
                f"-> {result['prediction']}"
            )

        except Exception as e:
            print("Request failed:", e)

        time.sleep(0.5)


#lifespan - manages FastAPI

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


#FastAPI
app = FastAPI(lifespan=lifespan)


#when /predict is requested, run predict
@app.post("/predict")
def predict(data: dict):

    label = predict_one(data)

    return {
        "prediction": label
    }




#run all, STRG C to stop

if __name__ == "__main__":

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        log_level="warning" 
    )