
import random

import numpy as np
from river import anomaly


# normal data
def normal():
    return {
        "temperature": round(random.normalvariate(70, 2) / 100, 2),
        "humidity": round(random.normalvariate(60, 2) / 100, 2),
        "noise_level": round(random.normalvariate(30, 2) / 100, 2),
    }


# abnormal data
def abnormal():
    return {
        "temperature": round(random.normalvariate(80, 3) / 100, 2),
        "humidity": round(random.normalvariate(70, 3) / 100, 2),
        "noise_level": round(random.normalvariate(36, 3) / 100, 2),
    }


# 25 test with always a new model
# 250 normal as trainigdata
# 200 normal, 50 abnormal as testdata
# no further learning after scoring, just testing

results = []

for i in range(25):
    
    model = anomaly.HalfSpaceTrees(
        n_trees=25,
        height=12,
        window_size=250,
        seed=42
    )

    # train on normal
    for _ in range(250):
        model.learn_one(normal())

    # score
    normal_scores = [model.score_one(normal()) for _ in range(200)]
    anomaly_scores = [model.score_one(abnormal()) for _ in range(50)]

    threshold = np.percentile(normal_scores, 95) # threshold for label

    tn = sum(s < threshold for s in normal_scores)
    fp = sum(s >= threshold for s in normal_scores)
    fn = sum(s < threshold for s in anomaly_scores)
    tp = sum(s >= threshold for s in anomaly_scores)

    results.append((tn, fp, fn, tp))
    print("Test", i+1 , "done")


# average results
tn = sum(r[0] for r in results) / 25
fp = sum(r[1] for r in results) / 25
fn = sum(r[2] for r in results) / 25
tp = sum(r[3] for r in results) / 25



accuracy = (tp + tn) / 250

if tp + fp == 0:
    precision = 0
else:
    precision = tp / (tp + fp)

if tp + fn == 0:
    recall = 0
else:
    recall = tp / (tp + fn)


print("\n")
print("Averages of 25 Tests with Prelabeld Data ")
print("250 normal datapoints in Train ")
print ("200 normal and 50 anormal datapoints in Test")
print("\n#############################################")
print("\n")
print(f"Average Accuracy:  {accuracy:.3f}")
print(f"Average Precision: {precision:.3f}")
print(f"Average Recall:    {recall:.3f}")

print("\nAverage Confusion Matrix:")
print(f"[[{tn:.2f}, {fp:.2f}],")
print(f" [{fn:.2f}, {tp:.2f}]]")
print("\n")
