import random
import numpy as np
from pyod.models.iforest import IForest


#  make samples
def normal():
    return [
        random.normalvariate(70, 2),
        random.normalvariate(60, 3),
        random.normalvariate(30, 2),
    ]

def anomaly():
    return [
        random.normalvariate(85, 2),
        random.normalvariate(45, 3),
        random.normalvariate(55, 2),
    ]

# model and training

model = IForest(contamination=0.05, random_state=42)
X_train = [normal() for _ in range(300)]
model.fit(np.array(X_train))

# stream data into model after training

y_true = []
y_pred = []
scores = []

for i in range(600):

    if random.random() < 0.2:
        x = anomaly()
        true = 1
    else:
        x = normal()
        true = 0

    x_np = np.array(x).reshape(1, -1)

    score = model.decision_function(x_np)[0]
    pred = model.predict(x_np)[0]

    y_true.append(true)
    y_pred.append(pred)
    scores.append(score)

    pred_label = "ANOMALY" if pred == 1 else "NORMAL"
    true_label = "ANOMALY" if true == 1 else "NORMAL"

    print(
        f"{i:03d} | "
        f"T={x[0]:.1f} H={x[1]:.1f} S={x[2]:.1f} "
        f"| score= {score:.4f} "
        f"| pred={pred_label} true={true_label}"
    )


# statistics

y_true = np.array(y_true)
y_pred = np.array(y_pred)
scores = np.array(score)

TP = np.sum((y_true == 1) & (y_pred == 1))
TN = np.sum((y_true == 0) & (y_pred == 0))
FP = np.sum((y_true == 0) & (y_pred == 1))
FN = np.sum((y_true == 1) & (y_pred == 0))

accuracy = (TP + TN) / len(y_true)
precision = TP / (TP + FP + 1e-9)
recall = TP / (TP + FN + 1e-9)


print("\n================ FINAL MODEL STATS ================\n")

print(f"Accuracy:  {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall:    {recall:.3f}")


print("\nConfusion Matrix:")
print("-------------------")
print(f"TP={TP} FP={FP}")
print(f"FN={FN} TN={TN}")
print("-------------------")


