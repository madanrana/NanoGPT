import json
import matplotlib.pyplot as plt

with open("logs/loss_history.json") as f:

    loss = json.load(f)

plt.plot(loss)

plt.xlabel("Training Step")
plt.ylabel("Loss")

plt.title("Training Loss Curve")

plt.show()