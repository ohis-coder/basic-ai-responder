import numpy as np

# Define logits for simplicity (these can be any values you want to model)
logits = np.array([0.41, 0.8, 0.74, 0.23, -0.12, -0.2, -0.14])

# Softmax function with temperature
def softmax(x, temp=1.0):
    exps = np.exp((x - np.max(x)) / temp)  # Temperature adjustment
    return exps / exps.sum()

# Function to generate a reply based on logits and temperature
def reply(logits, temp=1.0):
    probabilities = softmax(logits, temp)
    print("🔢 Logits:", logits)
    print("📊 Probabilities:", probabilities)
    
    # Choose a word based on the probabilities
    words = ["Good", "morning", "to", "you", "sir", "madam", "AI"]
    return np.random.choice(words, p=probabilities)

# Run the model with temperature variation
for temp in [0.5, 1.0, 1.5]:  # Testing different temperatures
    print(f"\nTemp = {temp}")
    print("🧠 AI says:", reply(logits, temp))
