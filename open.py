import pickle

with open("artifacts/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("artifacts/encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

print(scaler)
print(encoder)
