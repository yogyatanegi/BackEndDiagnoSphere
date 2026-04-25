import pickle

# Load the old pickle model
model = pickle.load(open("models/parkinsons.pkl", "rb"))

# Save the model in the new stable XGBoost JSON format
model.save_model("models/parkinsons.json")

print("✅ Parkinsons model converted successfully")
