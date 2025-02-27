import pandas as pd
import numpy as np
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib


# Load all datasets from 2007-2015
matchup_files = glob.glob("Datasets/matchups-*.csv")  # Ensure correct path
print(f"Found dataset files: {matchup_files}")

# Check if files exist
if not matchup_files:
    raise FileNotFoundError("No dataset files found. Check dataset path and ensure CSV files exist.")

all_data = []

for file in matchup_files:
    print(f"Loading file: {file}")
    df = pd.read_csv(file)

    # Check if file is empty
    if df.empty:
        print(f"Warning: {file} is empty!")

    all_data.append(df)

# Ensure data was loaded before concatenation
if not all_data:
    raise ValueError("No valid data was loaded. Check dataset files.")

# Combine all data
df = pd.concat(all_data, ignore_index=True)
print("Dataset successfully combined!")

allowed_columns = [
    "game", "season", "home_team", "away_team", "starting_min",
    "home_0", "home_1", "home_2", "home_3", "home_4",
    "away_0", "away_1", "away_2", "away_3", "away_4", "outcome"
]
df = df[allowed_columns]

# Drop missing values
df = df.dropna()

# Encode categorical variables
team_encoder = LabelEncoder()
df["home_team"] = team_encoder.fit_transform(df["home_team"])
df["away_team"] = team_encoder.transform(df["away_team"])

player_columns = ["home_0", "home_1", "home_2", "home_3", "home_4", "away_0", "away_1", "away_2", "away_3", "away_4"]
player_encoder = LabelEncoder()
df[player_columns] = df[player_columns].apply(lambda col: player_encoder.fit_transform(col))

# Define features and target
feature_cols = ["season", "home_team", "away_team", "starting_min", "home_0", "home_1", "home_2", "home_3", "away_0", "away_1", "away_2", "away_3", "away_4", "outcome"]
target_col = "home_4"

# Split data into train/test sets
X_train, X_test, y_train, y_test = train_test_split(df[feature_cols], df[target_col], test_size=0.2, random_state=42)

# Initialize and train Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42)
rf_model.fit(X_train, y_train)

# Predict on test set
y_pred = rf_model.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Random Forest Model Accuracy: {accuracy:.4f}")

# Save the model
joblib.dump(rf_model, "rf_nba_model.pkl")

# Function to make a prediction
def predict_fifth_player(season, home_team, away_team, starting_min, home_0, home_1, home_2, home_3, away_0, away_1, away_2, away_3, away_4, outcome):
    input_data = pd.DataFrame([[season, home_team, away_team, starting_min, home_0, home_1, home_2, home_3, away_0, away_1, away_2, away_3, away_4, outcome]],
                              columns=feature_cols)  # Ensure feature names match training data
    prediction = rf_model.predict(input_data)
    return player_encoder.inverse_transform(prediction)[0]

# Example usage
example_prediction = predict_fifth_player(2015, 1, 5, 12, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1)
print(f"Predicted Fifth Player: {example_prediction}")
