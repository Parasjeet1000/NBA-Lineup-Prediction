import pandas as pd

# ==================== Step 1: Load Test Data and Test Labels ====================
# Update file paths if necessary
test_file_path = "NBA_test(2007-2009).csv"
test_labels_file_path = "NBA_test_labels(2007-2009).csv"

# Load the test dataset and test labels
test_data = pd.read_csv(test_file_path)
test_labels = pd.read_csv(test_labels_file_path)

# ==================== Step 2: Load Predictions and Feature Importance from Excel ====================
excel_file_path = "nba_predictions_results_all.xlsx"
xls = pd.ExcelFile(excel_file_path)

# --- Load Predictions ---
# The Excel file contains a separate sheet for each home position's predictions.
# For each test case, only one home position is missing.
predictions_list = []
home_positions = ['home_0', 'home_1', 'home_2', 'home_3', 'home_4']

for pos in home_positions:
    sheet_name = f"{pos}_predictions"
    if sheet_name in xls.sheet_names:
        df_pred = pd.read_excel(xls, sheet_name=sheet_name)
        # Optionally, add a column to indicate which position was missing
        df_pred["missing_position"] = pos
        predictions_list.append(df_pred)

# Combine all predictions into one DataFrame (each test case should appear only once)
combined_predictions = pd.concat(predictions_list, ignore_index=True)

# --- Load Feature Importance ---
fi_list = []
for pos in home_positions:
    sheet_name = f"{pos}_feature_importance"
    if sheet_name in xls.sheet_names:
        fi_df = pd.read_excel(xls, sheet_name=sheet_name)
        fi_df["position"] = pos
        fi_list.append(fi_df)

combined_feature_importance = pd.concat(fi_list, ignore_index=True)

# ==================== Step 3: Merge Predictions with Test Labels ====================
# Assume that the test labels file contains the removed player's names in a column named "removed_value".
# We add these as the actual removed player to the predictions DataFrame.
# (Make sure that the order of test labels matches the order of test cases in combined_predictions.)
combined_predictions["actual_removed_player"] = test_labels["removed_value"]

# ==================== Step 4: Calculate Accuracy and Analyze Predictions ====================
# Create a column to flag correct predictions
combined_predictions["correct_prediction"] = combined_predictions["predicted_player"] == combined_predictions["actual_removed_player"]

# Calculate overall accuracy (percentage)
accuracy = combined_predictions["correct_prediction"].mean() * 100

# Count correct and incorrect predictions
correct_predictions = combined_predictions["correct_prediction"].sum()
incorrect_predictions = len(combined_predictions) - correct_predictions

# ------------------- Distribution Analysis -------------------
# Top predicted players
predicted_distribution = combined_predictions["predicted_player"].value_counts().head(10)
predicted_distribution_df = pd.DataFrame({
    'Predicted Player': predicted_distribution.index,
    'Count': predicted_distribution.values
})

# Top actual removed players
actual_distribution = combined_predictions["actual_removed_player"].value_counts().head(10)
actual_distribution_df = pd.DataFrame({
    'Actual Removed Player': actual_distribution.index,
    'Count': actual_distribution.values
})

# ==================== Step 5: Analyze Matches Per Season ====================
matches_per_year = test_data['season'].value_counts().sort_index()
matches_per_year_df = pd.DataFrame({
    'Season': matches_per_year.index,
    'Number of Matches': matches_per_year.values
})
average_matches = matches_per_year.mean()

# ==================== Step 6: Save Results ====================
combined_predictions.to_csv("nba_test_predictions_vs_actual.csv", index=False)
matches_per_year_df.to_csv("nba_matches_per_year.csv", index=False)
predicted_distribution_df.to_csv("nba_top_predicted_players.csv", index=False)
actual_distribution_df.to_csv("nba_top_actual_removed_players.csv", index=False)
combined_feature_importance.to_csv("nba_feature_importance.csv", index=False)

# ==================== Step 7: Display Key Results ====================
# If using Jupyter Notebook, you can use display() or your ace_tools.
from IPython.display import display

print("=== Predictions vs Actual (First 10 Rows) ===")
display(combined_predictions.head(10))

print("\n=== Matches Per Season ===")
display(matches_per_year_df)

print("\n=== Top Predicted Players ===")
display(predicted_distribution_df)

print("\n=== Top Actual Removed Players ===")
display(actual_distribution_df)

print("\n=== Feature Importance (First 10 Rows) ===")
display(combined_feature_importance.head(10))

# Print overall accuracy and match stats
print(f"\nModel Accuracy: {accuracy:.2f}%")
print(f"Correct Predictions: {correct_predictions}")
print(f"Incorrect Predictions: {incorrect_predictions}")
print(f"Total Test Cases: {len(combined_predictions)}")
print(f"Average number of matches per season: {average_matches:.2f}")
