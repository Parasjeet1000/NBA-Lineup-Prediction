import os
import pandas as pd
import numpy as np
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

def create_position_dataset(data, missing_position):
    print(f"Preparing dataset by removing {missing_position} from input features...")
    # Copy data to avoid modifying original
    position_data = data.copy()

    # Define column groups
    home_players = [f'home_{i}' for i in range(5)]
    away_players = [f'away_{i}' for i in range(5)]
    model_features = ['season', 'home_team', 'away_team', 'starting_min']

    # Remove the missing position from input features
    input_home_players = [p for p in home_players if p != missing_position]
    input_features = model_features + input_home_players + away_players
    target_column = missing_position

    # Store original team names and game IDs
    original_home_teams = position_data['home_team'].copy()
    original_away_teams = position_data['away_team'].copy()
    game_ids = position_data['game'].copy()

    # Select features and target
    X = position_data[input_features].copy()
    y = position_data[target_column].copy()

    print(f"Dataset for missing position {missing_position} prepared with {X.shape[0]} samples.")
    return X, y, input_features, target_column, game_ids, original_home_teams, original_away_teams


def train_position_model(X, y, input_features, target_column, game_ids, original_home_teams, original_away_teams):
    print(f"\nTraning for: {target_column}...")

    # Encode categorical variables
    categorical_columns = ['home_team', 'away_team'] + [col for col in X.columns if 'home_' in col or 'away_' in col]
    label_encoders = {}
    print("Encoding categorical variables...")
    for col in categorical_columns:
        label_encoders[col] = LabelEncoder()
        X[col] = label_encoders[col].fit_transform(X[col])
    print("Categorical variables encoded.")

    # Encode target variable
    print("Encoding target variable...")
    label_encoders[target_column] = LabelEncoder()
    y = label_encoders[target_column].fit_transform(y)
    print("Target variable encoded.")

    # Split the data with reference data
    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test, game_train, game_test, home_team_train, home_team_test, away_team_train, away_team_test = train_test_split(
        X, y, game_ids, original_home_teams, original_away_teams, test_size=0.2, random_state=42
    )
    print(f"Data split: {len(X_train)} training samples and {len(X_test)} testing samples.")

    # Scale only starting_min
    print("Scaling the 'starting_min' feature...")
    scaler = StandardScaler()
    X_train['starting_min'] = scaler.fit_transform(X_train[['starting_min']])
    X_test['starting_min'] = scaler.transform(X_test[['starting_min']])
    print("Feature scaling complete.")

    # --- Hyperparameter Tuning with RandomizedSearchCV ---
    print("Starting hyperparameter tuning with RandomizedSearchCV...")
    from sklearn.model_selection import RandomizedSearchCV  # Removed train_test_split from here
    from scipy.stats import randint

    # Define parameter distributions
    param_distributions = {
        'n_estimators': randint(100, 250),
        'max_depth': [10, 15, 20, None],
        'min_samples_split': randint(2, 10),
        'min_samples_leaf': randint(1, 5),
        'max_features': ['auto', 'sqrt', 'log2']
    }

    random_search = RandomizedSearchCV(
        estimator=RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced'),
        param_distributions=param_distributions,
        n_iter=20,  # Number of parameter settings to sample
        cv=3,       # Using 3-fold cross-validation to reduce memory usage
        scoring='accuracy',
        n_jobs=-1,
        random_state=42
    )

    # Optionally, tune on a subset of the training data to reduce memory load:
    X_tune, _, y_tune, _ = train_test_split(X_train, y_train, test_size=0.8, random_state=42)
    random_search.fit(X_tune, y_tune)
    print("Best parameters on subset:", random_search.best_params_)

    best_params = random_search.best_params_

    print("Initializing and training the Random Forest model with tuned hyperparameters...")
    rf_model = RandomForestClassifier(
        n_estimators=best_params['n_estimators'],
        max_depth=best_params['max_depth'],
        min_samples_split=best_params['min_samples_split'],
        min_samples_leaf=best_params['min_samples_leaf'],
        max_features=best_params['max_features'],
        class_weight='balanced',
        n_jobs=-1,
        random_state=42
    )
    rf_model.fit(X_train, y_train)
    print("Model training complete.")

    rf_predictions = rf_model.predict(X_test)

    # Create a results DataFrame
    results = pd.DataFrame({
        'game': game_test,
        'season': X_test['season'],
        'home_team': home_team_test,  # Original team names
        'away_team': away_team_test,  # Original team names
        'actual_player': label_encoders[target_column].inverse_transform(y_test),
        'predicted_player': label_encoders[target_column].inverse_transform(rf_predictions)
    })

    # Calculate feature importances
    feature_importances = pd.DataFrame(
        rf_model.feature_importances_,
        index=input_features,
        columns=['importance']
    ).sort_values('importance', ascending=False)

    # Compute the accuracy and print it
    accuracy = accuracy_score(y_test, rf_predictions)
    print(f"Completed training for {target_column}. Accuracy achieved: {accuracy:.4f}\n")

    return rf_model, label_encoders, scaler, results, feature_importances, accuracy




def save_all_results(results_all, models, output_filename='nba_predictions_results_all.xlsx'):
    print(f"Saving all results to {output_filename}...")
    with pd.ExcelWriter(output_filename, mode='w') as writer:
        for position, results in results_all.items():
            # Save predictions
            sheet_name_pred = f'{position}_predictions'
            results.to_excel(writer, sheet_name=sheet_name_pred, index=False)
            # Save feature importances; models[position]['feature_importances'] holds the dataframe
            sheet_name_imp = f'{position}_feature_importance'
            models[position]['feature_importances'].to_excel(writer, sheet_name=sheet_name_imp)
    print(f"All results have been successfully saved to {output_filename}.")


# Define the directory and file pattern for matchup files
matchup_files = glob.glob("Datasets/matchups-*.csv")
print(f"Found {len(matchup_files)} matchup file(s). Beginning file load...")

data_frames = []
for file in matchup_files:
    try:
        print(f"Loading file: {file}")
        df = pd.read_csv(file)
        data_frames.append(df)
    except Exception as e:
        print(f"Error loading {file}: {e}")

if not data_frames:
    print("No data files were loaded. Exiting the program.")
    exit()

# Combine all dataframes, drop NaNs and duplicates
combined_data = pd.concat(data_frames, ignore_index=True).dropna()
combined_data.drop_duplicates(inplace=True)
print("Data combined. NaN values and duplicates have been removed.")

# Filter for winning home team samples
combined_data = combined_data[combined_data['outcome'] == 1]
print(f"Filtered dataset for winning home teams: {len(combined_data)} record(s) found.")
print(f"Seasons included in the dataset: {combined_data['season'].unique().tolist()}")

# Train models for each possible missing home position
models = {}
results_all = {}
for position in [f'home_{i}' for i in range(5)]:
    print(f"\n--- Starting model training for missing position: {position} ---")
    X, y, input_features, target_column, game_ids, original_home_teams, original_away_teams = create_position_dataset(
        combined_data, position)
    model, encoders, scaler, results, importances, accuracy = train_position_model(
        X, y, input_features, target_column, game_ids, original_home_teams, original_away_teams
    )

    models[position] = {
        'model': model,
        'encoders': encoders,
        'scaler': scaler,
        'feature_importances': importances,
        'accuracy': accuracy
    }
    results_all[position] = results
    print(f"--> Finished training for {position} with an accuracy of {accuracy:.4f}")

# Save all results into one Excel file
save_all_results(results_all, models)
print("\nAll models have been trained and the output has been saved to a Excel file called nba_predictions_results_all.xlsx" )