import os
import pandas as pd
import numpy as np
import glob
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from scipy.stats import randint
import joblib

# ----- Function Definitions -----

def create_position_dataset(data, missing_position):
    """
    Create a dataset for a given missing position.
    If 'game' is not present in data, it creates a dummy game ID.
    """
    print(f"Preparing dataset by removing {missing_position} from input features...")
    position_data = data.copy()

    # Define column groups
    home_players = [f'home_{i}' for i in range(5)]
    away_players = [f'away_{i}' for i in range(5)]
    model_features = ['season', 'home_team', 'away_team', 'starting_min']

    # Remove the missing position from input features
    input_home_players = [p for p in home_players if p != missing_position]
    input_features = model_features + input_home_players + away_players
    target_column = missing_position

    # Get original team names and game IDs (if available)
    original_home_teams = position_data['home_team'].copy()
    original_away_teams = position_data['away_team'].copy()
    if 'game' in position_data.columns:
        game_ids = position_data['game'].copy()
    else:
        game_ids = pd.Series(range(len(position_data)))

    # Select features and target
    X = position_data[input_features].copy()
    y = position_data[target_column].copy()

    print(f"Dataset for missing position {missing_position} prepared with {X.shape[0]} samples.")
    return X, y, input_features, target_column, game_ids, original_home_teams, original_away_teams

def train_position_model(X, y, input_features, target_column, game_ids, original_home_teams, original_away_teams):
    """
    Trains a Random Forest model for the specified missing position.
    This version uses one-hot encoding for all categorical columns except 'starting_min'
    and the target column. The target is then label-encoded.
    """
    print(f"\nTraining for: {target_column}...")

    # Determine columns to one-hot encode: all columns except 'starting_min' and the target.
    categorical_cols = [col for col in X.columns if col != 'starting_min']
    onehot_cols = [col for col in categorical_cols if col != target_column]
    print("Performing one-hot encoding on columns:", onehot_cols)
    X = pd.get_dummies(X, columns=onehot_cols)
    # Store the final feature set for later reindexing
    input_features_updated = list(X.columns)

    # Encode target variable (do NOT one-hot encode target)
    print("Encoding target variable...")
    target_encoder = LabelEncoder()
    y_encoded = target_encoder.fit_transform(y)
    print("Target variable encoded.")

    # Split the data
    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test, game_train, game_test, home_team_train, home_team_test, away_team_train, away_team_test = train_test_split(
        X, y_encoded, game_ids, original_home_teams, original_away_teams, test_size=0.2, random_state=42
    )
    print(f"Data split: {len(X_train)} training samples and {len(X_test)} testing samples.")

    # Scale only the numerical feature 'starting_min'
    print("Scaling the 'starting_min' feature...")
    scaler = StandardScaler()
    X_train['starting_min'] = scaler.fit_transform(X_train[['starting_min']])
    X_test['starting_min'] = scaler.transform(X_test[['starting_min']])
    print("Feature scaling complete.")

    # Hyperparameter tuning with RandomizedSearchCV
    print("Starting hyperparameter tuning with RandomizedSearchCV...")
    param_distributions = {
        'n_estimators': randint(100, 300),
        'max_depth': [5, 10, 15, None],
        'min_samples_split': randint(2, 10),
        'min_samples_leaf': randint(1, 5),
        'max_features': ['auto', 'sqrt', 'log2']
    }
    random_search = RandomizedSearchCV(
        estimator=RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced'),
        param_distributions=param_distributions,
        n_iter=20,
        cv=3,
        scoring='accuracy',
        n_jobs=-1,
        random_state=42
    )
    # Tune on a subset to reduce memory load
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
    # (Note: since season, home_team, etc. are now one-hot encoded, we cannot directly recover their original values;
    # we still include game IDs and the original home/away teams stored earlier.)
    results = pd.DataFrame({
        'game': game_test,
        'actual_player': target_encoder.inverse_transform(y_test),
        'predicted_player': target_encoder.inverse_transform(rf_predictions)
    })

    # Feature importances (using the one-hot encoded feature names)
    feature_importances = pd.DataFrame(
        rf_model.feature_importances_,
        index=input_features_updated,
        columns=['importance']
    ).sort_values('importance', ascending=False)

    # Compute accuracy
    accuracy = accuracy_score(y_test, rf_predictions)
    print(f"Completed training for {target_column}. Accuracy achieved: {accuracy:.4f}\n")

    # Return the model, a dictionary of encoders (store target encoder and onehot_cols info),
    # the scaler, the results, feature importances, accuracy, and the updated input_features.
    return rf_model, {'target': target_encoder, 'onehot_cols': onehot_cols, 'input_features': input_features_updated}, scaler, results, feature_importances, accuracy

def save_all_results(results_all, models, output_filename='nba_predictions_results_all.xlsx'):
    print(f"Saving all results to {output_filename}...")
    with pd.ExcelWriter(output_filename, mode='w') as writer:
        for position, results in results_all.items():
            sheet_name_pred = f'{position}_predictions'
            results.to_excel(writer, sheet_name=sheet_name_pred, index=False)
            sheet_name_imp = f'{position}_feature_importance'
            models[position]['feature_importances'].to_excel(writer, sheet_name=sheet_name_imp)
    print(f"All results have been successfully saved to {output_filename}.")

# ----- Training Phase -----
print("----- Training Phase -----")
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

combined_data = pd.concat(data_frames, ignore_index=True).dropna()
combined_data.drop_duplicates(inplace=True)
print("Data combined. NaN values and duplicates have been removed.")

# Filter for winning home team samples
combined_data = combined_data[combined_data['outcome'] == 1]
print(f"Filtered dataset for winning home teams: {len(combined_data)} record(s) found.")
print(f"Seasons included in the dataset: {combined_data['season'].unique().tolist()}")

# Train models for each missing home position
models = {}
results_all = {}
for position in [f'home_{i}' for i in range(5)]:
    print(f"\n--- Starting model training for missing position: {position} ---")
    X, y, input_features, target_column, game_ids, original_home_teams, original_away_teams = create_position_dataset(
        combined_data, position)
    rf_model, encoder_dict, scaler, results, importances, acc = train_position_model(
        X, y, input_features, target_column, game_ids, original_home_teams, original_away_teams)
    models[position] = {
        'model': rf_model,
        'encoders': encoder_dict,
        'scaler': scaler,
        'feature_importances': importances,
        'accuracy': acc,
        'input_features': encoder_dict['input_features']
    }
    results_all[position] = results
    print(f"--> Finished training for {position} with an accuracy of {acc:.4f}")

# Optionally, save training results to an Excel file
# save_all_results(results_all, models)

# ----- Testing Phase -----
print("\n----- Testing Phase -----")
print("Loading test data from NBA_test(2007-2009).csv")
test_data = pd.read_csv('NBA_test(2016).csv')
test_data['original_index'] = test_data.index  # Track original indices

# Dictionary to store test predictions for each missing position
test_results = {}

# Loop over each home position
for position in [f'home_{i}' for i in range(5)]:
    print(f"\nProcessing test data for missing position: {position}")
    # Select rows where the missing player is indicated by '?'
    test_subset = test_data[test_data[position] == '?'].copy()
    if test_subset.empty:
        print(f"No test samples with missing {position}.")
        continue

    # Store original indices for alignment
    original_indices = test_subset['original_index'].values

    # Retrieve saved model components for this position
    model_info = models[position]
    rf_model = model_info['model']
    encoder_dict = model_info['encoders']
    scaler = model_info['scaler']

    # Create feature dataset for prediction using the same function
    X_test, _, _, _, _, _, _ = create_position_dataset(test_subset, position)
    # Apply one-hot encoding to the same columns as in training:
    onehot_cols = encoder_dict['onehot_cols']
    X_test = pd.get_dummies(X_test, columns=onehot_cols)
    # Reindex to ensure same feature columns as in training
    X_test = X_test.reindex(columns=model_info['input_features'], fill_value=0)
    X_test = X_test.fillna(0)

    # Scale the numerical feature 'starting_min'
    if 'starting_min' in X_test.columns:
        X_test['starting_min'] = scaler.transform(X_test[['starting_min']])

    # Predict missing player for this position
    predictions = rf_model.predict(X_test)
    predicted_players = encoder_dict['target'].inverse_transform(predictions)

    # Store predictions in the subset
    test_subset[f'predicted_{position}'] = predicted_players
    test_subset['original_index'] = original_indices
    test_results[position] = test_subset

# Combine predictions from all positions (assuming one missing position per row)
if test_results:
    combined_test_results = pd.concat(test_results.values(), ignore_index=True)
    combined_test_results.sort_values('original_index', inplace=True)

    # For each row, select the predicted value from the respective predicted column
    def get_predicted_player(row):
        for pos in [f'home_{i}' for i in range(5)]:
            col = f'predicted_{pos}'
            if col in row and pd.notnull(row[col]):
                return row[col]
        return None
    combined_test_results['predicted_player'] = combined_test_results.apply(get_predicted_player, axis=1)

    # Load test labels from NBA_test_labels(2007-2009).csv; ensure it has 'original_index'
    labels_df = pd.read_csv('NBA_test_labels(2016).csv')
    if 'original_index' not in labels_df.columns:
        labels_df = labels_df.reset_index().rename(columns={'index': 'original_index'})

    # Merge test labels based on original_index
    combined_test_results = combined_test_results.merge(labels_df[['original_index', 'removed_value']],
                                                        on='original_index', how='left')
    combined_test_results.rename(columns={'removed_value': 'actual_player'}, inplace=True)

    # Calculate accuracy on the rows that received predictions and actual labels
    test_accuracy = accuracy_score(combined_test_results['actual_player'], combined_test_results['predicted_player'])
    print(f"\nTest Accuracy: {test_accuracy:.4f}")

    # Save combined test predictions to an Excel file
    combined_test_results.to_excel('NBA_test_predictions.xlsx', index=False)
    print("Test results saved to NBA_test_predictions.xlsx")
else:
    print("No test predictions were made.")
