import pandas as pd
import glob
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from scipy.stats import randint


def create_position_dataset(data, missing_position):
    print(f"Preparing dataset by removing {missing_position} from input features...")
    position_data = data.copy()

    home_players = [f'home_{i}' for i in range(5)]
    away_players = [f'away_{i}' for i in range(5)]
    model_features = ['season', 'home_team', 'away_team', 'starting_min']

    # Remove the missing position from input features
    input_home_players = [p for p in home_players if p != missing_position]
    input_features = model_features + input_home_players + away_players
    target_column = missing_position

    # Create or retrieve IDs
    if 'game' in position_data.columns:
        game_ids = position_data['game'].copy()
    else:
        game_ids = pd.Series(range(len(position_data)))

    X = position_data[input_features].copy()
    y = position_data[target_column].copy()

    print(f"Dataset for missing position {missing_position} prepared with {X.shape[0]} samples.")
    return X, y, input_features, target_column, game_ids

def train_position_model(X, y, input_features, target_column, game_ids):
    print(f"\nTraining for: {target_column}...")

    # One-hot encoding
    categorical_cols = [col for col in X.columns if col != 'starting_min']
    onehot_cols = [col for col in categorical_cols if col != target_column]
    print("Performing one-hot encoding on columns:", onehot_cols)
    X = pd.get_dummies(X, columns=onehot_cols)
    input_features_updated = list(X.columns)

    print("Encoding target variable...")
    target_encoder = LabelEncoder()
    y_encoded = target_encoder.fit_transform(y)
    print("Target variable encoded.")

    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test, game_train, game_test = train_test_split(
        X, y_encoded, game_ids, test_size=0.2, random_state=42
    )
    print(f"Data split: {len(X_train)} training samples and {len(X_test)} testing samples.")

    print("Scaling the 'starting_min' feature...")
    scaler = StandardScaler()
    X_train['starting_min'] = scaler.fit_transform(X_train[['starting_min']])
    X_test['starting_min'] = scaler.transform(X_test[['starting_min']])
    print("Feature scaling complete.")

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
    acc = accuracy_score(y_test, rf_predictions)
    print(f"Completed training for {target_column}. Accuracy achieved: {acc:.4f}\n")

    encoders = {
        'target': target_encoder,
        'onehot_cols': onehot_cols,
        'input_features': input_features_updated
    }
    return rf_model, encoders, scaler, acc

def run_test_phase(test_data, test_labels, model, encoders, scaler, position):
    """
    Given a single model, encoders, and scaler for a specific position,
    apply to the test data and produce predictions. Return a DataFrame of results.
    """
    test_subset = test_data[test_data[position] == '?'].copy()
    if test_subset.empty:
        print(f"No missing samples for {position}.")
        return None

    original_indices = test_subset['original_index'].values

    X_test, _, _, _, _ = create_position_dataset(test_subset, position)
    onehot_cols = encoders['onehot_cols']
    X_test = pd.get_dummies(X_test, columns=onehot_cols)
    X_test = X_test.reindex(columns=encoders['input_features'], fill_value=0)
    X_test = X_test.fillna(0)
    if 'starting_min' in X_test.columns:
        X_test['starting_min'] = scaler.transform(X_test[['starting_min']])

    predictions = model.predict(X_test)
    predicted_players = encoders['target'].inverse_transform(predictions)
    test_subset[f'predicted_{position}'] = predicted_players
    test_subset['original_index'] = original_indices
    return test_subset

def merge_test_results(test_results, test_labels, output_filename):
    if not test_results:
        print("No predictions were made.")
        return None

    combined_test_results = pd.concat(test_results, ignore_index=True)
    combined_test_results.sort_values('original_index', inplace=True)

    def get_predicted_player(row):
        for pos in [f'home_{i}' for i in range(5)]:
            col = f'predicted_{pos}'
            if col in row and pd.notnull(row[col]):
                return row[col]
        return None
    combined_test_results['predicted_player'] = combined_test_results.apply(get_predicted_player, axis=1)

    if 'original_index' not in test_labels.columns:
        test_labels = test_labels.reset_index().rename(columns={'index': 'original_index'})
    combined_test_results = combined_test_results.merge(
        test_labels[['original_index', 'removed_value']],
        on='original_index', how='left'
    )
    combined_test_results.rename(columns={'removed_value': 'actual_player'}, inplace=True)

    test_accuracy = accuracy_score(combined_test_results['actual_player'], combined_test_results['predicted_player'])
    print(f"\nTest Accuracy: {test_accuracy:.4f}")

    combined_test_results.to_excel(output_filename, index=False)
    print(f"Test results saved to {output_filename}.")
    return test_accuracy

# ----- Main Script with Automatic Splits -----

if __name__ == '__main__':

    # Define your splits
    splits = [
        {
            "name": "2007-2009",
            "train_start": 2007,
            "train_end": 2009,
            "test_data": "Test_Datasets/NBA_test(2007-2009).csv",
            "test_labels": "Test_Datasets/NBA_test_labels(2007-2009).csv",
            "output_excel": "NBA_test_predictions(2007-2009).xlsx"
        },
        {
            "name": "2010-2012",
            "train_start": 2010,
            "train_end": 2012,
            "test_data": "Test_Datasets/NBA_test(2010-2012).csv",
            "test_labels": "Test_Datasets/NBA_test_labels(2010-2012).csv",
            "output_excel": "NBA_test_predictions(2010-2012).xlsx"
        },
        {
            "name": "2013-2015",
            "train_start": 2013,
            "train_end": 2015,
            "test_data": "Test_Datasets/NBA_test(2013-2016).csv",
            "test_labels": "Test_Datasets/NBA_test_labels(2013-2016).csv",
            "output_excel": "NBA_test_predictions(2013-2016).xlsx"
        },

    ]

    matchup_files = glob.glob("Datasets/matchups-*.csv")
    if not matchup_files:
        print("No matchup files found. Exiting.")
        exit()
    data_frames = []
    for file in matchup_files:
        try:
            df = pd.read_csv(file)
            data_frames.append(df)
        except Exception as e:
            print(f"Error loading {file}: {e}")
    combined_data = pd.concat(data_frames, ignore_index=True).dropna()
    combined_data.drop_duplicates(inplace=True)
    print("\nAll matchup data loaded and combined.")

    # Filter for winning home teams
    combined_data = combined_data[combined_data['outcome'] == 1]
    print(f"Filtered for winning home teams: {len(combined_data)} total records after filtering.\n")

    # We'll store final results in a summary dictionary to print after all splits
    summary_results = {}

    for split_info in splits:
        split_name = split_info["name"]
        train_start = split_info["train_start"]
        train_end   = split_info["train_end"]
        test_data_file   = split_info["test_data"]
        test_labels_file = split_info["test_labels"]
        output_excel     = split_info["output_excel"]

        print(f"\n===== PROCESSING SPLIT: {split_name} =====")
        # Filter training data by season range
        train_data = combined_data[combined_data['season'].between(train_start, train_end)]
        print(f"Training data for {split_name}: {len(train_data)} records (Seasons {train_start}-{train_end}).")

        # Train a model for each missing home position
        models_for_split = {}
        train_accuracies_for_split = {}  # Track each position's training accuracy
        for position in [f'home_{i}' for i in range(5)]:
            print(f"\n--- Training model for missing position: {position} in {split_name} ---")
            X, y, input_feats, target_col, game_ids = create_position_dataset(train_data, position)
            rf_model, encoders, scaler, acc = train_position_model(X, y, input_feats, target_col, game_ids)
            models_for_split[position] = {
                'model': rf_model,
                'encoders': encoders,
                'scaler': scaler
            }
            train_accuracies_for_split[position] = acc
            print(f"Trained {position} with accuracy: {acc:.4f}")

        # Test Phase
        print(f"\n----- Testing {split_name} with test data: {test_data_file} -----")
        test_df = pd.read_csv(test_data_file)
        test_df['original_index'] = test_df.index

        partial_results = []
        for position in [f'home_{i}' for i in range(5)]:
            print(f"\nPredicting missing {position} for split {split_name}...")
            sub_res = run_test_phase(test_df, pd.read_csv(test_labels_file),
                                     models_for_split[position]['model'],
                                     models_for_split[position]['encoders'],
                                     models_for_split[position]['scaler'],
                                     position)
            if sub_res is not None:
                partial_results.append(sub_res)

        test_accuracy_for_split = merge_test_results(partial_results, pd.read_csv(test_labels_file), output_excel)

        # Store the final results for printing later
        summary_results[split_name] = {
            "train_accuracies": train_accuracies_for_split,
            "test_accuracy": test_accuracy_for_split
        }

    print("\nAll splits have been processed.")

    # Print final summary of training & testing accuracy
    print("\n========== FINAL SUMMARY ==========")
    for split_name, result_dict in summary_results.items():
        print(f"\nSPLIT: {split_name}")
        train_acc_dict = result_dict["train_accuracies"]
        for position, acc in train_acc_dict.items():
            print(f"  - {position} training accuracy: {acc:.4f}")
        print(f"  => Overall test accuracy: {result_dict['test_accuracy']:.4f}")

