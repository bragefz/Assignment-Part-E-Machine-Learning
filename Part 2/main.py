import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, cohen_kappa_score, f1_score

# Read the data
df = pd.read_csv('../car_data.csv')

# Define category orders for ordinal features
category_orders = {
    'buying': ['low', 'med', 'high', 'vhigh'],
    'maint': ['low', 'med', 'high', 'vhigh'],
    'doors': ['2', '3', '4', '5more'],
    'persons': ['2', '4', 'more'],
    'lug_boot': ['small', 'med', 'big'],
    'safety': ['low', 'med', 'high']
}

# Create copies for encoding
X = df.drop(columns=['class'])
y = df['class']

# Apply ordinal encoding to features
X_encoded = X.copy()
for feature, categories in category_orders.items():
    encoder = OrdinalEncoder(categories=[categories])
    X_encoded[feature] = encoder.fit_transform(X[feature].values.reshape(-1, 1))

# Apply label encoding to classes
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
class_names = label_encoder.classes_

# Split the data into training and testing sets (80% training, 20% testing)
# stratify=y_encoded ensures that class distribution remains the same in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"\nTraining set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")
print(f"Class mapping: {dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))}")

# Display information about class distribution
print("Full Dataset Class Distribution:")
class_counts_full = pd.Series(y_encoded).value_counts().sort_index()
for i, count in enumerate(class_counts_full):
    percentage = 100 * count / len(y_encoded)
    print(f"  {class_names[i]}: {count} instances ({percentage:.1f}%)")

print("\nTraining Set Class Distribution:")
class_counts_train = pd.Series(y_train).value_counts().sort_index()
for i, count in enumerate(class_counts_train):
    percentage = 100 * count / len(y_train)
    print(f"  {class_names[i]}: {count} instances ({percentage:.1f}%)")

print("\nTest Set Class Distribution:")
class_counts_test = pd.Series(y_test).value_counts().sort_index()
for i, count in enumerate(class_counts_test):
    percentage = 100 * count / len(y_test)
    print(f"  {class_names[i]}: {count} instances ({percentage:.1f}%)")



def evaluate_model(model, X_train, X_test, y_train, y_test, class_names, model_name, params):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Calculate performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    kappa = cohen_kappa_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')

    print(f"\n{model_name} Results with {params}:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Cohen's Kappa: {kappa:.4f}")
    print(f"Macro F1 Score: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    # Create and plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    return {
        'model': model,
        'accuracy': accuracy,
        'kappa': kappa,
        'f1': f1,
        'predictions': y_pred
    }


# K-Nearest Neighbors (KNN) Experiments


# Parameters to test
n_neighbors_values = [3, 5, 7]
weights_values = ['uniform', 'distance']

# Dictionary to store KNN results
knn_results = {}

# Loop through all combinations of parameters
for n in n_neighbors_values:
    for weights in weights_values:
        # Create KNN model with specified parameters
        model_name = f"KNN (n_neighbors={n}, weights='{weights}')"
        params = f"n_neighbors={n}, weights='{weights}'"

        knn = KNeighborsClassifier(n_neighbors=n, weights=weights)

        # Evaluate the model
        result = evaluate_model(knn, X_train, X_test, y_train, y_test, class_names, model_name, params)

        # Store result
        knn_results[f"knn_n{n}_{weights}"] = result

# Find best KNN model based on F1 score
best_knn_key = max(knn_results, key=lambda k: knn_results[k]['f1'])
best_knn = knn_results[best_knn_key]
print(f"\nBest KNN Model: {best_knn_key}")
print(f"F1 Score: {best_knn['f1']:.4f}")
print(f"Accuracy: {best_knn['accuracy']:.4f}")



# Random Forest Experiments

# Parameters to test
n_estimators_values = [50, 100, 200]
class_weight_values = [None, 'balanced']

# Dictionary to store RF results
rf_results = {}

# Loop through all combinations of parameters
for n_est in n_estimators_values:
    for class_weight in class_weight_values:
        # Create Random Forest model with specified parameters
        model_name = f"Random Forest (n_estimators={n_est}, class_weight='{class_weight}')"
        params = f"n_estimators={n_est}, class_weight='{class_weight}'"

        rf = RandomForestClassifier(n_estimators=n_est, class_weight=class_weight, random_state=42)

        # Evaluate the model
        result = evaluate_model(rf, X_train, X_test, y_train, y_test, class_names, model_name, params)

        # Store result
        rf_results[f"rf_n{n_est}_{class_weight}"] = result

# Find best RF model based on F1 score
best_rf_key = max(rf_results, key=lambda k: rf_results[k]['f1'])
best_rf = rf_results[best_rf_key]
print(f"\nBest Random Forest Model: {best_rf_key}")
print(f"F1 Score: {best_rf['f1']:.4f}")
print(f"Accuracy: {best_rf['accuracy']:.4f}")


# Comparison of Best Models

print("\n===== Comparison of Best Models =====")
print(f"Best KNN: {best_knn_key}")
print(f"  - F1 Score: {best_knn['f1']:.4f}")
print(f"  - Accuracy: {best_knn['accuracy']:.4f}")
print(f"  - Kappa: {best_knn['kappa']:.4f}")

print(f"\nBest Random Forest: {best_rf_key}")
print(f"  - F1 Score: {best_rf['f1']:.4f}")
print(f"  - Accuracy: {best_rf['accuracy']:.4f}")
print(f"  - Kappa: {best_rf['kappa']:.4f}")

# Compare with baseline (picking the most common class)
from sklearn.dummy import DummyClassifier

dummy = DummyClassifier(strategy='most_frequent')
dummy.fit(X_train, y_train)
y_pred_dummy = dummy.predict(X_test)
dummy_accuracy = accuracy_score(y_test, y_pred_dummy)
dummy_f1 = f1_score(y_test, y_pred_dummy, average='macro')

print(f"\nBaseline (Most Frequent Class):")
print(f"  - F1 Score: {dummy_f1:.4f}")
print(f"  - Accuracy: {dummy_accuracy:.4f}")

# Plot the comparison of all models
models = list(knn_results.keys()) + list(rf_results.keys())
f1_scores = [knn_results[k]['f1'] for k in knn_results] + [rf_results[k]['f1'] for k in rf_results]
accuracies = [knn_results[k]['accuracy'] for k in knn_results] + [rf_results[k]['accuracy'] for k in rf_results]

# Create a summary table of all models
results_df = pd.DataFrame({
    'Model': models,
    'F1 Score': f1_scores,
    'Accuracy': accuracies,
    'Kappa': [knn_results[k]['kappa'] for k in knn_results] + [rf_results[k]['kappa'] for k in rf_results]
})

# Sort by F1 score descending
results_df = results_df.sort_values('F1 Score', ascending=False)
print("\nAll Models Performance Summary (Sorted by F1 Score):")
print(results_df)