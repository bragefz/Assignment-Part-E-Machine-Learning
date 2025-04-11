from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, adjusted_rand_score, adjusted_mutual_info_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.model_selection import train_test_split

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


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)


# Function to evaluate clustering performance
def evaluate_clustering(X, y_true, labels, model_name, params):
    # Number of clusters found
    n_clusters = len(np.unique(labels[labels != -1]))
    n_noise = list(labels).count(-1) if -1 in labels else 0

    # Calculate silhouette score if there's more than one cluster and no points are classified as noise
    sil_score = np.nan
    if n_clusters > 1 and n_noise < len(labels):
        # Filter out noise points for silhouette calculation
        if -1 in labels:
            mask = labels != -1
            sil_score = silhouette_score(X[mask], labels[mask])
        else:
            sil_score = silhouette_score(X, labels)

    # Calculate external metrics comparing with true labels
    ari = adjusted_rand_score(y_true, labels)
    ami = adjusted_mutual_info_score(y_true, labels)

    print(f"\n{model_name} Results with {params}:")
    print(f"Number of clusters: {n_clusters}")
    print(f"Number of noise points: {n_noise} ({n_noise / len(labels):.1%})")
    print(f"Silhouette Score: {sil_score:.4f}" if not np.isnan(sil_score) else "Silhouette Score: N/A")
    print(f"Adjusted Rand Index: {ari:.4f}")
    print(f"Adjusted Mutual Info Score: {ami:.4f}")

    # Compare clusters with true class distribution
    cluster_class_distribution = pd.crosstab(
        pd.Series(labels, name='Cluster'),
        pd.Series(y_true, name='True Class')
    )
    print("\nCluster-Class Distribution:")
    print(cluster_class_distribution)

    return {
        'n_clusters': n_clusters,
        'n_noise': n_noise,
        'silhouette': sil_score,
        'ari': ari,
        'ami': ami,
        'labels': labels
    }


# K-Means Clustering Experiments


# Parameters to test
k_values = [2, 3, 4, 6, 8]

# Dictionary to store KMeans results
kmeans_results = {}

# Loop through different k values
for k in k_values:
    # Create KMeans model with specified parameters
    model_name = f"K-Means"
    params = f"n_clusters={k}"

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    # Evaluate the clustering
    result = evaluate_clustering(X_scaled, y_encoded, labels, model_name, params)
    kmeans_results[f"kmeans_k{k}"] = result

# Find best KMeans model based on silhouette score
valid_keys = [k for k in kmeans_results if not np.isnan(kmeans_results[k]['silhouette'])]
if valid_keys:
    best_kmeans_key = max(valid_keys, key=lambda k: kmeans_results[k]['silhouette'])
    best_kmeans = kmeans_results[best_kmeans_key]
    print(f"\nBest K-Means Model: {best_kmeans_key}")
    print(f"Silhouette Score: {best_kmeans['silhouette']:.4f}")
    print(f"ARI: {best_kmeans['ari']:.4f}")
    print(f"AMI: {best_kmeans['ami']:.4f}")
else:
    print("\nCould not determine best K-Means model (no valid silhouette scores)")

# Plot silhouette scores for different k values
plt.figure(figsize=(10, 6))
silhouette_scores = [kmeans_results[f"kmeans_k{k}"]['silhouette'] for k in k_values
                     if not np.isnan(kmeans_results[f"kmeans_k{k}"]['silhouette'])]
valid_k = [k for k in k_values if not np.isnan(kmeans_results[f"kmeans_k{k}"]['silhouette'])]

if valid_k:
    plt.plot(valid_k, silhouette_scores, 'o-', linewidth=2)
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score vs. Number of Clusters')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('kmeans_silhouette_scores.png')
    print("\nSilhouette score plot saved as 'kmeans_silhouette_scores.png'")


# DBSCAN Experiments


# Parameters to test
eps_values = [0.3, 0.5, 1.5]
min_samples_values = [10, 30]

# Dictionary to store DBSCAN results
dbscan_results = {}

# Loop through all combinations of parameters
for eps in eps_values:
    for min_samples in min_samples_values:
        # Create DBSCAN model with specified parameters
        model_name = f"DBSCAN"
        params = f"eps={eps}, min_samples={min_samples}"

        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(X_scaled)

        # Evaluate the clustering
        result = evaluate_clustering(X_scaled, y_encoded, labels, model_name, params)

        # Store result
        dbscan_results[f"dbscan_eps{eps}_min{min_samples}"] = result

# Find best DBSCAN model based on ARI (since silhouette might not be available for all)
best_dbscan_key = max(dbscan_results, key=lambda k: dbscan_results[k]['ari'])
best_dbscan = dbscan_results[best_dbscan_key]
print(f"\nBest DBSCAN Model: {best_dbscan_key}")
print(f"ARI: {best_dbscan['ari']:.4f}")
print(f"AMI: {best_dbscan['ami']:.4f}")
if not np.isnan(best_dbscan['silhouette']):
    print(f"Silhouette Score: {best_dbscan['silhouette']:.4f}")


# Comparison of Best Models

print("\n===== Comparison of All Clustering Models =====")

# Create a summary table of all models
results_list = []

# Add KMeans results
for key in kmeans_results:
    results_list.append({
        'Model': key,
        'Clusters': kmeans_results[key]['n_clusters'],
        'Noise Points': kmeans_results[key]['n_noise'],
        'Silhouette': kmeans_results[key]['silhouette'],
        'ARI': kmeans_results[key]['ari'],
        'AMI': kmeans_results[key]['ami']
    })

# Add DBSCAN results
for key in dbscan_results:
    results_list.append({
        'Model': key,
        'Clusters': dbscan_results[key]['n_clusters'],
        'Noise Points': dbscan_results[key]['n_noise'],
        'Silhouette': dbscan_results[key]['silhouette'],
        'ARI': dbscan_results[key]['ari'],
        'AMI': dbscan_results[key]['ami']
    })

# Create DataFrame and sort by ARI (as silhouette might be NaN for some)
results_df = pd.DataFrame(results_list)
results_df = results_df.sort_values('ARI', ascending=False)
print("\nAll Models Performance Summary (Sorted by ARI):")
print(results_df)

# Also sort by AMI for comparison
results_df_ami = results_df.sort_values('AMI', ascending=False)
print("\nAll Models Performance Summary (Sorted by AMI):")
print(results_df_ami)

# Check if any silhouette scores are available and sort by that too
if not results_df['Silhouette'].isna().all():
    valid_silhouette = results_df[~results_df['Silhouette'].isna()]
    if not valid_silhouette.empty:
        valid_silhouette = valid_silhouette.sort_values('Silhouette', ascending=False)
        print("\nModels with Valid Silhouette Scores (Sorted by Silhouette):")
        print(valid_silhouette)


# Visualization of Best Clustering Results


# Use dimensionality reduction for visualization
from sklearn.decomposition import PCA

# Apply PCA to reduce to 2 dimensions for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Plot the best KMeans clustering
if 'best_kmeans_key' in locals():
    plt.figure(figsize=(10, 8))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_results[best_kmeans_key]['labels'], cmap='viridis', s=50, alpha=0.8)
    plt.title(f'Best K-Means Clustering: {best_kmeans_key}')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.colorbar(label='Cluster')
    plt.tight_layout()
    plt.savefig('best_kmeans_clustering.png')
    print("\nBest K-Means clustering visualization saved as 'best_kmeans_clustering.png'")

# Plot the best DBSCAN clustering
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=dbscan_results[best_dbscan_key]['labels'],
                      cmap='viridis', s=50, alpha=0.8)
plt.title(f'Best DBSCAN Clustering: {best_dbscan_key}')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(scatter, label='Cluster')
plt.tight_layout()
plt.savefig('best_dbscan_clustering.png')
print("\nBest DBSCAN clustering visualization saved as 'best_dbscan_clustering.png'")

# Plot the true classes for comparison
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_encoded, cmap='viridis', s=50, alpha=0.8)
plt.title('True Class Distribution')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(scatter, label='Class')
plt.tight_layout()
plt.savefig('true_classes.png')
print("\nTrue class distribution visualization saved as 'true_classes.png'")


# For the best KMeans model
if 'best_kmeans_key' in locals():
    # Create a mapping from cluster to most common class
    kmeans_labels = kmeans_results[best_kmeans_key]['labels']
    kmeans_mapping = {}
    for cluster in np.unique(kmeans_labels):
        mask = kmeans_labels == cluster
        if mask.any():
            most_common_class = np.bincount(y_encoded[mask]).argmax()
            kmeans_mapping[cluster] = most_common_class

    # Create predicted labels using the mapping
    kmeans_pred = np.array([kmeans_mapping[label] for label in kmeans_labels])

    # Calculate accuracy of the mapping
    kmeans_accuracy = np.mean(kmeans_pred == y_encoded)
    print(f"\nK-Means clustering accuracy (after mapping): {kmeans_accuracy:.4f}")

# Handle noise points (-1) separately
dbscan_labels = dbscan_results[best_dbscan_key]['labels']
unique_clusters = np.unique(dbscan_labels)
dbscan_mapping = {}

for cluster in unique_clusters:
    if cluster != -1:  # Skip noise points for mapping
        mask = dbscan_labels == cluster
        if mask.any():
            most_common_class = np.bincount(y_encoded[mask]).argmax()
            dbscan_mapping[cluster] = most_common_class

# For noise points (-1), assign the most common class overall as a fallback
if -1 in unique_clusters:
    most_common_class_overall = np.bincount(y_encoded).argmax()
    dbscan_mapping[-1] = most_common_class_overall

# Create predicted labels using the mapping
dbscan_pred = np.array([dbscan_mapping[label] for label in dbscan_labels])

# Calculate accuracy of the mapping
dbscan_accuracy = np.mean(dbscan_pred == y_encoded)
print(f"DBSCAN clustering accuracy (after mapping): {dbscan_accuracy:.4f}")