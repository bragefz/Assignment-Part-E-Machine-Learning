import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder

df = pd.read_csv('../car_data.csv')

X = df.drop(columns=['class'])
y = df[['class']]

# 1. Class balance check
plt.figure(figsize=(10, 6))
colors = sns.color_palette("viridis", 4)
class_counts = df['class'].value_counts()
class_percents = (class_counts / len(df) * 100).round(1)

ax = sns.barplot(x=class_counts.index, y=class_counts.values, palette=colors)
for i, (count, percent) in enumerate(zip(class_counts.values, class_percents)):
    ax.text(i, count + 20, f'{count}\n({percent}%)', ha='center', va='bottom', fontweight='bold')

plt.title('Class Distribution of 1728 Instances', fontsize=16, pad=20)
plt.ylabel('Number of Instances', fontsize=14)
plt.xlabel('Car Evaluation Class', fontsize=14)
plt.xticks(rotation=0, fontsize=12)
plt.yticks(fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('class_distribution.png')


# 2. Feature distributions and relationships
# First encode the data properly for visualization

# Define category orders for features
category_orders = {
    'buying': ['low', 'med', 'high', 'vhigh'],
    'maint': ['low', 'med', 'high', 'vhigh'],
    'doors': ['2', '3', '4', '5more'],
    'persons': ['2', '4', 'more'],
    'lug_boot': ['small', 'med', 'big'],
    'safety': ['low', 'med', 'high']
}

# Create a copy for encoded version
df_encoded = df.copy()

# Encode features with OrdinalEncoder
for feature, categories in category_orders.items():
    # Create OrdinalEncoder with specified category order
    encoder = OrdinalEncoder(categories=[categories])
    # Reshape data for OrdinalEncoder (requires 2D array)
    reshaped_data = df[feature].values.reshape(-1, 1)
    # Apply encoding
    df_encoded[feature] = encoder.fit_transform(reshaped_data)

# Just for correlation analysis, will use unordered encoding on classes for machine learning
class_order_mapping = {'unacc': 0, 'acc': 1, 'good': 2, 'vgood': 3}
df_encoded['class_ordered'] = df['class'].map(class_order_mapping)
class_names = list(class_order_mapping.keys())

# 5. Calculate statistics
stats_df = pd.DataFrame(columns=['mean', 'mode', 'Q1', 'median', 'Q3', 'std_dev', 'variance'])
for feature in X.columns:
    # Calculate each statistic
    mean_val = df_encoded[feature].mean()
    median_val = df_encoded[feature].median()
    mode_val = df_encoded[feature].mode()[0]
    std_dev = df_encoded[feature].std()
    variance = df_encoded[feature].var()
    q1 = df_encoded[feature].quantile(0.25)
    q3 = df_encoded[feature].quantile(0.75)

    # Add to dataframe with Q1 and Q3 on either side of median
    stats_df.loc[feature] = [mean_val, mode_val, q1, median_val, q3, std_dev, variance]

print("Statistics for each feature:")
print(stats_df)


"""Feature distributions by class"""
plt.figure(figsize=(10, 6))
for i, feature in enumerate(X.columns):
    plt.subplot(2, 3, i+1)
    sns.boxplot(x='class', y=feature, data=df_encoded, palette="viridis")
    plt.title(f'{feature.capitalize()} by Class')
    plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('feature_boxplots.png')


"""Feature correlation heatmap"""
plt.figure(figsize=(10, 6))
corr = df_encoded.drop(columns=['class']).corr(method='spearman')  # Change to Spearman
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="viridis",
            linewidths=.5, cbar_kws={"shrink": .8})
plt.title('Feature Correlation Heatmap (Spearman)', fontsize=16, pad=20)  # Update title
plt.tight_layout()
plt.savefig('correlation_heatmap.png')

"""Jittered scatterplot matrix for all feature combinations including histograms on the diagonal."""
plt.figure(figsize=(10, 6))  # Larger figure to fit all combinations

# Define features
features = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']

# Create feature mappings for numeric conversion
feature_maps = {
    'buying': {'low': 0, 'med': 1, 'high': 2, 'vhigh': 3},
    'maint': {'low': 0, 'med': 1, 'high': 2, 'vhigh': 3},
    'doors': {'2': 0, '3': 1, '4': 2, '5more': 3},
    'persons': {'2': 0, '4': 1, 'more': 2},
    'lug_boot': {'small': 0, 'med': 1, 'big': 2},
    'safety': {'low': 0, 'med': 1, 'high': 2}
}

# Create color mapping for classes
class_colors = {'unacc': 'darkblue', 'acc': 'skyblue', 'good': 'lightgreen', 'vgood': 'teal'}
colors = [class_colors[c] for c in df['class']]

# Set up jitter amount
jitter_amount = 0.15

# Define number of rows and columns in the grid
n_features = len(features)

# Create subplots in a grid
for i, feature1 in enumerate(features):
    for j, feature2 in enumerate(features):
        # Create subplot
        plt.subplot(n_features, n_features, i * n_features + j + 1)

        if i == j:  # Diagonal elements - show histograms
            # Create a separate count for each class
            for class_name, color in class_colors.items():
                subset = df[df['class'] == class_name]
                values = [feature_maps[feature1][val] for val in subset[feature1]]
                plt.hist(values, alpha=0.5, color=color, bins=range(len(feature_maps[feature1]) + 1))

            plt.title(feature1, fontsize=10)
            plt.xticks(range(len(feature_maps[feature1])))

        else:  # Off-diagonal - show jittered scatter
            # Generate jitter
            x_jitter = np.random.uniform(-jitter_amount, jitter_amount, size=len(df))
            y_jitter = np.random.uniform(-jitter_amount, jitter_amount, size=len(df))

            # Map to numeric and add jitter
            x_values = np.array([feature_maps[feature2][val] for val in df[feature2]]) + x_jitter
            y_values = np.array([feature_maps[feature1][val] for val in df[feature1]]) + y_jitter

            # Plot with transparency
            plt.scatter(x_values, y_values, c=colors, alpha=0.3, s=10)

            # Only show labels on edge plots
            if i == n_features - 1:
                plt.xlabel(feature2, fontsize=10)
            else:
                plt.xticks([])

            if j == 0:
                plt.ylabel(feature1, fontsize=10)
            else:
                plt.yticks([])

            # Set axis limits to keep points within bounds
            plt.xlim(-0.5, len(feature_maps[feature2]) - 0.5)
            plt.ylim(-0.5, len(feature_maps[feature1]) - 0.5)

# Add a single legend for the entire figure
legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                              markerfacecolor=color, markersize=10, label=class_name)
                   for class_name, color in class_colors.items()]

plt.figlegend(handles=legend_elements, loc='lower center',
              bbox_to_anchor=(0.5, 0), ncol=len(class_colors))

plt.tight_layout()
plt.subplots_adjust(bottom=0.085)  # Make room for the legend
plt.savefig('jittered_scatterplot_matrix.png', dpi=300, bbox_inches='tight')

"""Histograms of features."""
plt.figure(figsize=(10, 6))
for i, feature in enumerate(features):
    plt.subplot(2, 3, i + 1)

    # Plot histogram for each class with a different color
    for class_name, color in class_colors.items():
        subset = df[df['class'] == class_name]
        values = [feature_maps[feature][val] for val in subset[feature]]
        plt.hist(values, alpha=0.5, color=color,
                 bins=range(len(feature_maps[feature]) + 1),
                 label=class_name)

    plt.title(f'{feature.capitalize()} Distribution by Class')
    plt.xlabel(feature.capitalize())
    plt.ylabel('Frequency')
    plt.xticks(range(len(feature_maps[feature])), list(feature_maps[feature].keys()))

    # Only add legend to the first plot to avoid repetition
    if i == 0:
        plt.legend()

plt.tight_layout()
plt.savefig('feature_histograms.png')

plt.show()