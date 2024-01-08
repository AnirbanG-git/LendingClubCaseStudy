import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import warnings

warnings.filterwarnings('ignore')

def get_null_stats(df):
    null_counts = df.isnull().sum()

    #Create a pandas series with null percentage stats
    null_percentage = round((df.isnull().sum() / df.shape[0]) * 100, 2)
    # Combine counts and percentages into a single DataFrame
    null_summary = pd.DataFrame({'Count': null_counts, 'Percentage': null_percentage})

    return null_counts, null_percentage, null_summary

def print_null_stats(df):
    null_counts, null_percentage, null_summary = get_null_stats(df)
    print(null_counts)
    print(null_percentage)
    print(null_summary)

def drop_high_null_cols(df, null_percentage):
    # Identify columns where the percentage of null values is greater than 60%
    columns_to_drop = null_percentage[null_percentage > 60].index

    # Drop these columns from the DataFrame
    df.drop(columns=columns_to_drop, inplace=True)

def print_col_val_desc(df, data_dictionary):
    # Transposing the first row of loan data
    first_row = df.head(1).T
    first_row.columns = ['Value']
    first_row['Column'] = first_row.index

    # Resetting index so that 'Column' becomes a regular column
    first_row.reset_index(drop=True, inplace=True)

    # Merging the first row of loan data with the data dictionary on the column names
    merged_data = first_row.merge(data_dictionary, left_on='Column', right_on='LoanStatNew', how='left')

    # Selecting only the necessary columns
    result = merged_data[['Column', 'Value', 'Description']]

    # Save the result DataFrame to a text file
    with open('loan_data_summary.txt', 'w') as file:
        file.write(result.to_string(index=False))

def print_unique_values(df):
    for column in df.columns:
        unique_values = df[column].unique()
        num_unique_values = len(unique_values)

        # Print the total number of unique values
        print(f"Column '{column}' has {num_unique_values} unique values.")

        # Print the top 10 unique values
        print(f"Top 10 unique values: {unique_values[:10]}")
        print("\n")  # New line for better readability between columns

def print_unique_values_and_null_stats(df):
    for column in df.columns:
        # Call the function to print unique values
        print_unique_values(df[[column]])

        # Get null statistics
        null_counts = df[column].isnull().sum()
        null_percentage = (null_counts / len(df)) * 100

        # Print the total count and percentage of null values
        print(f"Total null values: {null_counts}")
        print(f"Percentage of null values: {null_percentage:.2f}%")
        # Add a separator line
        print("============\n")

def plot_all_categorical(df):
    df_copy = df.copy()
    # Categorical Features
    categorical_features = df_copy.select_dtypes(include=[object]).columns
    for feature in categorical_features:
        df_copy[feature].fillna("NA", inplace=True)
        plot_categorical(df, feature)


def plot_all_numerical_univariate(df):
    df_copy = df.copy()
    # Numerical Features
    numerical_features = df_copy.select_dtypes(include=[np.number]).columns
    for feature in numerical_features:
        df_copy[feature].fillna(0, inplace=True)
        plot_numerical_univariate(df_copy, feature)


def convert_to_numeric(df, features):
    for feature in features:
        df[feature] = pd.to_numeric(df[feature].str.rstrip('%'), errors='coerce')

def plot_categorical(df, feature):
    print(f"Value counts for '{feature}':")
    value_counts = df[feature].value_counts()[:30]  # Select the top 30 values
    print(value_counts)
    print("="*40)
        
    plt.figure(figsize=(15, 4))
    sns.countplot(data=df, x=feature, hue='loan_status', order=value_counts.index)  # Use the top 30 values for ordering
    plt.title(f'{feature} by loan_status')
    plt.xticks(rotation=45)
    plt.show()

def plot_numerical_univariate(df, feature, feature_label=""):
    if not feature_label:
        feature_label = feature
    
    # Calculate quantiles
    quantiles = df[feature].quantile([0.25, 0.5, 0.75]).tolist()
    # Ensure unique bin edges by adding a small epsilon where quantiles are not unique
    quantiles = sorted(set(quantiles + [df[feature].min(), df[feature].max()]))
    eps = 1e-4  # Epsilon to ensure unique bin edges
    bins = [quantiles[0]]
    for q in quantiles[1:]:
        if q <= bins[-1]:
            q = bins[-1] + eps
        bins.append(q)

    # Bin labels
    labels = ["Low", "Medium", "High", "Very High"][:len(bins)-1]

    # Bin the data
    binned_feature = feature + '_binned'
    df[binned_feature] = pd.cut(df[feature], bins=bins, labels=labels, include_lowest=True)

    print(f"Summary statistics for '{feature_label}':")
    print(df[feature].describe())
    print("="*40)

    # Initialize the subplot
    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(20, 15))

    # Histogram
    sns.histplot(df[feature], bins=30, kde=True, ax=axs[0])
    axs[0].set_title(f'Distribution of {feature_label}')

    # Boxplot for the feature
    sns.boxplot(y=feature, data=df, ax=axs[1])
    axs[1].set_title(f'Box plot of {feature_label}')

    # Boxplot for the binned feature
    sns.boxplot(x=binned_feature, y=feature, data=df, ax=axs[2])
    axs[2].set_title(f"Box Plot of {feature_label} by Categories")
    axs[2].set_ylabel(feature_label)
    axs[2].set_xlabel(feature_label + ' Category')

    plt.tight_layout()
    plt.show()


def plot_numerical_bivariate(df, feature, target_feature, feature_label="", target_label=""):
    if not feature_label:
        feature_label = feature
    if not target_label:
        target_label = target_feature

    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(20, 4))

    sns.boxplot(x=target_feature, y=feature, data=df)
    axs.set_title(f'{feature_label} by {target_label}')
    axs.set_ylabel(feature_label)
    axs.set_xlabel(target_label)
    plt.show()


def plot_categorical_univariate(df, feature, feature_label="", top_20=False):
    if not feature_label:
        feature_label = feature
    
    # Calculate summary statistics
    top_20_value_counts = df[feature].value_counts(dropna=False).nlargest(20)
    value_counts = df[feature].value_counts(dropna=False)
    mode_value = value_counts.idxmax()
    proportions = value_counts / len(df)
    
    # Display summary statistics
    print(f"Summary statistics for '{feature_label}':")
    print(value_counts)
    print(f"\nMode: {mode_value}")
    print("="*40)

    if top_20:
        value_counts = top_20_value_counts
        proportions = value_counts / len(df)
    
    # Initialize the subplot
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(20, 12))
    
    # Bar plot for the frequency of each category
    sns.countplot(x=feature, data=df, ax=axs[0], order=value_counts.index)
    axs[0].set_title(f'Count Plot of {feature_label}')
    axs[0].set_xlabel(feature_label)
    axs[0].set_ylabel('Count')
    axs[0].tick_params(axis='x', rotation=90)  # Rotate x-axis labels for better visibility if needed
    
    # Pie chart for the proportion of each category
    proportions.plot(kind='pie', autopct='%1.1f%%', startangle=90, ax=axs[1])
    axs[1].set_title(f'Proportion of {feature_label}')
    axs[1].set_ylabel('')  # Hide the y-axis label for the pie chart
    axs[1].axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    
    plt.tight_layout()
    plt.show()

def plot_categorical_bivariate(df, feature, target_feature, feature_label="", target_label="", top_20=False):
    if not feature_label:
        feature_label = feature
    if not target_label:
        target_label = target_feature

    # Filter for top 20 categories based on count if top_20 is True
    if top_20:
        top_categories = df[feature].value_counts().head(20).index
        df = df[df[feature].isin(top_categories)]

    # Create a contingency table for counts
    crosstab_count = pd.crosstab(df[feature], df[target_feature])

    # Calculate percentages
    crosstab_percent = crosstab_count.div(crosstab_count.sum(axis=1), axis=0) * 100

    # Initialize the subplot
    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(20, 20))

    # Count plot for the feature
    sns.countplot(x=feature, hue=target_feature, data=df, ax=axs[0], order=top_categories if top_20 else None)
    axs[0].set_title(f'Count Plot of {feature_label} by {target_label}')
    axs[0].set_xlabel(feature_label)
    axs[0].set_ylabel('Count')
    axs[0].legend(title=target_label)
    axs[0].tick_params(axis='x', rotation=90)

    # Stacked bar plot using the crosstab
    crosstab_percent.plot(kind='bar', stacked=True, ax=axs[1])
    axs[1].set_title(f'Stacked Bar Plot of {feature_label} by {target_label}')
    axs[1].set_xlabel(feature_label)
    axs[1].set_ylabel('Percentage')
    axs[1].tick_params(axis='x', rotation=90)

    # Annotate the stacked bar plot with percentages
    for rect in axs[1].patches:
        height = rect.get_height()
        x = rect.get_x() + rect.get_width() / 2
        y = rect.get_y() + height
        label_text = f'{height:.1f}%'
        if height > 0:  # Only add text if there's room
            axs[1].text(x, y, label_text, ha='center', va='bottom', fontsize=8, color='black')

    # Heat map for the crosstab
    sns.heatmap(crosstab_count, annot=True, fmt='d', cmap='Blues', ax=axs[2])
    axs[2].set_title(f'Heat Map of {feature_label} by {target_label}')
    axs[2].set_xlabel(target_label)
    axs[2].set_ylabel(feature_label)

    # Ensure the layout is tight so plots don't overlap
    plt.tight_layout()
    plt.show()

def cap_outliers(df, feature):
    # Calculate the IQR
    Q1 = df[feature].quantile(0.25)
    Q3 = df[feature].quantile(0.75)
    IQR = Q3 - Q1

    # Define the bounds for capping
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Cap the outliers
    df[feature+ "_capped"] = df[feature].clip(lower=lower_bound, upper=upper_bound)

def create_date_features(df, date_feature):
    """
    Create new date-related features from a specified date column in a DataFrame.

    Parameters:
    df (DataFrame): Pandas DataFrame containing the date column.
    date_feature (str): Name of the column with date values.

    Returns:
    DataFrame: Original DataFrame with new date-related features.
    """

    # Extract various components from the date
    df[date_feature + '_year'] = df[date_feature].dt.year
    df[date_feature + '_month'] = df[date_feature].dt.month
    # df[date_feature + '_day'] = df[date_feature].dt.day
    df[date_feature + '_quarter'] = df[date_feature].dt.quarter
    df[date_feature + '_dayofweek'] = df[date_feature].dt.dayofweek
    df[date_feature + '_weekofyear'] = df[date_feature].dt.isocalendar().week
    df[date_feature + '_month_name'] = df[date_feature].dt.month_name()
    df[date_feature + '_day_name'] = df[date_feature].dt.day_name()
    df[date_feature + '_year_month'] = df[date_feature].dt.to_period('M')

import matplotlib.pyplot as plt
import seaborn as sns

def univariate_date_analysis(df, date_feature):
    """
    Perform univariate analysis on date-related features created from the original date column.

    Parameters:
    df (DataFrame): Pandas DataFrame containing the date-related features.
    date_feature (str): Base name of the original date column from which the features were derived.

    Returns:
    None
    """
    date_features = [col for col in df.columns if col.startswith(date_feature)]
    fig, axes = plt.subplots(len(date_features), 1, figsize=(20, 5 * len(date_features)))

    if len(date_features) == 1:  # In case there is only one date feature
        axes = [axes]
    
    for ax, feature in zip(axes, date_features):
        if 'year_month' in feature:
            # For year_month, we want a lineplot over time
            temp_series = df[feature].value_counts().sort_index()
            ax.plot(temp_series.index.astype(str), temp_series.values)
            ax.set_title(f'Trend Over Time for {feature}')
            ax.set_xlabel('Date')
            ax.set_ylabel('Count')
            ax.tick_params(axis='x', rotation=45)
        else:
            # For other features, we can use a simple count plot
            sns.countplot(x=feature, data=df, ax=ax)
            ax.set_title(f'Distribution for {feature}')
            ax.set_xlabel('Value')
            ax.set_ylabel('Count')
            ax.tick_params(axis='x', rotation=45)
            
    plt.tight_layout()
    plt.show()

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def bivariate_date_analysis(df, date_feature, target_feature):
    """
    Perform bivariate analysis on date-related features with a target feature (e.g., loan_status) in a DataFrame.

    Parameters:
    df (DataFrame): The Pandas DataFrame containing the features.
    date_feature (str): The base name of the date-related features (e.g., 'issue_d' if you have 'issue_d_year', 'issue_d_month', etc.).
    target_feature (str): The target feature for bivariate analysis (e.g., 'loan_status').
    """
    # List of generated date features
    date_features = [col for col in df.columns if (col.startswith(date_feature) and col!=date_feature) ]
    
    # Check if these features exist in the dataframe
    date_features = [feature for feature in date_features if feature in df.columns]
    
        # Determine the number of rows needed for subplots based on the number of date features
    num_rows = len(date_features)
    
    # Set up the matplotlib figure
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(num_rows, 1, figsize=(20, 5 * num_rows), sharex=False)
    
    # Flatten axes array if only one row
    if num_rows == 1:
        axes = [axes]
    
    # Iterate over each date feature and create a subplot
    for i, feature in enumerate(date_features):
        ax = axes[i]
        
        # Choose the appropriate plot type based on the feature
        if "year" in feature or "month" in feature or "day" in feature:
            sns.countplot(x=feature, hue=target_feature, data=df, palette="viridis", ax=ax)
        elif "quarter" in feature or "dayofweek" in feature:
            sns.barplot(x=feature, y=target_feature, data=df, estimator=lambda x: len(x) / len(df) * 100, palette="viridis", ax=ax)
        elif "weekofyear" in feature:
            sns.lineplot(x=feature, y=target_feature, data=df, estimator=lambda x: len(x) / len(df) * 100, palette="viridis", ax=ax)
        
        ax.set_title(f'Bivariate Analysis of {feature.capitalize()} with {target_feature.capitalize()}')
        ax.set_ylabel('Percentage' if 'weekofyear' in feature else 'Count')
        ax.set_xlabel(feature.capitalize())
        ax.legend(title=target_feature.capitalize())
    
    # Adjust the layout
    plt.tight_layout()
    plt.show()

def bucketize(df, feature, bins, labels):
    # Check if the length of bins is one more than the length of labels
    if len(bins) != len(labels) + 1:
        raise ValueError("The number of bins must be exactly one more than the number of labels.")

    # Create a new column for the feature category
    df[feature + '_category'] = pd.cut(df[feature], bins=bins, labels=labels, include_lowest=True)
