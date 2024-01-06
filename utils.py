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
        print(f"Value counts for '{feature}':")
        value_counts = df_copy[feature].value_counts()[:30]  # Select the top 30 values
        print(value_counts)
        print("="*40)
        
        plt.figure(figsize=(15, 4))
        sns.countplot(data=df_copy, x=feature, hue='loan_status', order=value_counts.index)  # Use the top 30 values for ordering
        plt.title(f'{feature} by loan_status')
        plt.xticks(rotation=45)
        plt.show()

def plot_all_numerical(df):
    df_copy = df.copy()
    # Numerical Features
    numerical_features = df_copy.select_dtypes(include=[np.number]).columns
    for feature in numerical_features:
        df_copy[feature].fillna(0, inplace=True)
        print(f"Summary statistics for '{feature}':")
        print(df_copy[feature].describe())
        print("="*40)
        plt.figure(figsize=(15, 4))
        plt.subplot(1, 2, 1)
        sns.histplot(df_copy[feature], bins=20, kde=True)
        plt.title(f'Distribution of {feature}')
        plt.subplot(1, 2, 2)
        sns.boxplot(x='loan_status', y=feature, data=df_copy)
        plt.title(f'{feature} by loan_status')
        plt.show()






