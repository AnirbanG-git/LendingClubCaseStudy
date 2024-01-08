import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
from matplotlib.table import Table

import warnings

warnings.filterwarnings('ignore')

# This function calculates the count and percentage of null values in each column of a DataFrame.
# It returns these statistics both separately and combined in a summary DataFrame.
def get_null_stats(df):

    null_counts = df.isnull().sum()

    #Create a pandas series with null percentage stats
    null_percentage = round((df.isnull().sum() / df.shape[0]) * 100, 2)
    # Combine counts and percentages into a single DataFrame
    null_summary = pd.DataFrame({'Count': null_counts, 'Percentage': null_percentage})

    return null_counts, null_percentage, null_summary

# This function prints the count and percentage of null values for each column in a DataFrame.
# It utilizes the get_null_stats function to obtain the necessary statistics.
def print_null_stats(df):
    null_counts, null_percentage, null_summary = get_null_stats(df)
    print(null_counts)
    print(null_percentage)
    print(null_summary)

# This function removes columns from a DataFrame where the percentage of null values exceeds a specified threshold.
# It modifies the DataFrame in place, dropping columns with high proportions of null values.
def drop_high_null_cols(df, null_percentage):
    # Identify columns where the percentage of null values is greater than 60%
    columns_to_drop = null_percentage[null_percentage > 60].index

    # Drop these columns from the DataFrame
    df.drop(columns=columns_to_drop, inplace=True)

# This function prints and saves a summary of each column's first value and its description from a data dictionary.
# It merges the first row of the DataFrame with a data dictionary for detailed insight into each column.
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

# This function prints the number of unique values and the top 10 unique values for each column in a DataFrame.
# It is useful for getting a quick overview of the variety in each column.
def print_unique_values(df):
    for column in df.columns:
        unique_values = df[column].unique()
        num_unique_values = len(unique_values)

        # Print the total number of unique values
        print(f"Column '{column}' has {num_unique_values} unique values.")

        # Print the top 10 unique values
        print(f"Top 10 unique values: {unique_values[:10]}")
        print("\n")  # New line for better readability between columns

# This function combines printing unique values and null statistics for each column in a DataFrame.
# It provides a comprehensive view of the data's uniqueness and completeness.
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

# This function creates count plots for all categorical columns in a DataFrame.
# It fills missing values with 'NA' and uses the plot_categorical function for each categorical column.
def plot_all_categorical(df):
    df_copy = df.copy()
    # Categorical Features
    categorical_features = df_copy.select_dtypes(include=[object]).columns
    for feature in categorical_features:
        df_copy[feature].fillna("NA", inplace=True)
        plot_categorical(df, feature)

# This function generates univariate plots for all numerical columns in a DataFrame.
# It fills missing values with 0 and uses the plot_numerical_univariate function for each numerical column.
def plot_all_numerical_univariate(df):
    df_copy = df.copy()
    # Numerical Features
    numerical_features = df_copy.select_dtypes(include=[np.number]).columns
    for feature in numerical_features:
        df_copy[feature].fillna(0, inplace=True)
        plot_numerical_univariate(df_copy, feature)

# This function converts specified columns in a DataFrame from strings to numeric values, handling percentage strings.
# It is particularly useful for converting percentage formatted strings to float values.
def convert_to_numeric(df, features):
    for feature in features:
        df[feature] = pd.to_numeric(df[feature].str.rstrip('%'), errors='coerce')

# This function creates a count plot for a specific categorical column in a DataFrame, grouped by 'loan_status'.
# It displays the top 30 values by count and is useful for visualizing the distribution of categories.
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

# This function visualizes the distribution of a numerical column using histograms and boxplots.
# It includes statistical descriptions and categorizes data into bins for detailed analysis.
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

# This function visualizes the relationship between a numerical and a categorical column using box and violin plots.
# It is useful for exploring how a numerical feature varies with different categories of a target feature.
def plot_numerical_bivariate(df, feature, target_feature, feature_label="", target_label=""):
    if not feature_label:
        feature_label = feature
    if not target_label:
        target_label = target_feature
    
    # Define a custom color palette
    # The colors correspond to the default Seaborn color palette
    default_colors = sns.color_palette()  # Get the default color palette
    category_colors = {"Fully Paid": default_colors[0], "Charged Off": default_colors[1]}

    # Initialize the subplot with 2 rows and 1 column
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(20, 8))

    # Boxplot
    sns.boxplot(x=target_feature, y=feature, data=df, ax=axs[0], 
                palette=[category_colors.get(x, "#333333") for x in df[target_feature].unique()])
    axs[0].set_title(f'Box Plot of {feature_label} by {target_label}')
    axs[0].set_ylabel(feature_label)
    axs[0].set_xlabel(target_label)

    # Violin plot
    sns.violinplot(x=target_feature, y=feature, data=df, ax=axs[1], 
                   palette=[category_colors.get(x, "#333333") for x in df[target_feature].unique()])
    axs[1].set_title(f'Violin Plot of {feature_label} by {target_label}')
    axs[1].set_ylabel(feature_label)
    axs[1].set_xlabel(target_label)

    plt.tight_layout()
    plt.show()

# This function visualizes the distribution of a categorical column using bar and pie charts.
# It includes options for focusing on the top 20 categories and annotating the plots with percentages.
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
    
    # Initialize the subplot with 3 rows and 1 column
    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(20, 18))
    
    # Bar plot for the frequency of each category
    barplot = sns.countplot(x=feature, data=df, ax=axs[0], order=value_counts.index)
    axs[0].set_title(f'Count Plot of {feature_label}')
    axs[0].set_xlabel(feature_label)
    axs[0].set_ylabel('Count')
    axs[0].tick_params(axis='x', rotation=90)
    
    # Annotate the bar plot with the percentage
    for p in barplot.patches:
        height = p.get_height()
        barplot.text(p.get_x() + p.get_width() / 2.,
                     height + 3,
                     '{:1.2f}%'.format((height/len(df))*100),
                     ha="center", va='bottom', rotation=0)
    
    # Percentage bar chart for each category
    (proportions*100).plot(kind='bar', ax=axs[1])
    axs[1].set_title(f'Percentage of {feature_label}')
    axs[1].set_xlabel(feature_label)
    axs[1].set_ylabel('Percentage')
    axs[1].tick_params(axis='x', rotation=90)
    # Annotate the percentage bar chart
    for p in axs[1].patches:
        axs[1].annotate(format(p.get_height(), '.1f') + '%', 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha = 'center', va = 'center', 
                        xytext = (0, 9), 
                        textcoords = 'offset points')
    
    # Pie chart for the proportion of each category
    proportions.plot(kind='pie', autopct='%1.1f%%', startangle=90, ax=axs[2])
    axs[2].set_title(f'Proportion of {feature_label}')
    axs[2].set_ylabel('')  # Hide the y-axis label for the pie chart
    axs[2].axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    
    plt.tight_layout()
    plt.show()

# This function creates visualizations to analyze the relationship between two categorical variables.
# It includes count, stacked bar, and heatmap plots, with options for top categories and sorted views.
def plot_categorical_bivariate(df, feature, target_feature, feature_label="", target_label="", 
                               top_20=False, include_tables=False, sort_by=None):
    if not feature_label:
        feature_label = feature
    if not target_label:
        target_label = target_feature

    # Defaulting to one of the target categories if sort_by is not provided
    if sort_by is None:
        sort_by = df[target_feature].unique()[0]

    # Filter for top 20 categories based on count if top_20 is True
    if top_20:
        top_categories = df[feature].value_counts().head(20).index
        df = df[df[feature].isin(top_categories)]

    # Create a contingency table for counts
    crosstab_count = pd.crosstab(df[feature], df[target_feature], margins=True, margins_name='Total')

    # Calculate count of sort_by category for each feature
    sort_by_counts = df[df[target_feature] == sort_by].groupby(feature).size()

    # Sort features based on the count of sort_by category
    sorted_features_by_sort_by_count = sort_by_counts.sort_values(ascending=False).index.tolist()

    # Filter and sort the dataframe for count plot
    df_sorted_for_countplot = df[df[feature].isin(sorted_features_by_sort_by_count)]
    df_sorted_for_countplot[feature] = pd.Categorical(df_sorted_for_countplot[feature], categories=sorted_features_by_sort_by_count, ordered=True)

    # Define color mapping for the two categories
    color_mapping = {df[target_feature].unique()[0]: "#FF7F0F", df[target_feature].unique()[1]: "#3274A1"}

    # Determine the number of rows for the subplot
    nrows = 3 if not include_tables else 5

    # Initialize the subplot
    fig, axs = plt.subplots(nrows=nrows, ncols=1, figsize=(20, 5*nrows),
                            gridspec_kw={'height_ratios': [2]*3 + [0.5]*2 if include_tables else [2]*3})

    # Count plot for the feature, sorted by sort_by count
    sns.countplot(x=feature, hue=target_feature, data=df_sorted_for_countplot, palette=color_mapping, ax=axs[0])
    axs[0].set_title(f'Count Plot of {feature_label} by {target_label}')
    axs[0].set_xlabel(feature_label)
    axs[0].set_ylabel('Count')
    axs[0].legend(title=target_label)
    axs[0].tick_params(axis='x', rotation=90)

    # Stacked bar plot using the sorted crosstab
    sorted_percent = crosstab_count.div(crosstab_count['Total'], axis=0) * 100
    sorted_percent_sorted = sorted_percent.sort_values(f'{sort_by}', ascending=False)

    # Exclude 'Total' column and row for the plot
    sorted_percent_sorted = sorted_percent_sorted.drop(columns=['Total']).drop('Total')

    stacked_bar = sorted_percent_sorted.plot(kind='bar', stacked=True, ax=axs[1], color=[color_mapping.get(x, "#333333") for x in sorted_percent.columns[:-1]])
    axs[1].set_title(f'Stacked Bar Plot of {feature_label} by {target_label}')
    axs[1].set_xlabel(feature_label)
    axs[1].set_ylabel('Percentage')
    axs[1].tick_params(axis='x', rotation=90)

    # Annotate stacked bar plot with percentages
    for bars in stacked_bar.containers:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                axs[1].annotate(f'{height:.2f}%', 
                                (bar.get_x() + bar.get_width() / 2, bar.get_y() + height/2),
                                ha='center', va='center', fontsize=9, color='white')

    # Heat map for the crosstab minus the total column and row
    sns.heatmap(crosstab_count.iloc[:-1, :-1], annot=True, fmt='d', cmap='Blues', ax=axs[2])
    axs[2].set_title(f'Heat Map of {feature_label} by {target_label}')
    axs[2].set_xlabel(target_label)
    axs[2].set_ylabel(feature_label)

    if include_tables:
        # Function to create a table
        def create_table(ax, data, title):
            ax.axis('tight')
            ax.axis('off')
            ax.set_title(title)
            table = ax.table(
                cellText=data.values,
                colLabels=data.columns,
                cellLoc='center',
                loc='center'
            )
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.2)

        # Top 5 categories sorted by proportion
        top_5_data = sorted_percent_sorted.reset_index().head(5)
        create_table(axs[3], top_5_data, f'Top 5 Categories by {sort_by} Proportion')

        # Bottom 5 categories sorted by proportion
        bottom_5_data = sorted_percent_sorted.reset_index().tail(5)
        create_table(axs[4], bottom_5_data, f'Bottom 5 Categories by {sort_by} Proportion')

    plt.tight_layout()
    plt.show()

# This function caps outliers in a numerical column based on the Interquartile Range (IQR) method.
# It creates a new column in the DataFrame where values outside the IQR are capped at determined bounds.
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

# This function generates new date-related features from a specified date column in a DataFrame.
# It extracts components like year, month, quarter, etc., creating multiple new columns for detailed analysis.
def create_date_features(df, date_feature):

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

# This function performs univariate analysis on date-related features in a DataFrame.
# It visualizes the distribution of each date component, such as year, month, or day of the week.
def univariate_date_analysis(df, date_feature):
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


# This function conducts bivariate analysis between date-related features and a target feature in a DataFrame.
# It uses count, bar, and line plots to examine how date components relate to the target feature.
def bivariate_date_analysis(df, date_feature, target_feature):
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

# This function categorizes a numerical column into discrete buckets based
def bucketize(df, feature, bins, labels):
    # Check if the length of bins is one more than the length of labels
    if len(bins) != len(labels) + 1:
        raise ValueError("The number of bins must be exactly one more than the number of labels.")

    # Create a new column for the feature category
    df[feature + '_category'] = pd.cut(df[feature], bins=bins, labels=labels, include_lowest=True)

def get_iqr(df, feature):
    Q1 = df[feature].quantile(0.25)
    Q3 = df[feature].quantile(0.75)
    IQR = Q3 - Q1

    return Q1, Q3, IQR

def get_outlier_bounds(df, feature):
    Q1, Q3, IQR = get_iqr(df, feature)
    lower_bound = Q3 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    print(f"The lower(Q3 - 1.5 * IQR) outlier bound for {feature} is: {lower_bound}")
    print(f"The upper(Q3 + 1.5 * IQR) outlier bound for {feature} is: {upper_bound}")

    return lower_bound, upper_bound