import pandas as pd
import matplotlib.pyplot as plt


def load_data(data):
    """ Load the list of dictionaries into a pandas DataFrame. """
    return pd.DataFrame(data)


def auto_one_hot_encode(df):
    """ Automatically detects string columns to one-hot encode them. """
    encoded_dfs = {}
    # Detecting columns that contain string data (commonly used as categorical variables)
    categorical_vars = df.select_dtypes(include=['object']).columns
    for column in categorical_vars:

        expanded_df = df[column].str.get_dummies(sep=',')
        # Rename columns
        expanded_df.columns = [f"{column}_{subcat}" for subcat in expanded_df.columns]
        # Concatenate the one-hot encoded columns back to the original DataFrame
        df = pd.concat([df, expanded_df], axis=1)
        df.drop(column, axis=1, inplace=True)  # Drop original column if no longer needed
        encoded_dfs[column] = expanded_df

        # Plot dist
        plt.figure(figsize=(10, 5))
        expanded_df.sum().plot(kind='bar')
        plt.title(f'Frequency of {column} categories')
        plt.ylabel('Frequency')
        plt.xlabel('Category')
        plt.show()

    return df, encoded_dfs


# Example Usage
data = [
    {"name": "Alice", "type": "A,B", "color": "Red"},
    {"name": "Bob", "type": "B", "color": "Blue,Green", "Animal": "Cat"},
    {"name": "Charlie", "type": "A,C", "color": "Red", "Animal": "Dog, Cat"}
]

# Load the data
df = load_data(data)

# Automatically detect and one-hot encode categorical variables
df_encoded, individual_encoded_dfs = auto_one_hot_encode(df)

print(individual_encoded_dfs)