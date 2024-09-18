import pandas as pd
import matplotlib.pyplot as plt

def load_data(data):
    """ Load the list of dictionaries into a pandas DataFrame. """
    return pd.DataFrame(data)

def one_hot_encode(df, columns):
    """ One-hot encodes the specified columns with comma-separated values and concatenates them back to the original DataFrame. """
    for column in columns:
        # Splitting and getting dummies
        dummies = df[column].str.get_dummies(sep=',')
        # Prefixing new columns to avoid column name clashes
        dummies.columns = [f"{column}_{subcat}" for subcat in dummies.columns]
        # Concatenate the dummies DataFrame to the original DataFrame
        df = pd.concat([df, dummies], axis=1)
        # Optionally drop the original column if it's no longer needed
        df.drop(column, axis=1, inplace=True)
    return df

def visualize_data(df):
    """ Visualizes the frequency of categories in one-hot encoded DataFrame. """
    for column in df.columns:
        plt.figure(figsize=(10, 5))
        df[column].value_counts().plot(kind='bar')
        plt.title(f'Frequency of {column}')
        plt.ylabel('Frequency')
        plt.xlabel('Category')
        plt.show()

# Example Usage
data = [
    {"name": "Alice", "type": "A,B", "color": "Red"},
    {"name": "Bob", "type": "B", "color": "Blue,Green", 'Food': "Apple,Pear"},
    {"name": "Charlie", "type": "A,C", "color": "Red", "Food": "Apple"}
]

# Load the data
df = load_data(data)

# One-hot encode categorical variables with comma-separated values
categorical_columns = ['type', 'color']
df_encoded = one_hot_encode(df, categorical_columns)

# Visualize the results
visualize_data(df_encoded)

print(df_encoded)
