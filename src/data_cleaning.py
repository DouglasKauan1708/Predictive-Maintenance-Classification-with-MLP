# Import required libraries
import pandas as pd
import warnings
import wandb
warnings.filterwarnings('ignore')

def data_cleaning(df): 

    # Removing columns
    clean_df = df.copy()
    clean_df = clean_df.drop(columns=["UDI","Product ID", "Target"])

    # Encoding
    failure_map = {'No Failure': 0,
                'Power Failure': 1,
                'Overstrain Failure': 2,
                'Heat Dissipation Failure': 3,
                'Tool Wear Failure': 4,
                'Random Failures': 5}
    clean_df['Failure Type'] = clean_df['Failure Type'].map(failure_map)
    clean_df = pd.get_dummies(clean_df, columns=['Type'], drop_first=True)
    clean_df = df.drop(columns=["Type_L", "Type_M"]) # Unrelevant columns

    # Separating our columns
    X = clean_df.drop(columns=["Failure Type"])
    y = clean_df["Failure Type"]

    return clean_df, X, y