import pandas as pd
from sklearn.model_selection import train_test_split

def split_train_test (X, y, test_size: float = 0.2 , random_state : int = 42):
    X_train , X_test , y_train , y_test = train_test_split(X, y,
    test_size = test_size,
    random_state = random_state,
    stratify = y # preserves class proportions
    )

    train_df = pd.concat([X_train, y_train], axis = 1)
    test_df = pd.concat([X_test, y_test], axis = 1)

    
    return train_df, test_df