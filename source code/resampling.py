from imblearn.over_sampling import SMOTE

def apply_smote(x_train, y_train):
    # Create an instance of SMOTE
    smote = SMOTE(random_state=42)
    
    # Apply SMOTE to the training data
    x_resampled, y_resampled = smote.fit_resample(x_train, y_train)
    
    return x_resampled, y_resampled
