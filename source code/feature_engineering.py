from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

def create_pipeline(x_train, y_train):
    # Identify categorical and numerical columns
    categorical_columns = x_train.select_dtypes(include=['object']).columns
    numerical_columns = x_train.select_dtypes(include=['number']).columns

    # Define the preprocessing steps
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numerical_columns),
        ('cat', OneHotEncoder(drop='first'), categorical_columns),
        ('lda', LDA(n_components=1), numerical_columns)
    ])

    # Define the model training step
    model = GradientBoostingClassifier()

    # Combine preprocessing and model training into a Pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    # Fit the pipeline to the training data
    pipeline.fit(x_train, y_train)

    return pipeline
