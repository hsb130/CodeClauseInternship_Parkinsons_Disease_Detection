from sklearn.model_selection import GridSearchCV

def perform_grid_search(pipeline, param_grid, x_train, y_train):
    # Create GridSearchCV object
    grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    
    # Perform grid search
    grid_search.fit(x_train, y_train)
    
    # Return the best estimator
    return grid_search.best_estimator_
