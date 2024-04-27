from sklearn.model_selection import train_test_split

def split_data(df, test_size=0.1, random_state=42):
    df.drop(columns='name', inplace=True)
    x = df.drop(columns='status')
    y = df['status']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)
    return x_train, x_test, y_train, y_test