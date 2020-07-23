import pandas as pd

# reading csv file
data = pd.read_csv('https://s3.amazonaws.com/csvpastebin/uploads/327fb8f66f90a98c2ed4454665efae9d/weather.csv')

# shape of dataset
print("Shape:", data.shape)

# column names
print("\nFeatures:", data.columns)

# storing the feature matrix (X) and response vector (y)
X = data[data.columns[:-1]]
y = data[data.columns[-1]]

# printing first 5 rows of feature matrix
print("\nFeature matrix:\n", X.head())

# printing first 5 values of response vector
print("\nResponse vector:\n", y.head())

feature_names = data.feature_names
target_names = data.target_names

# printing features and target names of our dataset
print("Feature names:", feature_names)
print("Target names:", target_names)
