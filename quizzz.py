import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn import preprocessing

# Define the file path (adjust the path if necessary)
file_path = 'C:/Users/CLUE/Desktop/diabetes.csv'

# Load the dataset
df = pd.read_csv(file_path)

# Display the first few rows of the dataset
print(df.head())

# Define features (X) and target (y)
X = df.drop(columns=['Outcome'])
y = df['Outcome']

# Split the dataset into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Naive Bayes classifier
model = GaussianNB()

# Train the classifier
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)

# Print the accuracy
print(f'The accuracy of the Naive Bayes classifier is: {accuracy:.2%}')

# Adding an example to predict with the first row of the test set
example_row = X_test.iloc[0].values.reshape(1, -1)
predicted = model.predict(example_row)
predicted_proba = model.predict_proba(example_row)

print(f'Example row: {X_test.iloc[0].values}')
print(f'Predicted outcome: {predicted[0]}')
print(f'Predicted probabilities: {predicted_proba[0]}')

# Display a few rows of the training set with actual values for verification
print('\nSample of the training set:')
print(X_train.head())
print('\nActual outcomes:')
print(y_train.head())

