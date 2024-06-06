import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Define the file path (adjust the path if necessary)
file_path = 'C:/Users/CLUE/Desktop/diabetes.csv'

# Load the dataset
df = pd.read_csv(file_path)

# Define features (X) and target (y)
X = df.drop(columns=['Outcome'])
y = df['Outcome']

# Split the dataset into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Naive Bayes classifier
nb_classifier = GaussianNB()

# Train the classifier
nb_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = nb_classifier.predict(X_test)

# Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)

# Print the accuracy
print(f'The accuracy of the Naive Bayes classifier is: {accuracy:.2%}')
