import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load the Iris dataset
iris = pd.read_csv('IRIS.csv')
iris_features = iris.drop('species', axis=1)

# Display the tennis dataset
tennis_data = {
    'Outlook': ['Sunny', 'Sunny', 'Sunny', 'Sunny', 'Sunny', 'Overcast', 'Overcast', 'Overcast', 'Overcast', 'Rain', 'Rain', 'Rain', 'Rain', 'Rain'],
    'Temperature': ['Hot', 'Hot', 'Mild', 'Cool', 'Mild', 'Hot', 'Cool', 'Mild', 'Hot', 'Mild', 'Cool', 'Mild', 'Mild', 'Mild'],
    'Humidity': ['High', 'High', 'High', 'Normal', 'Normal', 'High', 'Normal', 'High', 'Normal', 'Normal', 'Normal', 'Normal', 'Normal', 'High'],
    'Wind': ['Weak', 'Strong', 'Weak', 'Weak', 'Strong', 'Weak', 'Strong', 'Strong', 'Weak', 'Weak', 'Strong', 'Weak', 'Strong', 'Strong'],
    'Play Tennis?': ['No', 'No', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No']
}

tennis_df = pd.DataFrame(tennis_data)

# Encode categorical variables
le = LabelEncoder()
for column in tennis_df.columns[:-1]:
    tennis_df[column] = le.fit_transform(tennis_df[column])

tennis_df['Play Tennis?'] = le.fit_transform(tennis_df['Play Tennis?'])

# Combine the tennis features with the Iris features for scaling
combined_features = pd.concat([tennis_df.drop('Play Tennis?', axis=1), iris_features], axis=0)

# Standardize the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(combined_features)

# Separate the scaled tennis data
scaled_tennis_features = scaled_features[:tennis_df.shape[0]]

# Prepare the final dataset
X = scaled_tennis_features
y = tennis_df['Play Tennis?']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the neural network
model = Sequential([
    Dense(8, input_dim=X_train.shape[1], activation='relu'),
    Dense(4, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=4, verbose=1)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Loss: {loss}, Accuracy: {accuracy}')

# Make predictions
predictions = model.predict(X_test)
predictions = (predictions > 0.5).astype(int)
print(predictions)
