#importing all libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("Iris.csv",sep=r"\s+")

# Show columns
print("Columns:")
print(df.columns)

# Features (inputs)
X = df.drop(columns=["species"])

# Target (output)
y = df["species"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create model
model = DecisionTreeClassifier(random_state=42)

# Train model
model.fit(X_train, y_train)

# Predict test data
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)

# Print decision tree rules
rules = export_text(model, feature_names=list(X.columns))

print("\nDecision Tree Rules:")
print(rules)