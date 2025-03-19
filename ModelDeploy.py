from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

class RandomForestIrisClassifier:
    def __init__(self, n_estimators=100, test_size=0.3, random_state=42):
        # Initialize the classifier's parameters
        self.n_estimators = n_estimators
        self.test_size = test_size
        self.random_state = random_state
        
        # Load the Iris dataset
        self.iris = load_iris()
        self.X = self.iris.data
        self.y = self.iris.target
        
        # Split the dataset into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, random_state=self.random_state
        )
        
        # Initialize the Random Forest classifier
        self.rf_classifier = RandomForestClassifier(
            n_estimators=self.n_estimators, random_state=self.random_state
        )
        
    def train(self):
        """Train the Random Forest classifier on the training data."""
        self.rf_classifier.fit(self.X_train, self.y_train)
        
    def evaluate(self):
        """Evaluate the model using accuracy and classification report."""
        # Make predictions on the test set
        y_pred = self.rf_classifier.predict(self.X_test)
        
        # Calculate accuracy
        accuracy = accuracy_score(self.y_test, y_pred)
        print(f"Accuracy: {accuracy * 100:.2f}%")
        
        # Generate and print the classification report
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred, target_names=self.iris.target_names))

# Example usage:
if __name__ == "__main__":
    # Create an instance of the classifier
    rf_classifier = RandomForestIrisClassifier(n_estimators=100, test_size=0.3, random_state=42)
    
    # Train the model
    rf_classifier.train()
    
    # Evaluate the model
    rf_classifier.evaluate()