import streamlit as st
import numpy as np

# Placeholder function to simulate model prediction
def predict(sepal_length, sepal_width, petal_length, petal_width):
    # Replace this with actual model prediction logic
    # For now, it randomly predicts one of the Iris species
    species = np.random.choice(["Iris-setosa", "Iris-versicolor", "Iris-virginica"])
    return species

def main():
    st.title("Project Model Deployment")

    # Input fields for features
    st.header("Input Features")
    sepal_length = st.slider("sepal_length", min_value=0.0, max_value=10.0, value=3.28, step=0.01)
    sepal_width = st.slider("sepal_width", min_value=0.0, max_value=10.0, value=4.72, step=0.01)
    petal_length = st.slider("petal_length", min_value=0.0, max_value=10.0, value=3.05, step=0.01)
    petal_width = st.slider("petal_width", min_value=0.0, max_value=10.0, value=2.12, step=0.01)

    # Make Prediction button
    if st.button("Make Prediction"):
        # Prepare input data
        input_data = np.array([sepal_length, sepal_width, petal_length, petal_width]).reshape(1, -1)
        
        # Get prediction
        prediction = predict(sepal_length, sepal_width, petal_length, petal_width)
        
        # Display prediction
        st.success(f"The prediction is: {prediction}")

if __name__ == '__main__':
    main()
