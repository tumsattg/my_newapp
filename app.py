import streamlit as st
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load the digits dataset
digits = load_digits()
X = digits.data
y = digits.target

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a random forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Test the classifier on the test set
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)

st.title('Handwritten Digit Classifier')
st.write('精度：', acc)

# Create a file uploader widget
uploaded_file = st.file_uploader('画像ファイルを選択してください。', type=['jpg', 'png'])
if uploaded_file is not None:
    # Read the image and display it
    image = plt.imread(uploaded_file)
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.axis('off')
    st.pyplot()

    # Apply the classifier to the uploaded image
    image = image.reshape(-1, 8*8)
    pred = clf.predict(image)
    st.write('Predicted digit:', pred[0])
