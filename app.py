import streamlit as st
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

#digitsデータセットの読み込み
digits = load_digits()
X = digits.data
y = digits.target

#テストデータと訓練データに分類
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#ランダム森を用いて訓練
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

#テストデータを用いて訓練し精度を計算
yhat = clf.predict(X_test)
acc = accuracy_score(y_test, yhat)

st.title('Handwritten Digit Classifier')
st.write('精度：', acc)

#ファイルアップローダーのウィジェットを作成
uploaded_file = st.file_uploader('画像ファイルを選択してください。', type=['jpg', 'png'])
if uploaded_file is not None:
    #画像を読み込んで表示
    image = plt.imread(uploaded_file)
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.axis('off')
    st.pyplot()

    #アップロードされた画像に分類器を当てはめる
    image = image.reshape(-1, 8*8)
    pred = clf.predict(image)
    st.write('Predicted digit:', pred[0])
