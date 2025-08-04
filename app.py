# import pickle
# from flask import Flask, request, render_template

# # Load the trained model and vectorizer
# with open("model.pkl", "rb") as f:
#     model = pickle.load(f)

# with open("vectorizer.pkl", "rb") as f:
#     vectorizer = pickle.load(f)

# # Initialize Flask app
# app = Flask(__name__)

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     title = request.form['title']
#     summary = request.form['summary']

#     text = (title + " " + summary).lower()
#     vec = vectorizer.transform([text])

#     predicted_genre = model.predict(vec)[0]

#     result = f"Predicted Genre: <strong>{predicted_genre}</strong>"
    
#     return render_template("index.html",
#                            prediction_text=result,
#                            title=title,
#                            summary=summary)

# if __name__ == '__main__':  
#     app.run(debug=True)
#***************************************************************
import streamlit as st
import pickle

# Load model and vectorizer
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# App UI
st.title("📚 Book Genre Predictor")
st.write("Enter a book title and summary, and get the predicted genre!")

book_name = st.text_input("Book Name")
summary = st.text_area("Book Summary")

if st.button("Predict Genre"):
    if not book_name or not summary:
        st.warning("Please enter both fields.")
    else:
        # Preprocess and predict
        text = (book_name + " " + summary).lower()
        text_vec = vectorizer.transform([text])
        prediction = model.predict(text_vec)[0]
        
        st.success(f"🧠 Predicted Genre: **{prediction}**")
