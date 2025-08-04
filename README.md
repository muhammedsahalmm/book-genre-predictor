📚 BOOK GENRE PREDICTOR

  This project predicts the genre of a book based on its title and summary using machine learning and natural language processing (NLP) techniques.
  The project includes a trained Naive Bayes classifier, TF-IDF text vectorization, and a user-friendly Streamlit web app for deployment.


FEATURES

  Predicts genres like Fantasy, Horror, Crime Fiction, Science Fiction, and more.
  Uses TF-IDF Vectorizer for feature extraction.
  Trained with a Multinomial Naive Bayes classifier.
  Simple and interactive Streamlit UI for predictions.
  Hosted and deployed using Streamlit Cloud.
  Clean and modular code structure.


TECH STACK

  Python
  Pandas, Scikit-learn
  Streamlit
  Git & GitHub


PROJECT STRUCTURE

  book-genre-predictor/
  
├── app.py                  # Streamlit web app
├── train.py                # Model training script
├── model.pkl               # Saved trained model (excluded via .gitignore)
├── vectorizer.pkl          # TF-IDF vectorizer (excluded)
├── cleaned_books_dataset.csv  # Cleaned dataset (excluded)
├── requirements.txt        # Project dependencies
└── .gitignore              # Hidden sensitive files


WORKINGS

  Text Input: The user provides a book title and summary.
  TF-IDF Vectorization: Transforms the input text into feature vectors.
  Prediction: The trained Naive Bayes model predicts the book genre.
  Display: The result is shown instantly on the web app.


TO RUN LOCALLY

  # 1. Clone the repo
  git clone https://github.com/muhammedsahalmm/book-genre-predictor.git
  cd book-genre-predictor
  
  # 2. Create virtual environment
  python -m venv .venv
  .venv\Scripts\activate    # Windows
  
  # 3. Install dependencies
  pip install -r requirements.txt
  
  # 4. Run the Streamlit app
  streamlit run app.py


🌐 Live App
👉 


ACKNOWLEDGEMENT

  Dataset: Public book metadata (pre-cleaned manually).
  Libraries: scikit-learn, Streamlit


Contact

Muhammed Sahal M M
Email: [sahalmusawirofficial@gmail.com]
GitHub: [muhammedsahalmm]

