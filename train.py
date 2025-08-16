# import pandas as pd
# import pickle
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import classification_report

# # ----------------------------------------------------------
# # STEP 1: Load and Prepare Data
# # ----------------------------------------------------------
# print("Loading and preparing data...")
# data = pd.read_csv("cleaned_books_dataset.csv")
# data.dropna(subset=['book_name', 'summary', 'genre'], inplace=True)

# # Combine book name and summary into a single text field
# data['text'] = (data['book_name'] + " " + data['summary']).str.lower()

# X = data['text']
# y = data['genre']

# # ----------------------------------------------------------
# # STEP 2: TF-IDF Vectorization
# # ----------------------------------------------------------
# print("Performing TF-IDF vectorization...")
# vectorizer = TfidfVectorizer(
#     max_features=10000,
#     stop_words='english',
#     ngram_range=(1, 4),
#     min_df=2,
#     max_df=0.85
# )
# X_vec = vectorizer.fit_transform(X)

# # ----------------------------------------------------------
# # STEP 3: Train-Test Split
# # ----------------------------------------------------------
# X_train, X_test, y_train, y_test = train_test_split(
#     X_vec, y, test_size=0.3, random_state=42
# )

# # ----------------------------------------------------------
# # STEP 4: Model Training (Random Forest)
# # ----------------------------------------------------------
# print("Training Random Forest classifier...")
# model = RandomForestClassifier(
#     n_estimators=300,
#     max_depth=50,
#     min_samples_split=4,
#     min_samples_leaf=2,
#     random_state=42
# )
# model.fit(X_train, y_train)

# # ----------------------------------------------------------
# # STEP 5: Model Evaluation
# # ----------------------------------------------------------
# print("Evaluating model...")
# y_pred = model.predict(X_test)
# report = classification_report(y_test, y_pred, zero_division=0)
# print("Model Evaluation:\n", report)

# # Save report to file
# with open("report.txt", "w") as f:
#     f.write(report)

# # ----------------------------------------------------------
# # STEP 6: Save Model and Vectorizer
# # ----------------------------------------------------------
# print("Saving model and vectorizer...")
# with open("model.pkl", "wb") as f:
#     pickle.dump(model, f)

# with open("vectorizer.pkl", "wb") as f:
#     pickle.dump(vectorizer, f)

# print("Training complete. Files saved.")
#accuracy is 66

    #******************************************************************************* 
    
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, f1_score
import numpy as np

# ----------------------------------------------------------
# STEP 1: Load and Prepare Data
# ----------------------------------------------------------
print("Loading and preparing data...")
data = pd.read_csv("cleaned_books_dataset.csv")
data.dropna(subset=['book_name', 'summary', 'genre'], inplace=True)

# Combine title and summary into one feature
data['text'] = (data['book_name'] + " " + data['summary']).str.lower()

X = data['text']
y = data['genre']

# ----------------------------------------------------------
# STEP 2: TF-IDF Vectorization
# ----------------------------------------------------------
print("Performing TF-IDF vectorization...")
vectorizer = TfidfVectorizer(
    max_features=15000,
    stop_words='english',
    ngram_range=(1, 4),
    min_df=2,
    max_df=0.9
)
X_vec = vectorizer.fit_transform(X)

# ----------------------------------------------------------
# STEP 3: Cross-Validation
# ----------------------------------------------------------
print("Performing 5-Fold Cross-Validation...")
model = MultinomialNB()
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_scores = cross_val_score(model, X_vec, y, cv=cv, scoring='f1_macro')
print("Cross-Validation Macro F1 Scores:", cv_scores)
print("Mean CV Macro F1 Score:", np.mean(cv_scores))

# ----------------------------------------------------------
# STEP 4: Train-Test Split (for final model training)
# ----------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.3, random_state=42, stratify=y
)

# ----------------------------------------------------------
# STEP 5: Final Model Training
# ----------------------------------------------------------
print("Training final Naive Bayes classifier...")
model.fit(X_train, y_train)

# ----------------------------------------------------------
# STEP 6: Final Model Evaluation
# ----------------------------------------------------------
print("Evaluating final model on test set...")
y_pred = model.predict(X_test)
report = classification_report(y_test, y_pred, zero_division=0)
macro_f1 = f1_score(y_test, y_pred, average='macro')

print("Test Set Evaluation:\n", report)
print("Test Set Macro F1 Score:", macro_f1)

# Save report to file
with open("report.txt", "w") as f:
    f.write("Cross-Validation Macro F1 Scores: " + str(cv_scores) + "\n")
    f.write(f"Mean CV Macro F1 Score: {np.mean(cv_scores):.4f}\n\n")
    f.write("Test Set Evaluation:\n")
    f.write(report)
    f.write(f"\nTest Set Macro F1 Score: {macro_f1:.4f}\n")

# ----------------------------------------------------------
# STEP 7: Save Model and Vectorizer
# ----------------------------------------------------------
print("Saving model and vectorizer...")
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("Training complete. Files saved.")