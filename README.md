# CodeAlpha_FAQChatbot
FAQ Chatbot built for the CodeAlpha AI Internship (Task 2).

## How it works
1. FAQs are stored in `faqs.json`
2. Text is preprocessed with NLTK (tokenize, stopword removal, lemmatization)
3. TF-IDF vectors are built for all FAQ questions
4. The user's question is matched using cosine similarity and the best answer is shown in a chat UI (Streamlit)

## Run
```
pip install -r requirements.txt
streamlit run app.py
```
