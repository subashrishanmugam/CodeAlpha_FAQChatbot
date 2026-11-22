import json, re
import nltk
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="FAQ Chatbot", page_icon="🤖")

# ---------- NLP preprocessing (NLTK) ----------
for pkg in ["stopwords", "wordnet", "omw-1.4"]:
    try:
        nltk.download(pkg, quiet=True)
    except Exception:
        pass

try:
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    STOP = set(stopwords.words("english"))
    LEMMA = WordNetLemmatizer()
    LEMMA.lemmatize("tests")  # trigger load to check data exists
except Exception:
    STOP, LEMMA = set(), None

def preprocess(text: str) -> str:
    tokens = re.findall(r"[a-z0-9]+", text.lower())          # tokenize + clean
    tokens = [t for t in tokens if t not in STOP]            # remove stopwords
    if LEMMA:
        tokens = [LEMMA.lemmatize(t) for t in tokens]        # lemmatize
    return " ".join(tokens)

# ---------- Load FAQs & build model ----------
@st.cache_resource
def load_model():
    with open("faqs.json", encoding="utf-8") as f:
        faqs = json.load(f)
    vec = TfidfVectorizer(ngram_range=(1, 2))
    matrix = vec.fit_transform([preprocess(x["q"]) for x in faqs])
    return faqs, vec, matrix

faqs, vectorizer, faq_matrix = load_model()

def get_answer(user_q: str, threshold: float = 0.2):
    vec = vectorizer.transform([preprocess(user_q)])
    scores = cosine_similarity(vec, faq_matrix)[0]
    best = scores.argmax()
    if scores[best] < threshold:
        return "Sorry, I couldn't find an answer to that. Please email services@codealpha.tech.", scores[best]
    return faqs[best]["a"], scores[best]

# ---------- Chat UI ----------
st.title("🤖 FAQ Chatbot")
st.caption("CodeAlpha AI Internship - Task 2 | TF-IDF + Cosine Similarity")

if "history" not in st.session_state:
    st.session_state.history = [("assistant", "Hi! Ask me anything about the internship.")]

for role, msg in st.session_state.history:
    st.chat_message(role).write(msg)

if q := st.chat_input("Type your question..."):
    ans, score = get_answer(q)
    st.session_state.history += [("user", q), ("assistant", ans)]
    st.chat_message("user").write(q)
    st.chat_message("assistant").write(f"{ans}\n\n_(match score: {score:.2f})_")
