import streamlit as st
from janome.tokenizer import Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer

# 形態素解析器
tokenizer = Tokenizer()

# ストップワード（必要に応じて追加）
stopwords = set([
    'こと', 'もの', 'これ', 'それ', 'ため', 'よう', 'さん', 'して', 'ます', 'です', 'いる', 'ある', 'する', 'なる'
])

def tokenize(text):
    tokens = tokenizer.tokenize(text)
    words = [token.base_form for token in tokens
             if token.base_form not in stopwords
             and token.part_of_speech.split(',')[0] == '名詞']
    return words

def extract_keywords(text, top_n=5):
    words = tokenize(text)
    if not words:
        return []
    vectorizer = TfidfVectorizer(tokenizer=lambda x: x, lowercase=False)
    tfidf = vectorizer.fit_transform([words])
    scores = zip(vectorizer.get_feature_names_out(), tfidf.toarray()[0])
    sorted_words = sorted(scores, key=lambda x: x[1], reverse=True)
    return [word for word, score in sorted_words[:top_n]]

def generate_hashtags(text, top_n=5):
    keywords = extract_keywords(text, top_n)
    return ['#' + word for word in keywords]

# Streamlit UI
st.title("ハッシュタグ自動生成アプリ")
st.write("SNS投稿に関連したハッシュタグを自動生成します。")

user_input = st.text_area("テキストを入力してください：", height=150)

if st.button("ハッシュタグを生成"):
    if user_input.strip() == "":
        st.warning("テキストを入力してください。")
    else:
        hashtags = generate_hashtags(user_input)
        full_output = user_input + "\n\n" + " ".join(hashtags)
        st.success("生成結果：")
        st.write(full_output)
