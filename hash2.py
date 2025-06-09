import streamlit as st
from janome.tokenizer import Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
import re

# Janomeで形態素解析
tokenizer = Tokenizer()

# ストップワード（意味の薄い語を除外）
stopwords = set([
    'こと', 'もの', 'これ', 'それ', 'ため', 'よう', 'さん', 'して', 'ます', 'です',
    'いる', 'ある', 'する', 'なる', '月', '時', '日', '〜', '・', '※', 'から',
    'ごろ', 'まで', '毎日', '午前', '午後', '内容', '曜日', '予約', '受付'
])

# 育児・子育て系キーワードを優先候補として登録
priority_keywords = set([
    '子育て', '育児', 'まぁぶる', 'プレママ', '手遊び', '絵本', '広場', '一時預かり',
    'ランチ', '親子', '0歳', '1歳', '2歳', '3歳', '赤ちゃん', 'ママ', 'イベント'
])

# 単語のフィルタリング
def is_valid_word(word):
    return (
        word not in stopwords and
        not re.fullmatch(r'\d+', word) and       # 数字だけの単語を除外
        not re.fullmatch(r'[a-zA-Z]+', word) and # 英字のみも除外（任意）
        not re.fullmatch(r'[\W_]+', word)        # 記号のみも除外
    )

# トークン化（名詞、動詞、形容詞）
def tokenize(text):
    tokens = tokenizer.tokenize(text)
    words = []
    for token in tokens:
        base = token.base_form
        pos = token.part_of_speech.split(',')[0]
        if pos in ['名詞', '動詞', '形容詞'] and is_valid_word(base):
            words.append(base)
    return words

# キーワード抽出
def extract_keywords(text, top_n=5):
    words = tokenize(text)
    if not words:
        return []

    # TF-IDFを計算
    vectorizer = TfidfVectorizer(tokenizer=lambda x: x, lowercase=False)
    tfidf = vectorizer.fit_transform([words])
    scores = zip(vectorizer.get_feature_names_out(), tfidf.toarray()[0])

    # スコア順に並び替え
    sorted_words = sorted(scores, key=lambda x: x[1], reverse=True)

    # 優先キーワードを先に並べる
    keywords = [word for word, _ in sorted_words if word in priority_keywords]

    # 残りを補完（重複除去）
    if len(keywords) < top_n:
        extras = [word for word, _ in sorted_words if word not in keywords]
        keywords.extend(extras)

    return keywords[:top_n]

# ハッシュタグ生成
def generate_hashtags(text, top_n=5):
    keywords = extract_keywords(text, top_n)
    return ['#' + word for word in keywords]

# Streamlit UI
st.title("子育て投稿向けハッシュタグ自動生成アプリ")
st.write("子育て施設・育児支援系のSNS投稿に最適なハッシュタグを自動生成します。")

user_input = st.text_area("投稿内容を入力してください：", height=150)

if st.button("ハッシュタグを生成"):
    if user_input.strip() == "":
        st.warning("テキストを入力してください。")
    else:
        hashtags = generate_hashtags(user_input, top_n=6)
        full_output = user_input + "\n\n" + " ".join(hashtags)
        st.success("生成結果：")
        st.write(full_output)
