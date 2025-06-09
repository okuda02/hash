import streamlit as st
from janome.tokenizer import Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
import re

tokenizer = Tokenizer()

# 除外する語
stopwords = set([
    'こと', 'もの', 'これ', 'それ', 'ため', 'よう', 'さん', 'して', 'ます', 'です',
    'いる', 'ある', 'する', 'なる', '月', '時', '日', '〜', '・', '※', 'から',
    'ごろ', 'まで', '毎日', '午前', '午後', '内容', '曜日', '予約', '受付'
])

# 固定の優先ハッシュタグ候補
fixed_priority_keywords = set([
    '中根学区子育てサロンゆりかご', 'ゆりかご', '汐路学区子育てサロン', 'すまいる',
    '子育て', '子育て支援', '瑞穂区', '名古屋ママ', '地域子育て支援拠点',
    '子育て応援拠点', 'さくらっこ♪', '子育てネットワークさくらっこ♪',
    '未就園児親子', '0歳ママ', '1歳ママ', '2歳ママ', '3歳ママ',
    '未就園児', '子育て広場', '子育て相談', 'マタニティ', '妊婦',
    'プレママ', '新米ママ', '令和4年ベビー', '令和5年ベビー', '令和6年ベビー', '令和7年ベビー',
    '絵本', '手遊び', '広場', 'ランチ', '親子', '一時預かり', 'イベント',
    'みずほこどもまつり2025', '瑞穂区子育てネットワークさくらっこ♪',
    '生涯学習センター', 'カプラ', 'スーパーアクロバット', '瑞穂区子育て'
])

# 単語が有効かどうか判定
def is_valid_word(word):
    return (
        word not in stopwords and
        not re.fullmatch(r'\d+', word) and
        not re.fullmatch(r'[a-zA-Z]+', word) and
        not re.fullmatch(r'[\W_]+', word) and
        len(word) > 2  # 1～2文字の単語は除外
    )

# テキストをトークン化
def tokenize(text):
    tokens = tokenizer.tokenize(text)
    words = []
    for token in tokens:
        base = token.base_form
        pos = token.part_of_speech.split(',')[0]
        if pos in ['名詞', '固有名詞'] and is_valid_word(base):
            words.append(base)
    return words

# キーワード抽出（TF-IDFと固定キーワードの融合）
def extract_keywords(text, top_n=25):
    words = tokenize(text)
    if not words:
        return []

    vectorizer = TfidfVectorizer(tokenizer=lambda x: x, lowercase=False)
    tfidf = vectorizer.fit_transform([words])
    scores = zip(vectorizer.get_feature_names_out(), tfidf.toarray()[0])
    sorted_words = sorted(scores, key=lambda x: x[1], reverse=True)

    fixed_in_text = [kw for kw in fixed_priority_keywords if kw in text]
    extracted = fixed_in_text[:]

    for word, score in sorted_words:
        if word not in extracted and is_valid_word(word) and len(extracted) < top_n:
            extracted.append(word)

    return extracted[:top_n]

# ハッシュタグ変換
def generate_hashtags(text, top_n=25):
    keywords = extract_keywords(text, top_n)
    hashtags = ['#' + re.sub(r'\s+', '', kw) for kw in keywords]
    return hashtags


# Streamlit UI
st.title("子育て投稿向けハッシュタグ自動生成アプリ（改良版）")
st.write("子育て団体の投稿に合わせてハッシュタグを多めに抽出・優先表示します。")

user_input = st.text_area("投稿内容を入力してください：", height=200)

if st.button("ハッシュタグを生成"):
    if user_input.strip() == "":
        st.warning("テキストを入力してください。")
    else:
        hashtags = generate_hashtags(user_input, top_n=25)
        full_output = user_input + "\n\n" + " ".join(hashtags)
        st.success("生成結果：")
        st.write(full_output)
