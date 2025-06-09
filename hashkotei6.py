import streamlit as st
from janome.tokenizer import Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
import re

# 形態素解析器
tokenizer = Tokenizer()

# 意味の薄い語を除外
stopwords = set([
    'こと', 'もの', 'これ', 'それ', 'ため', 'よう', 'さん', 'して', 'ます', 'です',
    'いる', 'ある', 'する', 'なる', '月', '時', '日', '〜', '・', '※', 'から',
    'ごろ', 'まで', '毎日', '午前', '午後', '内容', '曜日', '予約', '受付'
])

# 固定の優先ハッシュタグ候補
fixed_priority_keywords = [
    '中根学区子育てサロンゆりかご', 'ゆりかご', '汐路学区子育てサロン', 'すまいる',
    '子育て', '子育て支援', '瑞穂区', '名古屋ママ', '地域子育て支援拠点',
    '子育て応援拠点', 'さくらっこ♪', '子育てネットワークさくらっこ♪',
    '未就園児親子', '0歳ママ', '1歳ママ', '2歳ママ', '3歳ママ',
    '未就園児', '子育て広場', '子育て相談', 'マタニティ', '妊婦',
    'プレママ', '新米ママ', '令和4年ベビー', '令和5年ベビー', '令和6年ベビー', '令和7年ベビー',
    '絵本', '手遊び', '広場', 'ランチ', '親子', '一時預かり', 'イベント'
]
fixed_priority_keywords = set(fixed_priority_keywords)

# 毎回必ず付与する固定ハッシュタグ
always_include_hashtags = [
    '#子育て', '#子育て支援', '#瑞穂区', '#名古屋ママ',
    '#地域子育て支援拠点', '#子育て応援拠点', '#さくらっこ♪',
    '#子育てネットワークさくらっこ♪', '#瑞穂区ママ', '#名古屋ママ', "#プレママ", "#新米ママ", "#マタニティ", "#妊婦",
    "#令和4年ベビー", "#令和5年ベビー", "#令和6年ベビー"
]

# 単語のフィルタリング
def is_valid_word(word):
    return (
        word not in stopwords and
        not re.fullmatch(r'\d+', word) and
        not re.fullmatch(r'[a-zA-Z]+', word) and
        not re.fullmatch(r'[\W_]+', word) and
        len(word) > 1
    )

# 形態素解析
def tokenize(text):
    tokens = tokenizer.tokenize(text)
    words = []
    for token in tokens:
        base = token.base_form
        pos = token.part_of_speech.split(',')[0]
        if (pos in ['名詞', '固有名詞'] or pos == '名詞') and is_valid_word(base):
            words.append(base)
    return words

# キーワード抽出
def extract_keywords(text, top_n=15):
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
        if word not in extracted and len(extracted) < top_n:
            extracted.append(word)

    return extracted[:top_n]

# ハッシュタグ生成
def generate_hashtags(text, top_n=15):
    keywords = extract_keywords(text, top_n)
    auto_tags = ['#' + re.sub(r'\s+', '', kw) for kw in keywords]

    # 固定ハッシュタグのうち重複しないもののみ追加
    fixed_tags = [tag for tag in always_include_hashtags if tag not in auto_tags]

    return auto_tags, fixed_tags

# Streamlit UI
st.title("子育て投稿向けハッシュタグ自動生成アプリ")
#st.write("投稿本文に合わせた自動ハッシュタグに加え、固定の子育て支援タグも毎回追加されます。")

user_input = st.text_area("投稿内容を入力してください：", height=200)

if st.button("ハッシュタグを生成"):
    if user_input.strip() == "":
        st.warning("テキストを入力してください。")
    else:
        auto_tags, fixed_tags = generate_hashtags(user_input)
        
        hashtags_output = " ".join(auto_tags) + "\n\n" + " ".join(fixed_tags)
        full_output = user_input + "\n\n" + hashtags_output

        st.success("生成された投稿例：")
        st.write(full_output)
