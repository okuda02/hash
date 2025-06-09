import streamlit as st
from janome.tokenizer import Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
import re

tokenizer = Tokenizer()

# 意味の薄い語を除外
stopwords = set([
    'こと', 'もの', 'これ', 'それ', 'ため', 'よう', 'さん', 'して', 'ます', 'です',
    'いる', 'ある', 'する', 'なる', '月', '時', '日', '〜', '・', '※', 'から',
    'ごろ', 'まで', '毎日', '午前', '午後', '内容', '曜日', '予約', '受付'
])

# 固定の優先ハッシュタグ候補（投稿例に基づく）
fixed_priority_keywords = [
    '中根学区子育てサロンゆりかご', 'ゆりかご', '汐路学区子育てサロン', 'すまいる',
    '子育て', '子育て支援', '瑞穂区', '名古屋ママ', '地域子育て支援拠点',
    '子育て応援拠点', 'さくらっこ♪', '子育てネットワークさくらっこ♪',
    '未就園児親子', '0歳ママ', '1歳ママ', '2歳ママ', '3歳ママ',
    '未就園児', '子育て広場', '子育て相談', 'マタニティ', '妊婦',
    'プレママ', '新米ママ', '令和4年ベビー', '令和5年ベビー', '令和6年ベビー', '令和7年ベビー',
    '絵本', '手遊び', '広場', 'ランチ', '親子', '一時預かり', 'イベント'
]

fixed_priority_keywords = set(fixed_priority_keywords)  # 重複排除＆検索高速化

# 単語のフィルタリング
def is_valid_word(word):
    return (
        word not in stopwords and
        not re.fullmatch(r'\d+', word) and      # 数字だけの単語除外
        not re.fullmatch(r'[a-zA-Z]+', word) and # 英字のみ除外（任意）
        not re.fullmatch(r'[\W_]+', word)       # 記号のみ除外
    )

# 形態素解析で名詞・固有名詞・数字を幅広く抽出（固有名詞優先）
def tokenize(text):
    tokens = tokenizer.tokenize(text)
    words = []
    for token in tokens:
        base = token.base_form
        pos = token.part_of_speech.split(',')[0]
        # 固有名詞も入れる
        if (pos in ['名詞', '固有名詞'] or pos == '名詞') and is_valid_word(base):
            words.append(base)
        # 動詞・形容詞も一部入れてみる（コメントアウト解除して試すのもあり）
        # if pos in ['動詞', '形容詞'] and is_valid_word(base):
        #     words.append(base)
    return words

def extract_keywords(text, top_n=6):
    words = tokenize(text)
    if not words:
        return []

    # TF-IDF計算
    vectorizer = TfidfVectorizer(tokenizer=lambda x: x, lowercase=False)
    tfidf = vectorizer.fit_transform([words])
    scores = zip(vectorizer.get_feature_names_out(), tfidf.toarray()[0])
    sorted_words = sorted(scores, key=lambda x: x[1], reverse=True)

    # 固定辞書内の単語で文中にあるものは必ず入れる
    fixed_in_text = [kw for kw in fixed_priority_keywords if kw in text]

    # TF-IDF上位語も取得（重複しないように）
    extracted = fixed_in_text[:]
    for word, score in sorted_words:
        if word not in extracted and len(extracted) < top_n:
            extracted.append(word)

    return extracted[:top_n]

def generate_hashtags(text, top_n=6):
    keywords = extract_keywords(text, top_n)
    # ハッシュタグにする際、空白や特殊文字を除去（任意カスタム可）
    hashtags = []
    for kw in keywords:
        clean_kw = re.sub(r'\s+', '', kw)
        hashtags.append('#' + clean_kw)
    return hashtags


# Streamlit UI
st.title("子育て投稿向けハッシュタグ自動生成アプリ（改良版）")
st.write("子育て団体の投稿に合わせてハッシュタグを多めに抽出・優先表示します。")

user_input = st.text_area("投稿内容を入力してください：", height=200)

if st.button("ハッシュタグを生成"):
    if user_input.strip() == "":
        st.warning("テキストを入力してください。")
    else:
        hashtags = generate_hashtags(user_input, top_n=10)  # 10個抽出してみる
        full_output = user_input + "\n\n" + " ".join(hashtags)
        st.success("生成結果：")
        st.write(full_output)
