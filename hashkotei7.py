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
fixed_priority_keywords = set([
    '中根学区子育てサロンゆりかご', 'ゆりかご', '汐路学区子育てサロン', 'すまいる',
    '子育て', '子育て支援', '瑞穂区', '名古屋ママ', '地域子育て支援拠点',
    '子育て応援拠点', 'さくらっこ♪', '子育てネットワークさくらっこ♪',
    '未就園児親子', '0歳ママ', '1歳ママ', '2歳ママ', '3歳ママ',
    '未就園児', '子育て広場', '子育て相談', 'マタニティ', '妊婦',
    'プレママ', '新米ママ', '令和4年ベビー', '令和5年ベビー', '令和6年ベビー', '令和7年ベビー',
    '絵本', '手遊び', '広場', 'ランチ', '親子', '一時預かり', 'イベント'
])

# 毎回付けたい固定タグ（投稿に関係なく常に表示）
always_include_hashtags = [
    '#子育て', '#子育て支援', '#瑞穂区', '#名古屋ママ',
    '#地域子育て支援拠点', '#子育て応援拠点',
    '#さくらっこ♪', '#子育てネットワークさくらっこ♪',
    '#瑞穂区ママ', '#名古屋ママ', '#子育て支援クラブ'
]

# 単語のフィルタリング
def is_valid_word(word):
    return (
        word not in stopwords and
        not re.fullmatch(r'\d+', word) and      # 数字だけの単語除外
        not re.fullmatch(r'[a-zA-Z]+', word) and # 英字のみ除外
        not re.fullmatch(r'[\W_]+', word)       # 記号のみ除外
    )

# 形態素解析
def tokenize(text):
    tokens = tokenizer.tokenize(text)
    words = []
    for token in tokens:
        base = token.base_form
        pos = token.part_of_speech.split(',')[0]
        if pos in ['名詞', '固有名詞'] and is_valid_word(base):
            words.append(base)
    return words

# 団体名を本文から抽出してハッシュタグ化
def extract_group_tag(text):
    match = re.search(r'団体名[:：]\s*([^\n\r]*)', text)
    if match:
        group_name = match.group(1).strip()
        if group_name:
            return f"#{group_name}"
    return None

# キーワード抽出
def extract_keywords(text, top_n=15):
    words = tokenize(text)
    if not words:
        return []
    vectorizer = TfidfVectorizer(tokenizer=lambda x: x, lowercase=False)
    tfidf = vectorizer.fit_transform([words])
    scores = zip(vectorizer.get_feature_names_out(), tfidf.toarray()[0])
    sorted_words = sorted(scores, key=lambda x: x[1], reverse=True)

    # 固定候補から本文に含まれる語を抽出
    fixed_in_text = [kw for kw in fixed_priority_keywords if kw in text]

    # TF-IDF上位語と合わせて最大top_n件まで抽出
    extracted = fixed_in_text[:]
    for word, score in sorted_words:
        if word not in extracted and len(extracted) < top_n:
            extracted.append(word)

    return extracted[:top_n]

# ハッシュタグ生成（自動＋固定）
def generate_hashtags(text, top_n=15):
    keywords = extract_keywords(text, top_n)
    auto_tags = ['#' + re.sub(r'\s+', '', kw) for kw in keywords if len(kw) > 1]

    # 固定タグを追加（重複排除）
    fixed_tags = [tag for tag in always_include_hashtags if tag not in auto_tags]

    # 団体名タグを追加（あれば）
    group_tag = extract_group_tag(text)
    if group_tag and group_tag not in auto_tags and group_tag not in fixed_tags:
        fixed_tags.append(group_tag)

    return auto_tags, fixed_tags

# Streamlit UI
st.title("子育て投稿向けハッシュタグ自動生成アプリ（最新版）")
st.write("子育て支援に関する投稿から適切なハッシュタグを抽出します。\n投稿内容に応じたタグに加え、毎回表示されるタグや団体名のタグも自動付与されます。")

user_input = st.text_area("投稿内容を入力してください：", height=250)

if st.button("ハッシュタグを生成"):
    if user_input.strip() == "":
        st.warning("テキストを入力してください。")
    else:
        auto_tags, fixed_tags = generate_hashtags(user_input, top_n=15)
        all_tags = " ".join(auto_tags) + "\n\n" + " ".join(fixed_tags)
        full_output = user_input + "\n\n" + all_tags
        st.success("生成結果：")
        st.write(full_output)

