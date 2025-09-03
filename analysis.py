import pandas as pd
from konlpy.tag import Okt
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
import os
from datetime import datetime
import random

# 데이터 불러오기 및 전처리
df = pd.read_csv('donspike_news_with_body_all.csv')
df['날짜'] = pd.to_datetime(df['날짜'])
df['주'] = df['날짜'].dt.strftime('%Y-%U')
df['월'] = df['날짜'].dt.strftime('%Y-%m')

# 감성사전 불러오기
dict_df = pd.read_csv('SentiWord_Dict.txt', sep='\t', names=['word', 'polarity'], encoding='utf-8')
pos_words = set(dict_df[dict_df['polarity'] == 1]['word'])
neg_words = set(dict_df[dict_df['polarity'] == -1]['word'])

# 형태소 분석기 (Okt, stem=True)
okt = Okt()
def extract_sentiment_words(text):
    if pd.isnull(text):
        return [], []
    words = [w for w, p in okt.pos(str(text), stem=True) if p in ['Noun', 'Adjective', 'Verb']]
    pos = [w for w in words if w in pos_words]
    neg = [w for w in words if w in neg_words]
    return pos, neg, len(words)

df[['긍정어', '부정어', '의미단어수']] = df['본문'].apply(lambda x: pd.Series(extract_sentiment_words(x)))
df['긍정단어수'] = df['긍정어'].apply(len)
df['부정단어수'] = df['부정어'].apply(len)
df['긍정비율'] = df.apply(lambda row: row['긍정단어수'] / row['의미단어수'] if row['의미단어수'] > 0 else 0, axis=1)
df['부정비율'] = df.apply(lambda row: row['부정단어수'] / row['의미단어수'] if row['의미단어수'] > 0 else 0, axis=1)

# 주별 기사 수 3개 미만 제외
week_counts = df.groupby('주').size().reset_index(name='기사수')
valid_weeks = week_counts[week_counts['기사수'] >= 3]['주']
df_valid = df[df['주'].isin(valid_weeks)]

# 주별 집계
weekly_sentiment = df_valid.groupby('주').agg({'긍정비율':'mean', '부정비율':'mean'}).reset_index()

# 한글 주차 레이블 변환 함수
def iso_year_week_to_korean_label(year_week_str):
    year, week = map(int, year_week_str.split('-'))
    try:
        first_day_of_week = datetime.fromisocalendar(year, week, 1)
    except ValueError:
        return ''
    month = first_day_of_week.month
    year_short = str(year)[2:]
    first_day_of_month = datetime(year, month, 1)
    first_week_of_month = first_day_of_month.isocalendar()[1]
    month_week = week - first_week_of_month + 1
    if month_week <= 0:
        month_week = 1
    week_names = ['첫째주', '둘째주', '셋째주', '넷째주', '다섯째주', '여섯째주']
    week_str = week_names[month_week - 1] if 1 <= month_week <= len(week_names) else f'{month_week}째주'
    return f'{year_short}년 {month}월 {week_str}'

weekly_sentiment['주_한글'] = weekly_sentiment['주'].apply(iso_year_week_to_korean_label)
if len(weekly_sentiment) > 15:
    prev_month = ''
    for i, row in weekly_sentiment.iterrows():
        label = row['주_한글']
        if '첫째주' in label:
            prev_month = label.split(' ')[1]  # '12월'
            weekly_sentiment.at[i, '주_한글'] = label
        else:
            weekly_sentiment.at[i, '주_한글'] = ''

# 막대그래프 시각화 및 저장
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
sns.set(style='whitegrid', font='Malgun Gothic', font_scale=1.2)

plt.figure(figsize=(16,8))
bar_width = 0.35
indices = range(len(weekly_sentiment))
plt.bar(indices, weekly_sentiment['긍정비율'], width=bar_width, label='긍정비율', color='#4E79A7')
plt.bar([i + bar_width for i in indices], weekly_sentiment['부정비율'], width=bar_width, label='부정비율', color='#F28E2B')
plt.xticks([i + bar_width/2 for i in indices], weekly_sentiment['주_한글'], rotation=45, ha='right')
plt.title('돈스파이크 주별 긍정/부정 단어 비율 변화', fontsize=18, weight='bold')
plt.xlabel('주차 (한글)', fontsize=14)
plt.ylabel('단어 비율', fontsize=14)
plt.legend(title='', fontsize=13, loc='upper right', frameon=True, shadow=True)
plt.tight_layout()
plt.savefig('weekly_sentiment_ratio_bar.png', dpi=200)
plt.close()

# 금지어 적용
ban_words = {'있다', '마약', '없다', '받다'}

# 월별 긍정/부정 단어 합산, 상위 30개 추출
font_path = "C:/Windows/Fonts/malgun.ttf"  # 실제 경로로 수정
os.makedirs('wordcloud_monthly', exist_ok=True)

for month in sorted(df['월'].unique()):
    pos_counter = Counter()
    neg_counter = Counter()
    for idx, row in df[df['월'] == month].iterrows():
        pos_counter.update([w for w in row['긍정어'] if w not in ban_words])
        neg_counter.update([w for w in row['부정어'] if w not in ban_words])
    # 합산
    total_counter = pos_counter + neg_counter
    # 상위 30개만
    top30 = dict(total_counter.most_common(30))
    # 색상 함수
    def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
        if word in pos_counter:
            # 초록-파랑 계열
            return random.choice(['#2ecc40', '#1f77b4', '#27ae60', '#17becf'])
        elif word in neg_counter:
            # 노랑-주황-빨강 계열
            return random.choice(['#ffb300', '#ff5733', '#ff0000', '#e67e22'])
        else:
            return "#888888"
    if top30:
        wc = WordCloud(font_path=font_path, width=900, height=450, background_color='white', max_words=30, color_func=color_func)
        wc.generate_from_frequencies(top30)
        plt.figure(figsize=(12,6))
        plt.title(f'돈스파이크 {month} 긍·부정 단어 워드클라우드')
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f'wordcloud_monthly/{month}_sentiment_wordcloud.png', dpi=150)
        plt.close()
