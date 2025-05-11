from wordcloud import WordCloud, STOPWORDS
import os
import pandas as pd

df = pd.read_csv("data/netflix1.csv")
df.rename(columns={"listed_in": "category"}, inplace=True)

movie_df = df[df['type'] == 'Movie'].copy()
movie_df = movie_df.dropna(subset=['title', 'category'])

movie_df['category'] = movie_df['category'].str.split(', ')
movie_exploded = movie_df.explode('category')

wordcloud_dir = "wordclouds"
os.makedirs(wordcloud_dir, exist_ok=True)
stopwords = set(STOPWORDS)
stopwords.update(["Movie", "Film", "Series", "Season", "Netflix", "the", "The"])

# Her kategori için wordcloud oluşturma
for category in movie_exploded['category'].dropna().unique():
    titles = movie_exploded[movie_exploded['category'] == category]['title']
    text = " ".join(title for title in titles if isinstance(title, str))

    if not text.strip():
        continue

    wordcloud = WordCloud(
        stopwords=stopwords,
        background_color='black',
        width=800,
        height=400,
        colormap='plasma'
    ).generate(text)

    filename = f"{category.lower().replace('&', 'and').replace(' ', '_')}_titles_wordcloud.png"
    filepath = os.path.join(wordcloud_dir, filename)

    wordcloud.to_file(filepath)
    print(f"{category} için kelime bulutu kaydedildi: {filepath}")
