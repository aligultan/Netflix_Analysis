import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Netflix teması ayarları
plt.style.use('dark_background')
sns.set_style("dark", {"axes.facecolor": "#000000"})
plt.rcParams['axes.edgecolor'] = 'white'
plt.rcParams['axes.labelcolor'] = 'white'
plt.rcParams['xtick.color'] = 'white'
plt.rcParams['ytick.color'] = 'white'
plt.rcParams['text.color'] = 'white'
plt.rcParams['figure.facecolor'] = '#000000'
plt.rcParams['axes.facecolor'] = '#000000'
plt.rcParams['savefig.facecolor'] = '#000000'

red = '#8E1616'
gold = '#E8C999'

df = pd.read_csv("data/netflix1.csv")

df.rename(columns={"listed_in": "category"}, inplace=True)

fig_dir = "graphics"
os.makedirs(fig_dir, exist_ok=True)

# 1. Pie Chart: Film ve TV Show dağılımı
fig_path = os.path.join(fig_dir, "type_distribution_pie.png")
if not os.path.exists(fig_path):
    type_counts = df['type'].value_counts()
    plt.figure(figsize=(6, 6))
    plt.pie(type_counts, labels=type_counts.index, autopct='%1.1f%%',
            startangle=140, colors=[red, gold], textprops={'color': 'white'})
    plt.title('(%) Film ve TV Show Dağılımı', pad=30)
    plt.axis('equal')
    plt.savefig(fig_path)
    plt.show()

# 2. Ülke listesi
country_series = df['country'].dropna().str.split(', ')
all_countries = country_series.explode()
country_list = sorted(all_countries.unique())
with open("countries.txt", 'w', encoding='utf-8') as f:
    f.write(f"Toplam {len(country_list)} ülke bulundu.\n\n")
    for country in country_list:
        f.write(f"- {country}\n")

# 3. Ülke bazlı içerik dağılımı
country_expanded = df[['type', 'country']].dropna()
country_expanded['country'] = country_expanded['country'].str.split(', ')
country_expanded = country_expanded.explode('country')
country_expanded['country'] = country_expanded['country'].str.strip()

# "Not Given" filtrele
country_expanded = country_expanded[~country_expanded['country'].str.contains("Not Given", case=False, na=False)]

country_type_counts = country_expanded.groupby(['country', 'type']).size().unstack(fill_value=0)
top_10_countries = country_type_counts.sum(axis=1).sort_values(ascending=False).head(10)
top_data = country_type_counts.loc[top_10_countries.index]

# Toplam içerik sayısı (stacked bar)
fig_path = os.path.join(fig_dir, "top_10_countries_tv_film_distribution.png")
if not os.path.exists(fig_path):
    top_data.plot(kind='bar', stacked=True, figsize=(10, 6), color=[red, gold])
    plt.title("En Fazla İçeriğe Sahip 10 Ülke (TV Show & Film)")
    plt.xlabel("Ülke")
    plt.ylabel("İçerik Sayısı")
    plt.xticks(rotation=45, ha='right')
    plt.legend(title="Tür")
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.show()

top_10_movies = country_type_counts['Movie'].sort_values(ascending=False).head(10)
fig_path_movie = os.path.join(fig_dir, "top_10_movies_by_country.png")
if not os.path.exists(fig_path_movie):
    top_10_movies.plot(kind='bar', color=red, figsize=(10, 6))
    plt.title("En Fazla Movie İçeriğine Sahip 10 Ülke")
    plt.xlabel("Ülke")
    plt.ylabel("Movie Sayısı")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(fig_path_movie)
    plt.show()

#  Sadece TV Show
top_10_shows = country_type_counts['TV Show'].sort_values(ascending=False).head(10)
fig_path_show = os.path.join(fig_dir, "top_10_tv_shows_by_country.png")
if not os.path.exists(fig_path_show):
    top_10_shows.plot(kind='bar', color=red, figsize=(10, 6))
    plt.title("En Fazla TV Show İçeriğine Sahip 10 Ülke")
    plt.xlabel("Ülke")
    plt.ylabel("TV Show Sayısı")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(fig_path_show)
    plt.show()

# 4. Kategori bazlı analiz
category_expanded = df['category'].dropna().str.split(', ').explode()

# "International Movies" ve "International TV Shows" kategorilerini hariç tut
excluded_categories = ["International Movies", "International TV Shows"]
category_expanded = category_expanded[~category_expanded.isin(excluded_categories)]

# En çok geçen 10 kategori
top_10_categories = category_expanded.value_counts().head(10)

fig_path_categories = os.path.join(fig_dir, "top_10_categories.png")
if not os.path.exists(fig_path_categories):
    top_10_categories.plot(kind='bar', color=red, figsize=(10, 6))
    plt.title("En Fazla Görülen 10 Kategori (Uluslararası Kategoriler Hariç)")
    plt.xlabel("Kategori")
    plt.ylabel("Kategori Sayısı")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(fig_path_categories)
    plt.show()

# 5. Top 10 ülkenin en çok içerik sağladığı kategori (filtreli)
df_country_cat = df[['category', 'country']].dropna()
df_country_cat['country'] = df_country_cat['country'].str.split(', ')
df_country_cat['category'] = df_country_cat['category'].str.split(', ')
df_exploded = df_country_cat.explode('country').explode('category')
df_exploded['country'] = df_exploded['country'].str.strip()
df_exploded['category'] = df_exploded['category'].str.strip()

# Not Given filtrele
df_exploded = df_exploded[~df_exploded['country'].str.contains("Not Given", case=False, na=False)]

excluded_categories = ['international movies', 'international tv shows', 'not given', 'british tv shows']
df_exploded['category_lower'] = df_exploded['category'].str.lower()
filtered_df = df_exploded[~df_exploded['category_lower'].isin(excluded_categories)]
filtered_df = filtered_df[filtered_df['country'].isin(top_10_countries.index)]

top_category_per_country = (
    filtered_df.groupby(['country', 'category'])
    .size()
    .reset_index(name='count')
    .sort_values(['country', 'count'], ascending=[True, False])
    .drop_duplicates('country')
)

top_category_per_country['country'] = pd.Categorical(
    top_category_per_country['country'],
    categories=top_10_countries.index,
    ordered=True
)
top_category_per_country = top_category_per_country.sort_values('country')

fig_path = os.path.join(fig_dir, "top_category_per_top_10_countries.png")
if not os.path.exists(fig_path):
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=top_category_per_country,
        x='count',
        y='country',
        hue='category',
        dodge=False,
        palette='Set3'
    )
    plt.title("En Fazla İçeriğe Sahip 10 Ülkede En Popüler Kategori (Filtreli)")
    plt.xlabel("Kategori Sayısı")
    plt.ylabel("Ülke")
    plt.legend(title="Kategori", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.show()