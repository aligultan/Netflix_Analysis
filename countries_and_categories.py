import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

df = pd.read_csv("data/netflix1.csv")

print(df.head())
print(df.tail())
print(df.info())
print(df.isnull().sum())

print("Çift kayıt sayısı:", df.duplicated().sum())

df.rename(columns={"listed_in": "category"}, inplace=True)

print(df.head())

fig_dir = "graphics"
os.makedirs(fig_dir, exist_ok=True)

fig_path = os.path.join(fig_dir, "type_distribution_pie.png")

if not os.path.exists(fig_path):
    type_counts = df['type'].value_counts()

    plt.figure(figsize=(6,6))
    plt.pie(type_counts,
            labels=type_counts.index,
            autopct='%1.1f%%',
            startangle=140,
            colors=['#8E1616', '#E8C999'])
    plt.title('(%)Film ve TV Show Dağılımı', pad=30)
    plt.axis('equal')

    plt.savefig(fig_path)
    print(f"Grafik kaydedildi: {fig_path}")
    plt.show()
else:
    print(f"Grafik zaten mevcut: {fig_path}")


# 'country' sütunundaki boş değerleri çıkar, çoklu ülke varsa ayır
country_series = df['country'].dropna().str.split(', ')
all_countries = country_series.explode()

country_list = sorted(all_countries.unique())

output_path = "countries.txt"

with open(output_path, 'w', encoding='utf-8') as f:
    f.write(f"Toplam {len(country_list)} ülke bulundu.\n\n")
    for country in country_list:
        f.write(f"- {country}\n")

print(f"Ülke listesi dosyaya yazıldı: {output_path}")


# Çoklu ülke isimlerini ayır, her satırı tek ülkeye indir
country_expanded = df[['type', 'country']].dropna()
country_expanded['country'] = country_expanded['country'].str.split(', ')
country_expanded = country_expanded.explode('country')

# Her ülke için TV Show ve Film sayısını grupla
country_type_counts = country_expanded.groupby(['country', 'type']).size().unstack(fill_value=0)

# En çok içeriğe sahip ilk 10 ülkeyi al (film + show toplamına göre)
top_10_countries = country_type_counts.sum(axis=1).sort_values(ascending=False).head(10)
top_data = country_type_counts.loc[top_10_countries.index]

fig_dir = "graphics"
os.makedirs(fig_dir, exist_ok=True)
fig_path = os.path.join(fig_dir, "top_10_countries_tv_film_distribution.png")

if not os.path.exists(fig_path):
    top_data.plot(kind='bar', stacked=True, figsize=(10,6), color=['#8E1616', '#E8C999'])

    plt.title("En Fazla İçeriğe Sahip 10 Ülke (TV Show & Film)")
    plt.xlabel("Ülke")
    plt.ylabel("İçerik Sayısı")
    plt.xticks(rotation=45, ha='right')
    plt.legend(title="Tür")
    plt.tight_layout()

    plt.savefig(fig_path)
    print(f"Grafik kaydedildi: {fig_path}")
    plt.show()
else:
    print(f"Grafik zaten mevcut: {fig_path}")


# Movie ve TV Show için ayrı ayrı en çok içeriğe sahip ilk 10 ülkeyi al
top_10_movies = country_type_counts['Movie'].sort_values(ascending=False).head(10)
top_10_shows = country_type_counts['TV Show'].sort_values(ascending=False).head(10)

# Movie için grafik oluşturma
fig_dir = "graphics"
os.makedirs(fig_dir, exist_ok=True)
fig_path_movie = os.path.join(fig_dir, "top_10_movies_by_country.png")

if not os.path.exists(fig_path_movie):
    top_10_movies.plot(kind='bar', color='#8E1616', figsize=(10,6))

    plt.title("En Fazla Movie İçeriğine Sahip 10 Ülke")
    plt.xlabel("Ülke")
    plt.ylabel("Movie Sayısı")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    plt.savefig(fig_path_movie)
    print(f"Movie grafiği kaydedildi: {fig_path_movie}")
    plt.show()
else:
    print(f"Movie grafiği zaten mevcut: {fig_path_movie}")



# TV Show için grafik oluşturma
fig_path_show = os.path.join(fig_dir, "top_10_tv_shows_by_country.png")

if not os.path.exists(fig_path_show):
    top_10_shows.plot(kind='bar', color='#E8C999', figsize=(10,6))

    plt.title("En Fazla TV Show İçeriğine Sahip 10 Ülke")
    plt.xlabel("Ülke")
    plt.ylabel("TV Show Sayısı")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    plt.savefig(fig_path_show)
    print(f"TV Show grafiği kaydedildi: {fig_path_show}")
    plt.show()
else:
    print(f"TV Show grafiği zaten mevcut: {fig_path_show}")


# Kategorileri ayırma ve her bir kategori için sayım yapma
category_expanded = df['category'].dropna().str.split(', ').explode()

# Kategorilerin sayısını al ve en fazla görülen ilk 10 kategoriyi seç
top_10_categories = category_expanded.value_counts().head(10)

fig_dir = "graphics"
os.makedirs(fig_dir, exist_ok=True)
fig_path_categories = os.path.join(fig_dir, "top_10_categories.png")

if not os.path.exists(fig_path_categories):
    top_10_categories.plot(kind='bar', color='#8E1616', figsize=(10,6))

    plt.title("En Fazla Görülen 10 Kategori")
    plt.xlabel("Kategori")
    plt.ylabel("Kategori Sayısı")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    plt.savefig(fig_path_categories)
    print(f"Kategori grafiği kaydedildi: {fig_path_categories}")
    plt.show()
else:
    print(f"Kategori grafiği zaten mevcut: {fig_path_categories}")

# Daha önce hesaplanmış: top_10_countries (Series), top_10_countries.index (ülkeler)
# category ve country sütunlarını explode ederek genişlet
df_country_cat = df[['category', 'country']].dropna()
df_country_cat['country'] = df_country_cat['country'].str.split(', ')
df_country_cat['category'] = df_country_cat['category'].str.split(', ')

df_exploded = df_country_cat.explode('country').explode('category')
df_exploded['country'] = df_exploded['country'].str.strip()
df_exploded['category'] = df_exploded['category'].str.strip()

# Hariç tutulacak kategoriler (küçük harfe çevirerek filtreleme yapacağız)
excluded_categories = ['international movies', 'international tv shows', 'not given', 'british tv shows']

df_exploded['category_lower'] = df_exploded['category'].str.lower()
filtered_df = df_exploded[~df_exploded['category_lower'].isin(excluded_categories)]

# Sadece top 10 ülkeyi alalım
filtered_df = filtered_df[filtered_df['country'].isin(top_10_countries.index)]

# Ülke bazında en çok geçen kategoriyi bul
top_category_per_country = (
    filtered_df.groupby(['country', 'category'])
    .size()
    .reset_index(name='count')
    .sort_values(['country', 'count'], ascending=[True, False])
    .drop_duplicates('country')
)

# Ülkelerin sırasını korumak için kategorik hale getir
top_category_per_country['country'] = pd.Categorical(
    top_category_per_country['country'],
    categories=top_10_countries.index,
    ordered=True
)
top_category_per_country = top_category_per_country.sort_values('country')


custom_palette = [
    '#8E1616', '#E8C999', '#B27200', '#005F73', '#0A9396',
    '#94D2BD', '#6A994E', '#A5A58D', '#DDA15E', '#BC4749'
]

fig_path = os.path.join("graphics", "top_category_per_top_10_countries.png")

if not os.path.exists(fig_path):
    plt.figure(figsize=(10,6))
    sns.barplot(
        data=top_category_per_country,
        x='count',
        y='country',
        hue='category',
        dodge=False,
        palette=custom_palette
    )

    plt.title("En Fazla İçeriğe Sahip 10 Ülkede En Popüler Kategori (Filtreli)")
    plt.xlabel("Kategori Sayısı")
    plt.ylabel("Ülke")
    plt.legend(title="Kategori", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(fig_path)
    print(f"Grafik kaydedildi: {fig_path}")
    plt.show()
else:
    print(f"Grafik zaten mevcut: {fig_path}")


