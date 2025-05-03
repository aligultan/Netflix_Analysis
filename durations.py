import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns

df = pd.read_csv("data/netflix1.csv")
df.rename(columns={"listed_in": "category"}, inplace=True)

# 'date_added' sütunundan yıl bilgisini çıkar
df['date_added'] = pd.to_datetime(df['date_added'], errors='coerce')
df['year_added'] = df['date_added'].dt.year

df = df.dropna(subset=['duration', 'year_added'])

fig_dir = "graphics"
os.makedirs(fig_dir, exist_ok=True)

###  Filmler: Süre (dakika)
movie_df = df[df['type'] == 'Movie'].copy()
movie_df['minutes'] = movie_df['duration'].str.extract(r'(\d+)').astype(float)

# Yıla göre ortalama süre
movie_avg = movie_df.groupby('year_added')['minutes'].mean()
movie_fig = os.path.join(fig_dir, "film_sure_trendi_netflix.png")

if not os.path.exists(movie_fig):
    plt.figure(figsize=(10, 5))

    plt.style.use('dark_background')

    plt.plot(movie_avg.index, movie_avg.values, marker='o', color="#E50914", linewidth=2.5)

    plt.title("Yıllara Göre Film Süresi Ortalaması (dk)", color='white', fontsize=14)
    plt.xlabel("Yıl", color='white', fontsize=12)
    plt.ylabel("Ortalama Süre (dk)", color='white', fontsize=12)

    plt.grid(True, color='#333333', linestyle='--', alpha=0.7)

    plt.gca().spines['bottom'].set_color('#333333')
    plt.gca().spines['left'].set_color('#333333')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    plt.tick_params(axis='both', colors='white')

    plt.tight_layout()

    plt.savefig(movie_fig, facecolor='black', edgecolor='none')
    print(f"Netflix temalı film süresi grafiği kaydedildi: {movie_fig}")
    plt.show()
else:
    print(f"Film süresi grafiği zaten mevcut: {movie_fig}")

###  Diziler: Sezon sayısı
tv_df = df[df['type'] == 'TV Show'].copy()
tv_df['seasons'] = tv_df['duration'].str.extract(r'(\d+)').astype(float)

# Yıla göre ortalama sezon sayısı
tv_avg = tv_df.groupby('year_added')['seasons'].mean()
tv_fig = os.path.join(fig_dir, "tvshow_sezon_trendi_netflix.png")

if not os.path.exists(tv_fig):
    plt.figure(figsize=(10, 5))

    plt.style.use('dark_background')

    plt.plot(tv_avg.index, tv_avg.values, marker='o', color="#E50914", linewidth=2.5)

    plt.title("Yıllara Göre Dizi Sezon Ortalaması", color='white', fontsize=14)
    plt.xlabel("Yıl", color='white', fontsize=12)
    plt.ylabel("Ortalama Sezon", color='white', fontsize=12)

    plt.grid(True, color='#333333', linestyle='--', alpha=0.7)

    plt.gca().spines['bottom'].set_color('#333333')
    plt.gca().spines['left'].set_color('#333333')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    plt.tick_params(axis='both', colors='white')

    plt.tight_layout()

    plt.savefig(tv_fig, facecolor='black', edgecolor='none')
    print(f"Netflix temalı dizi sezon grafiği kaydedildi: {tv_fig}")
    plt.show()
else:
    print(f"Dizi sezon grafiği zaten mevcut: {tv_fig}")

# Kategorilere göre film süresi analizi
movie_df = df[df['type'] == 'Movie'].copy()

movie_df['minutes'] = movie_df['duration'].str.extract(r'(\d+)').astype(float)

# 'listed_in' sütununu ayır (birden fazla kategori olabilir)
movie_df = movie_df.dropna(subset=['category', 'minutes'])
movie_df['category'] = movie_df['category'].str.split(', ')
movie_exploded = movie_df.explode('category')

# Kategoriye göre ortalama süreyi hesapla
avg_duration_by_category = movie_exploded.groupby('category')['minutes'].mean().sort_values(ascending=False)

fig_dir = "graphics"
os.makedirs(fig_dir, exist_ok=True)
fig_path = os.path.join(fig_dir, "film_sure_kategoriye_gore_netflix.png")

if not os.path.exists(fig_path):
    plt.figure(figsize=(12, 8))

    plt.style.use('dark_background')
    netflix_colors = ["#E50914", "#B20710", "#831010", "#6E0D10", "#5C0B0B",
                      "#DB0000", "#A30000", "#CF0000", "#B9090B", "#960000",
                      "#FF0000", "#BF0000", "#FF1E1E", "#FF3939", "#FF5252",
                      "#D22F26", "#C11119", "#F85C4D", "#EA3C53", "#FF4D4D"]

    ax = sns.barplot(x=avg_duration_by_category.values,
                     y=avg_duration_by_category.index,
                     palette=netflix_colors)

    for i, v in enumerate(avg_duration_by_category.values):
        ax.text(v + 0.5, i, f"{v:.1f}", va='center', color='white', fontweight='bold')

    plt.title("Film Süresi (dk) - Kategorilere Göre Ortalama", fontsize=14, color='white')
    plt.xlabel("Ortalama Süre (dk)", fontsize=12, color='white')
    plt.ylabel("Kategori", fontsize=12, color='white')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#333333')
    ax.spines['left'].set_color('#333333')

    ax.grid(axis='x', linestyle='--', alpha=0.2, color='#555555')

    plt.tight_layout()
    plt.savefig(fig_path, facecolor='black', edgecolor='none')
    print(f"Netflix temalı kategori grafiği kaydedildi: {fig_path}")
    plt.show()
else:
    print(f"Kategori grafiği zaten mevcut: {fig_path}")


#Tv show lar için ortalama sezon sayısı
tv_df = df[df['type'] == 'TV Show'].copy()

tv_df['seasons'] = tv_df['duration'].str.extract(r'(\d+)').astype(float)
tv_df = tv_df.dropna(subset=['category', 'seasons'])

tv_df['category'] = tv_df['category'].str.split(', ')
tv_exploded = tv_df.explode('category')

avg_season_by_category = tv_exploded.groupby('category')['seasons'].mean().sort_values(ascending=False)

fig_dir = "graphics"
os.makedirs(fig_dir, exist_ok=True)
fig_path = os.path.join(fig_dir, "tvshow_sezon_kategoriye_gore_netflix.png")

if not os.path.exists(fig_path):
    plt.figure(figsize=(12, 8))
    plt.style.use('dark_background')

    # Netflix kırmızısı ve tonları
    netflix_colors = ["#E50914", "#B20710", "#831010", "#6E0D10", "#5C0B0B",
                      "#DB0000", "#A30000", "#CF0000", "#B9090B", "#960000",
                      "#FF0000", "#BF0000", "#FF1E1E", "#FF3939", "#FF5252",
                      "#D22F26", "#C11119", "#F85C4D", "#EA3C53", "#FF4D4D"]

    ax = sns.barplot(x=avg_season_by_category.values,
                     y=avg_season_by_category.index,
                     palette=netflix_colors)

    for i, v in enumerate(avg_season_by_category.values):
        ax.text(v + 0.05, i, f"{v:.2f}", va='center', color='white', fontweight='bold')

    plt.title("TV Show Sezon Sayısı - Kategorilere Göre Ortalama", fontsize=14, color='white')
    plt.xlabel("Ortalama Sezon Sayısı", fontsize=12, color='white')
    plt.ylabel("Kategori", fontsize=12, color='white')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#333333')
    ax.spines['left'].set_color('#333333')
    ax.grid(axis='x', linestyle='--', alpha=0.2, color='#555555')

    plt.tight_layout()
    plt.savefig(fig_path, facecolor='black', edgecolor='none')
    print(f"Netflix temalı TV Show kategori grafiği kaydedildi: {fig_path}")
    plt.show()
else:
    print(f"TV Show kategori grafiği zaten mevcut: {fig_path}")