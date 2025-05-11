import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from collections import Counter
from wordcloud import WordCloud

def main():

    fig_dir = "graphics"
    os.makedirs(fig_dir, exist_ok=True)

    print("Netflix direktör analizi başlatılıyor...")

    df = pd.read_csv("data/netflix1.csv")

    # Direktör sütununu temizleyip işleyelim
    # NaN değerlerini kaldıralım
    director_df = df.dropna(subset=['director'])

    # Bazı filmlerde/dizilerde birden fazla direktör olabilir, onları ayıralım
    director_df['director'] = director_df['director'].str.split(', ')
    director_exploded = director_df.explode('director')

    print(f"Toplam {len(director_exploded['director'].unique())} farklı direktör bulundu.")

    # En çok içeriğe sahip direktörleri bulalım
    top_directors = director_exploded['director'].value_counts().head(15)

    # En popüler direktörler grafiği
    directors_fig_path = os.path.join(fig_dir, "netflix_top_directors.png")

    if not os.path.exists(directors_fig_path):
        plt.figure(figsize=(12, 10))
        plt.style.use('dark_background')

        # Netflix kırmızısı ve tonları
        colors = sns.color_palette("Reds_r", n_colors=len(top_directors))

        ax = sns.barplot(y=top_directors.index, x=top_directors.values, palette=colors)

        # Bar değerlerini göster
        for i, v in enumerate(top_directors.values):
            ax.text(v + 0.1, i, str(v), va='center', color='white', fontweight='bold')

        plt.title("Netflix'te En Çok İçeriğe Sahip Direktörler", fontsize=16, color='white')
        plt.xlabel("İçerik Sayısı", fontsize=14, color='white')
        plt.ylabel("Direktör", fontsize=14, color='white')

        # Grafik stilini ayarla
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('#333333')
        ax.spines['left'].set_color('#333333')
        ax.grid(axis='x', linestyle='--', alpha=0.3, color='#555555')

        plt.tight_layout()
        plt.savefig(directors_fig_path, facecolor='black', edgecolor='none')
        print(f"En popüler direktörler grafiği kaydedildi: {directors_fig_path}")
        plt.close()
    else:
        print(f"En popüler direktörler grafiği zaten mevcut: {directors_fig_path}")

    # Direktörlerin hangi türde içerik ürettiğini analiz edelim
    print("Direktörlerin tür (film/dizi) tercihleri analiz ediliyor...")

    # En popüler 10 direktörü seçelim ve tür dağılımına bakalım
    top10_directors = top_directors.head(10).index
    director_type_df = director_exploded[director_exploded['director'].isin(top10_directors)]

    # Direktör-Tür matrisi
    director_type_matrix = pd.crosstab(director_type_df['director'], director_type_df['type'])

    # Toplam içerik sayısını hesaplayalım ve sıralayalım
    director_type_matrix['Total'] = director_type_matrix.sum(axis=1)
    director_type_matrix = director_type_matrix.sort_values('Total', ascending=False)
    director_type_matrix = director_type_matrix.drop('Total', axis=1)

    director_type_path = os.path.join(fig_dir, "netflix_director_content_type.png")

    if not os.path.exists(director_type_path):
        plt.figure(figsize=(12, 8))
        plt.style.use('dark_background')

        colors = ['#E50914', '#831010']

        ax = director_type_matrix.plot(kind='bar', color=colors, width=0.8)

        plt.title("En Popüler Direktörlerin Film ve Dizi Dağılımı", fontsize=16, color='white')
        plt.xlabel("Direktör", fontsize=14, color='white')
        plt.ylabel("İçerik Sayısı", fontsize=14, color='white')
        plt.legend(facecolor='#222222', edgecolor='#444444', labelcolor='white')

        plt.grid(axis='y', linestyle='--', alpha=0.3, color='#555555')
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['bottom'].set_color('#333333')
        plt.gca().spines['left'].set_color('#333333')
        plt.tick_params(axis='both', colors='white')

        plt.tight_layout()
        plt.savefig(director_type_path, facecolor='black', edgecolor='none')
        print(f"Direktör-tür dağılımı grafiği kaydedildi: {director_type_path}")
        plt.close()
    else:
        print(f"Direktör-tür dağılımı grafiği zaten mevcut: {director_type_path}")

    # Direktörlerin tercih ettiği kategorileri analiz edelim
    print("Direktörlerin kategori tercihleri analiz ediliyor...")

    # İçerik kategorilerini ayıklama
    df['listed_in'] = df['listed_in'].str.split(', ')

    # En popüler 5 direktörü seçelim
    top5_directors = top_directors.head(5).index

    # Her bir direktör için en çok çalıştığı kategorileri bul
    director_categories = {}

    for director in top5_directors:
        # Bu direktörün filmlerini/dizilerini seç
        director_titles = director_exploded[director_exploded['director'] == director]['show_id'].unique()

        # Bu içeriklerin kategorilerini toplayalım
        all_categories = []
        for title_id in director_titles:
            categories = df[df['show_id'] == title_id]['listed_in'].values
            for cats in categories:
                if isinstance(cats, list):
                    all_categories.extend(cats)

        # En çok kullanılan 5 kategoriyi bulalım
        category_counts = Counter(all_categories).most_common(5)
        director_categories[director] = dict(category_counts)

    # Direktör-kategori grafiği
    director_category_path = os.path.join(fig_dir, "netflix_director_categories.png")

    if not os.path.exists(director_category_path):
        fig, axs = plt.subplots(len(top5_directors), 1, figsize=(12, 15))
        plt.style.use('dark_background')

        for i, director in enumerate(top5_directors):
            categories = list(director_categories[director].keys())
            counts = list(director_categories[director].values())

            # Netflix kırmızısı ve tonları
            colors = sns.color_palette("Reds_r", n_colors=len(categories))

            axs[i].barh(categories, counts, color=colors)
            axs[i].set_title(f"{director}", color='white')
            axs[i].set_xlabel("İçerik Sayısı", color='white')
            axs[i].tick_params(axis='both', colors='white')
            axs[i].spines['top'].set_visible(False)
            axs[i].spines['right'].set_visible(False)
            axs[i].spines['bottom'].set_color('#333333')
            axs[i].spines['left'].set_color('#333333')
            axs[i].grid(axis='x', linestyle='--', alpha=0.3, color='#555555')

            # Etiketleri ekle
            for j, v in enumerate(counts):
                axs[i].text(v + 0.1, j, str(v), va='center', color='white')

        plt.suptitle("En Popüler 5 Direktörün En Çok Çalıştığı Kategoriler",
                     fontsize=16, color='white', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.savefig(director_category_path, facecolor='black', edgecolor='none')
        print(f"Direktör-kategori ilişkisi grafiği kaydedildi: {director_category_path}")
        plt.close()
    else:
        print(f"Direktör-kategori ilişkisi grafiği zaten mevcut: {director_category_path}")

    print("Direktörlerin ülkelere göre dağılımı analiz ediliyor...")

    country_director_df = df.dropna(subset=['country', 'director'])
    country_director_df['country'] = country_director_df['country'].str.split(', ')
    country_director_df['director'] = country_director_df['director'].str.split(', ')

    # Hem ülke hem direktör sütunlarını ayıralım
    country_director_exploded = country_director_df.explode('country').explode('director')

    # En çok içerik üreten 5 ülkeyi seçelim
    top_countries = country_director_exploded['country'].value_counts().head(5).index

    # Her ülke için en popüler 5 direktörü bulalım
    country_top_directors = {}

    for country in top_countries:
        country_directors = country_director_exploded[country_director_exploded['country'] == country]
        top5_country_dirs = country_directors['director'].value_counts().head(5)
        country_top_directors[country] = top5_country_dirs

    # Ülke-direktör grafiği
    country_director_path = os.path.join(fig_dir, "netflix_country_top_directors.png")

    if not os.path.exists(country_director_path):
        fig, axs = plt.subplots(len(top_countries), 1, figsize=(12, 15))
        plt.style.use('dark_background')

        for i, country in enumerate(top_countries):
            directors = country_top_directors[country].index
            counts = country_top_directors[country].values

            colors = sns.color_palette("Reds_r", n_colors=len(directors))

            axs[i].barh(directors, counts, color=colors)
            axs[i].set_title(f"{country}", color='white')
            axs[i].set_xlabel("İçerik Sayısı", color='white')
            axs[i].tick_params(axis='both', colors='white')
            axs[i].spines['top'].set_visible(False)
            axs[i].spines['right'].set_visible(False)
            axs[i].spines['bottom'].set_color('#333333')
            axs[i].spines['left'].set_color('#333333')
            axs[i].grid(axis='x', linestyle='--', alpha=0.3, color='#555555')

            # Etiketleri ekle
            for j, v in enumerate(counts):
                axs[i].text(v + 0.1, j, str(v), va='center', color='white')

        plt.suptitle("En Çok İçerik Üreten 5 Ülkenin En Popüler Direktörleri",
                     fontsize=16, color='white', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.savefig(country_director_path, facecolor='black', edgecolor='none')
        print(f"Ülke-direktör ilişkisi grafiği kaydedildi: {country_director_path}")
        plt.close()
    else:
        print(f"Ülke-direktör ilişkisi grafiği zaten mevcut: {country_director_path}")

    print("Direktörlerin rating tercihleri analiz ediliyor...")

    # Rating sınıflandırma fonksiyonunu tanımlayalım
    def classify_rating(rating):
        if pd.isna(rating):
            return "Belirtilmemiş"
        elif rating in ['G', 'TV-Y', 'TV-G']:
            return "Genel İzleyici"
        elif rating in ['PG', 'TV-Y7', 'TV-Y7-FV', 'TV-PG']:
            return "Ebeveyn Rehberliği"
        elif rating in ['PG-13', 'TV-14']:
            return "13+ Yaş"
        elif rating in ['R', 'TV-MA', 'NC-17']:
            return "Yetişkin"
        else:
            return "Diğer"

    # Rating gruplarını oluştur
    df['rating_group'] = df['rating'].apply(classify_rating)

    # En popüler 5 direktörün rating dağılımını analiz edelim
    director_rating_df = pd.merge(director_exploded[director_exploded['director'].isin(top5_directors)],
                                  df[['show_id', 'rating_group']],
                                  on='show_id', how='left')

    director_rating_matrix = pd.crosstab(director_rating_df['director'], director_rating_df['rating_group'])

    # Toplam içerik sayısına göre sıralayalım
    director_rating_matrix['Total'] = director_rating_matrix.sum(axis=1)
    director_rating_matrix = director_rating_matrix.sort_values('Total', ascending=False)
    director_rating_matrix = director_rating_matrix.drop('Total', axis=1)

    # Rating kategorilerini belirli bir sıra ile göstermek istiyorsak
    rating_order = ["Genel İzleyici", "Ebeveyn Rehberliği", "13+ Yaş", "Yetişkin", "Belirtilmemiş", "Diğer"]
    director_rating_matrix = director_rating_matrix.reindex(
        columns=[col for col in rating_order if col in director_rating_matrix.columns])

    director_rating_path = os.path.join(fig_dir, "netflix_director_rating_heatmap.png")

    if not os.path.exists(director_rating_path):
        plt.figure(figsize=(12, 8))
        plt.style.use('dark_background')

        # Kırmızı tonlarında bir renk haritası
        cmap = sns.color_palette("Reds", as_cmap=True)

        # Heatmap oluştur
        ax = sns.heatmap(director_rating_matrix, annot=True, fmt='d', cmap=cmap,
                         linewidths=0.5, cbar_kws={'label': 'İçerik Sayısı'})

        plt.title("En Popüler Direktörlerin Rating Tercihleri", fontsize=16, color='white')
        plt.xlabel("Rating Grubu", fontsize=14, color='white')
        plt.ylabel("Direktör", fontsize=14, color='white')

        # Renk çubuğu etiketini beyaz yap
        cbar = ax.collections[0].colorbar
        cbar.ax.yaxis.label.set_color('white')
        cbar.ax.tick_params(colors='white')

        plt.tick_params(axis='both', colors='white')

        plt.tight_layout()
        plt.savefig(director_rating_path, facecolor='black', edgecolor='none')
        print(f"Direktör-rating heatmap grafiği kaydedildi: {director_rating_path}")
        plt.close()
    else:
        print(f"Direktör-rating heatmap grafiği zaten mevcut: {director_rating_path}")

    # Direktör kelime bulutu
    print("Direktör isimlerinden kelime bulutu oluşturuluyor...")

    # Tüm direktörleri bir metin olarak birleştirelim
    all_directors_text = ' '.join(director_exploded['director'].tolist())

    # Kelime bulutu oluştur
    wordcloud_path = os.path.join(fig_dir, "netflix_directors_wordcloud.png")

    if not os.path.exists(wordcloud_path):
        wordcloud = WordCloud(width=800, height=400, background_color='black',
                              colormap='Reds', max_words=100).generate(all_directors_text)

        plt.figure(figsize=(10, 8))
        plt.style.use('dark_background')
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title("Netflix Direktörleri Kelime Bulutu", fontsize=16, color='white')
        plt.tight_layout()
        plt.savefig(wordcloud_path, facecolor='black', edgecolor='none')
        print(f"Direktörler kelime bulutu kaydedildi: {wordcloud_path}")
        plt.close()
    else:
        print(f"Direktörler kelime bulutu zaten mevcut: {wordcloud_path}")

    print("Netflix direktör analizi tamamlandı!")

main()