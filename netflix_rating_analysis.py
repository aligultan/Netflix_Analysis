import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def main():
    fig_dir = "graphics"
    os.makedirs(fig_dir, exist_ok=True)

    print("Netflix rating analizi başlatılıyor...")

    df = pd.read_csv("data/netflix1.csv")
    # Rating (yaş sınırı) dağılımını analiz et
    print("Rating dağılımı analiz ediliyor...")

    # NaN değerlerini "Belirtilmemiş" olarak işaretleyelim
    df['rating'] = df['rating'].fillna('Belirtilmemiş')

    rating_counts = df['rating'].value_counts().sort_values(ascending=False)

    # Rating grafiği
    rating_fig_path = os.path.join(fig_dir, "netflix_rating_distribution.png")

    if not os.path.exists(rating_fig_path):
        plt.figure(figsize=(12, 8))
        plt.style.use('dark_background')

        # Netflix kırmızısı ve tonları
        colors = sns.color_palette("Reds_r", n_colors=len(rating_counts))

        ax = sns.barplot(y=rating_counts.index, x=rating_counts.values, palette=colors)

        # Bar değerleri
        for i, v in enumerate(rating_counts.values):
            ax.text(v + 10, i, str(v), va='center', color='white', fontweight='bold')

        plt.title("Netflix İçerik Rating Dağılımı", fontsize=16, color='white')
        plt.xlabel("İçerik Sayısı", fontsize=14, color='white')
        plt.ylabel("Rating (Yaş Sınırı)", fontsize=14, color='white')

        # Grafik stil
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('#333333')
        ax.spines['left'].set_color('#333333')
        ax.grid(axis='x', linestyle='--', alpha=0.3, color='#555555')

        plt.tight_layout()
        plt.savefig(rating_fig_path, facecolor='black', edgecolor='none')
        print(f"Rating dağılımı grafiği kaydedildi: {rating_fig_path}")
        plt.close()
    else:
        print(f"Rating dağılımı grafiği zaten mevcut: {rating_fig_path}")

    # Rating gruplarını sınıflandır
    print("Rating grupları oluşturuluyor...")

    def classify_rating(rating):
        if rating == 'Belirtilmemiş':
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

    df['rating_group'] = df['rating'].apply(classify_rating)
    rating_group_counts = df['rating_group'].value_counts()

    rating_pie_path = os.path.join(fig_dir, "netflix_rating_groups_pie.png")

    if not os.path.exists(rating_pie_path):
        plt.figure(figsize=(10, 8))
        plt.style.use('dark_background')

        # Netflix kırmızısı ve tonları
        colors = ['#E50914', '#B20710', '#831010', '#5C0B0B', '#DB0000', '#A30000']

        plt.pie(rating_group_counts.values, labels=rating_group_counts.index, autopct='%1.1f%%',
                startangle=90, colors=colors, wedgeprops={'edgecolor': 'black', 'linewidth': 1},
                textprops={'color': 'white'})

        plt.title("Netflix İçeriklerinin Yaş Sınıfı Dağılımı", fontsize=16, color='white')
        plt.axis('equal')

        plt.tight_layout()
        plt.savefig(rating_pie_path, facecolor='black', edgecolor='none')
        print(f"Rating grupları pasta grafiği kaydedildi: {rating_pie_path}")
        plt.close()
    else:
        print(f"Rating grupları pasta grafiği zaten mevcut: {rating_pie_path}")

    # Film ve Dizilerde Rating Dağılımı
    print("Film ve dizilerde rating dağılımı analiz ediliyor...")

    # Rating ve içerik türü ilişkisi
    rating_type = df.groupby(['rating', 'type']).size().unstack(fill_value=0)

    # En çok kullanılan 10 rating'i seçelim (grafiği daha okunaklı yapmak için)
    top_ratings = rating_counts.head(10).index
    rating_type_filtered = rating_type.loc[top_ratings]

    rating_type_path = os.path.join(fig_dir, "netflix_rating_by_type.png")

    if not os.path.exists(rating_type_path):
        plt.figure(figsize=(12, 8))
        plt.style.use('dark_background')

        # Netflix kırmızısı ve tonları
        colors = ['#E50914', '#831010']

        ax = rating_type_filtered.plot(kind='bar', color=colors, width=0.8)

        plt.title("Rating Türlerine Göre Film ve Dizi Dağılımı", fontsize=16, color='white')
        plt.xlabel("Rating", fontsize=14, color='white')
        plt.ylabel("İçerik Sayısı", fontsize=14, color='white')
        plt.legend(facecolor='#222222', edgecolor='#444444', labelcolor='white')

        plt.grid(axis='y', linestyle='--', alpha=0.3, color='#555555')
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['bottom'].set_color('#333333')
        plt.gca().spines['left'].set_color('#333333')
        plt.tick_params(axis='both', colors='white')

        plt.tight_layout()
        plt.savefig(rating_type_path, facecolor='black', edgecolor='none')
        print(f"Rating-tür ilişkisi grafiği kaydedildi: {rating_type_path}")
        plt.close()
    else:
        print(f"Rating-tür ilişkisi grafiği zaten mevcut: {rating_type_path}")

    print("Rating'lerin yıllara göre değişimi analiz ediliyor...")

    # date_added sütunundan yıl bilgisini çıkaralım
    df['date_added'] = pd.to_datetime(df['date_added'], errors='coerce')
    df['year_added'] = df['date_added'].dt.year

    # NaN değerleri olan satırları filtreleyelim
    year_rating_df = df.dropna(subset=['year_added', 'rating_group'])

    # En popüler 4 rating grubunu seçelim
    popular_rating_groups = rating_group_counts.head(4).index
    year_rating_filtered = year_rating_df[year_rating_df['rating_group'].isin(popular_rating_groups)]

    # Yıla ve rating grubuna göre içerik sayısını hesaplayalım
    rating_trend = year_rating_filtered.groupby(['year_added', 'rating_group']).size().unstack(fill_value=0)

    # 2008 öncesi çok az veri var, 2008 sonrasını alalım
    rating_trend = rating_trend[rating_trend.index >= 2008]

    rating_trend_path = os.path.join(fig_dir, "netflix_rating_trend_by_year.png")

    if not os.path.exists(rating_trend_path):
        plt.figure(figsize=(12, 8))
        plt.style.use('dark_background')

        colors = ['#E50914', '#B20710', '#831010', '#5C0B0B']

        rating_trend.plot(kind='line', marker='o', color=colors, linewidth=2.5)

        plt.title("Yıllara Göre Rating Gruplarının Değişimi", fontsize=16, color='white')
        plt.xlabel("Yıl", fontsize=14, color='white')
        plt.ylabel("İçerik Sayısı", fontsize=14, color='white')
        plt.legend(facecolor='#222222', edgecolor='#444444', labelcolor='white')

        plt.grid(linestyle='--', alpha=0.3, color='#555555')
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['bottom'].set_color('#333333')
        plt.gca().spines['left'].set_color('#333333')
        plt.tick_params(axis='both', colors='white')

        plt.tight_layout()
        plt.savefig(rating_trend_path, facecolor='black', edgecolor='none')
        print(f"Rating trendi grafiği kaydedildi: {rating_trend_path}")
        plt.close()
    else:
        print(f"Rating trendi grafiği zaten mevcut: {rating_trend_path}")

    # Rating ve Ülke İlişkisi
    print("Rating ve ülke ilişkisi analiz ediliyor...")

    # En çok içeriğe sahip ülkeleri belirleyelim
    country_df = df.dropna(subset=['country'])
    country_df['country'] = country_df['country'].str.split(', ')
    country_exploded = country_df.explode('country')

    top_countries = country_exploded['country'].value_counts().head(5).index
    country_rating_df = country_exploded[country_exploded['country'].isin(top_countries)]

    # Ülke-Rating matrisi
    country_rating_matrix = pd.crosstab(country_rating_df['country'], country_rating_df['rating_group'])

    country_rating_matrix = country_rating_matrix[popular_rating_groups]

    country_rating_path = os.path.join(fig_dir, "netflix_country_rating_heatmap.png")

    if not os.path.exists(country_rating_path):
        plt.figure(figsize=(12, 8))
        plt.style.use('dark_background')

        cmap = sns.color_palette("Reds", as_cmap=True)

        ax = sns.heatmap(country_rating_matrix, annot=True, fmt='d', cmap=cmap,
                         linewidths=0.5, cbar_kws={'label': 'İçerik Sayısı'})

        plt.title("Ülkelere Göre Rating Dağılımı", fontsize=16, color='white')
        plt.xlabel("Rating Grubu", fontsize=14, color='white')
        plt.ylabel("Ülke", fontsize=14, color='white')

        # Renk çubuğu etiketini beyaz yap
        cbar = ax.collections[0].colorbar
        cbar.ax.yaxis.label.set_color('white')
        cbar.ax.tick_params(colors='white')

        plt.tick_params(axis='both', colors='white')

        plt.tight_layout()
        plt.savefig(country_rating_path, facecolor='black', edgecolor='none')
        print(f"Ülke-Rating heatmap grafiği kaydedildi: {country_rating_path}")
        plt.close()
    else:
        print(f"Ülke-Rating heatmap grafiği zaten mevcut: {country_rating_path}")

    # Rating ve Kategori İlişkisi
    print("Rating ve kategori ilişkisi analiz ediliyor...")

    category_df = df.dropna(subset=['listed_in', 'rating_group'])
    category_df['listed_in'] = category_df['listed_in'].str.split(', ')
    category_exploded = category_df.explode('listed_in')

    # En popüler kategorileri seçelim
    top_categories = category_exploded['listed_in'].value_counts().head(8).index
    category_rating_df = category_exploded[category_exploded['listed_in'].isin(top_categories)]

    # Kategori-Rating matrisi
    category_rating_matrix = pd.crosstab(category_rating_df['listed_in'], category_rating_df['rating_group'])

    category_rating_matrix = category_rating_matrix[popular_rating_groups]

    category_rating_path = os.path.join(fig_dir, "netflix_category_rating_heatmap.png")

    if not os.path.exists(category_rating_path):
        plt.figure(figsize=(12, 10))
        plt.style.use('dark_background')

        # Kırmızı tonlarında bir renk haritası
        cmap = sns.color_palette("Reds", as_cmap=True)

        ax = sns.heatmap(category_rating_matrix, annot=True, fmt='d', cmap=cmap,
                         linewidths=0.5, cbar_kws={'label': 'İçerik Sayısı'})

        plt.title("Kategorilere Göre Rating Dağılımı", fontsize=16, color='white')
        plt.xlabel("Rating Grubu", fontsize=14, color='white')
        plt.ylabel("Kategori", fontsize=14, color='white')

        # Renk çubuğu etiketini beyaz yap
        cbar = ax.collections[0].colorbar
        cbar.ax.yaxis.label.set_color('white')
        cbar.ax.tick_params(colors='white')

        plt.tick_params(axis='both', colors='white')

        plt.tight_layout()
        plt.savefig(category_rating_path, facecolor='black', edgecolor='none')
        print(f"Kategori-Rating heatmap grafiği kaydedildi: {category_rating_path}")
        plt.close()
    else:
        print(f"Kategori-Rating heatmap grafiği zaten mevcut: {category_rating_path}")

    print("Netflix rating analizi tamamlandı!")

main()