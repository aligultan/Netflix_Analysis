import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import seaborn as sns
import os
from collections import Counter

# Görsel stili ayarlama
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("viridis")

# Grafiklerin kaydedileceği dizini kontrol etme ve oluşturma
if not os.path.exists('graphics/curve_fitting'):
    os.makedirs('graphics/curve_fitting')


# Veri setini yükleme
def load_data():
    try:
        netflix_data = pd.read_csv('data/netflix1.csv')
        return netflix_data
    except FileNotFoundError:
        print(
            "Veri dosyası bulunamadı. Lütfen 'data' klasöründe 'netflix_titles.csv' dosyasının var olduğundan emin olun.")
        return None


# Curve fitting fonksiyonları
def linear_func(x, a, b):
    return a * x + b


def poly_func(x, a, b, c):
    return a * x ** 2 + b * x + c


def exp_func(x, a, b, c):
    return a * np.exp(b * x) + c


def logistic_func(x, L, k, x0):
    return L / (1 + np.exp(-k * (x - x0)))


# R-kare değerini hesaplama
def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)


# Genre bazlı analiz
def analyze_genres(data):
    # Liste olarak saklanan 'listed_in' (genre) sütununu ayırma
    genres = []
    for g in data['listed_in']:
        if isinstance(g, str):
            genres.extend([genre.strip() for genre in g.split(',')])

    # En popüler 10 türü bulma
    genre_counter = Counter(genres)
    top_genres = dict(genre_counter.most_common(10))

    # Her tür için yıla göre içerik sayısını hesaplama
    genre_growth = {}

    for genre in top_genres:
        # Bu türü içeren içerikleri filtrele
        genre_data = data[data['listed_in'].apply(lambda x: isinstance(x, str) and genre in x)]

        # Yıla göre grupla
        yearly_count = genre_data.groupby('release_year').size().reset_index(name='count')
        yearly_count = yearly_count[yearly_count['release_year'] >= 2000]

        if not yearly_count.empty:
            genre_growth[genre] = yearly_count

    # Türlere göre curve fitting uygula
    plt.figure(figsize=(15, 10))

    for i, (genre, yearly_data) in enumerate(genre_growth.items()):
        x_data = yearly_data['release_year'].values - 2000
        y_data = yearly_data['count'].values

        try:
            # Polinom modeli uygulama
            popt, _ = curve_fit(poly_func, x_data, y_data)

            # Model performansını değerlendirme
            y_pred = poly_func(x_data, *popt)
            r2 = r_squared(y_data, y_pred)

            # Görselleştirme
            x_smooth = np.linspace(min(x_data), max(x_data), 100)
            y_smooth = poly_func(x_smooth, *popt)

            # Tahmin - gelecek 5 yıl
            future_years = np.arange(max(x_data) + 1, max(x_data) + 6)
            future_counts = poly_func(future_years, *popt)

            # Çizim
            plt.scatter(x_data + 2000, y_data, alpha=0.6, label=f'{genre} (Veri)')
            plt.plot(x_smooth + 2000, y_smooth, linewidth=2)
            plt.plot(future_years + 2000, future_counts, '--', linewidth=2)

            print(f"{genre}: R²={r2:.4f}, 2025 tahmini: {int(poly_func(25, *popt))} içerik")

        except Exception as e:
            print(f"{genre} için curve fitting yapılamadı: {e}")

    # Grafiği biçimlendirme
    plt.title('Netflix Türlerine Göre İçerik Büyüme Eğrileri ve Tahminler', fontsize=16)
    plt.xlabel('Yıl', fontsize=14)
    plt.ylabel('İçerik Sayısı', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()

    # Grafiği kaydetme
    plt.savefig('graphics/curve_fitting/netflix_genre_growth_prediction.png', dpi=300, bbox_inches='tight')
    plt.close()


# Ülke bazlı analiz
def analyze_countries(data):
    # Boş olmayan ülke verilerini seçme
    country_data = data.dropna(subset=['country'])

    # Her içeriğin ilk ülkesini alma (birden fazla ülke olabilir)
    country_data['main_country'] = country_data['country'].apply(lambda x: x.split(',')[0].strip())

    # En çok içeriği olan 6 ülkeyi bulma
    top_countries = country_data['main_country'].value_counts().head(6).index

    plt.figure(figsize=(15, 10))

    for i, country in enumerate(top_countries):
        # Bu ülkeye ait içerikleri filtrele
        country_contents = country_data[country_data['main_country'] == country]

        # Yıla göre grupla
        yearly_count = country_contents.groupby('release_year').size().reset_index(name='count')
        yearly_count = yearly_count[yearly_count['release_year'] >= 2000]

        if not yearly_count.empty:
            x_data = yearly_count['release_year'].values - 2000
            y_data = yearly_count['count'].values

            try:
                # Polinom modeli uygulama
                popt, _ = curve_fit(poly_func, x_data, y_data)

                # Görselleştirme
                x_smooth = np.linspace(min(x_data), max(x_data), 100)
                y_smooth = poly_func(x_smooth, *popt)

                # Tahmin - gelecek 5 yıl
                future_years = np.arange(max(x_data) + 1, max(x_data) + 6)
                future_counts = poly_func(future_years, *popt)

                # Çizim
                plt.scatter(x_data + 2000, y_data, alpha=0.6, label=f'{country} (Veri)')
                plt.plot(x_smooth + 2000, y_smooth, linewidth=2)
                plt.plot(future_years + 2000, future_counts, '--', linewidth=2)

                # 2025 tahmini
                pred_2025 = poly_func(25, *popt)  # 2025 - 2000 = 25
                print(f"{country}: 2025 tahmini: {int(pred_2025)} içerik")

            except Exception as e:
                print(f"{country} için curve fitting yapılamadı: {e}")

    # Grafiği biçimlendirme
    plt.title('Netflix Ülkelere Göre İçerik Büyüme Eğrileri ve Tahminler', fontsize=16)
    plt.xlabel('Yıl', fontsize=14)
    plt.ylabel('İçerik Sayısı', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()

    # Grafiği kaydetme
    plt.savefig('graphics/curve_fitting/netflix_country_growth_prediction.png', dpi=300, bbox_inches='tight')
    plt.close()


# Rating bazlı analiz
def analyze_ratings(data):
    # Boş olmayan rating verilerini seçme
    rating_data = data.dropna(subset=['rating'])

    # En çok kullanılan 6 derecelendirmeyi bulma
    top_ratings = rating_data['rating'].value_counts().head(6).index

    plt.figure(figsize=(15, 10))

    for i, rating in enumerate(top_ratings):
        # Bu derecelendirmeye sahip içerikleri filtrele
        rating_contents = rating_data[rating_data['rating'] == rating]

        # Yıla göre grupla
        yearly_count = rating_contents.groupby('release_year').size().reset_index(name='count')
        yearly_count = yearly_count[yearly_count['release_year'] >= 2000]

        if not yearly_count.empty and len(yearly_count) > 3:  # En az 4 veri noktası olsun
            x_data = yearly_count['release_year'].values - 2000
            y_data = yearly_count['count'].values

            try:
                # Polinom modeli uygulama
                popt, _ = curve_fit(poly_func, x_data, y_data)

                # Görselleştirme
                x_smooth = np.linspace(min(x_data), max(x_data), 100)
                y_smooth = poly_func(x_smooth, *popt)

                # Tahmin - gelecek 5 yıl
                future_years = np.arange(max(x_data) + 1, max(x_data) + 6)
                future_counts = poly_func(future_years, *popt)

                # Çizim
                plt.scatter(x_data + 2000, y_data, alpha=0.6, label=f'{rating} (Veri)')
                plt.plot(x_smooth + 2000, y_smooth, linewidth=2)
                plt.plot(future_years + 2000, future_counts, '--', linewidth=2)

                # 2025 tahmini
                pred_2025 = poly_func(25, *popt)  # 2025 - 2000 = 25
                print(f"{rating}: 2025 tahmini: {int(pred_2025)} içerik")

            except Exception as e:
                print(f"{rating} için curve fitting yapılamadı: {e}")

    # Grafiği biçimlendirme
    plt.title('Netflix Derecelendirmelere Göre İçerik Büyüme Eğrileri ve Tahminler', fontsize=16)
    plt.xlabel('Yıl', fontsize=14)
    plt.ylabel('İçerik Sayısı', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()

    # Grafiği kaydetme
    plt.savefig('graphics/curve_fitting/netflix_rating_growth_prediction.png', dpi=300, bbox_inches='tight')
    plt.close()


def main():
    # Veri setini yükleme
    netflix_data = load_data()

    if netflix_data is not None:
        print("Netflix veri seti başarıyla yüklendi. Toplam kayıt sayısı:", len(netflix_data))

        # Tür bazlı analiz
        print("\n=== Türlere Göre Büyüme Analizi ===")
        analyze_genres(netflix_data)

        # Ülke bazlı analiz
        print("\n=== Ülkelere Göre Büyüme Analizi ===")
        analyze_countries(netflix_data)

        # Rating bazlı analiz
        print("\n=== Derecelendirmelere Göre Büyüme Analizi ===")
        analyze_ratings(netflix_data)

        print("\nTüm analizler tamamlandı. Sonuçları 'graphics/curve_fitting' klasöründe bulabilirsiniz.")


if __name__ == "__main__":
    main()