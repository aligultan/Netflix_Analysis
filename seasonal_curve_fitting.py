import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import seaborn as sns
import os
import re

# Görsel stili ayarlama
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("Set2")

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


# R-kare değerini hesaplama
def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)


# TV Show'ların sezon sayılarını çıkarma
def extract_seasons(description):
    if not isinstance(description, str):
        return None

    # "X seasons" veya "Season X" gibi kalıpları arama
    seasons_pattern = r'(\d+)\s+season'
    season_match = re.search(seasons_pattern, description.lower())

    if season_match:
        return int(season_match.group(1))
    else:
        # Tek sezon olabilir
        if 'season' in description.lower() and 'seasons' not in description.lower():
            return 1

    return None


# Sezonluk içeriklerin (TV Shows) analizi
def analyze_tv_shows(data):
    # Sadece TV Show'ları seçme
    tv_shows = data[data['type'] == 'TV Show'].copy()

    # Sezon sayılarını çıkarma
    tv_shows['seasons'] = tv_shows['description'].apply(extract_seasons)

    # Eksik değerleri description'dan farklı bir yöntemle tekrar doldurmaya çalışalım
    # Bazen sezon bilgisi title içinde olabilir
    for idx, row in tv_shows[tv_shows['seasons'].isna()].iterrows():
        if isinstance(row['title'], str) and 'season' in row['title'].lower():
            seasons_pattern = r'(\d+)\s+season'
            season_match = re.search(seasons_pattern, row['title'].lower())
            if season_match:
                tv_shows.at[idx, 'seasons'] = int(season_match.group(1))

    # Eksik değerleri (NaN) olan kayıtları kaldırma
    tv_shows = tv_shows.dropna(subset=['seasons'])

    # 10'dan fazla sezonu olan şovlar için bir üst limit belirleyelim
    max_seasons = 10
    tv_shows.loc[tv_shows['seasons'] > max_seasons, 'seasons'] = max_seasons

    # Sezonlara göre TV Show sayısını hesaplama
    season_counts = tv_shows['seasons'].value_counts().sort_index()

    # Görselleştirme
    plt.figure(figsize=(12, 8))

    # Sütun grafiği
    ax = season_counts.plot(kind='bar', color=sns.color_palette("Set2"))

    # Her sütunun üzerine değeri yazma
    for i, v in enumerate(season_counts):
        ax.text(i, v + 5, str(v), ha='center')

    plt.title('Netflix TV Show Sezon Dağılımı', fontsize=16)
    plt.xlabel('Sezon Sayısı', fontsize=14)
    plt.ylabel('TV Show Sayısı', fontsize=14)
    plt.xticks(rotation=0)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    # Grafiği kaydetme
    plt.savefig('graphics/curve_fitting/netflix_tv_show_season_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Yıllara göre ortalama sezon sayısının analizi
    yearly_avg_seasons = tv_shows.groupby('release_year')['seasons'].mean().reset_index()
    yearly_avg_seasons = yearly_avg_seasons[yearly_avg_seasons['release_year'] >= 2000]

    if not yearly_avg_seasons.empty:
        x_data = yearly_avg_seasons['release_year'].values - 2000  # Normalize years
        y_data = yearly_avg_seasons['seasons'].values

        try:
            # Polinom modeli uygulama
            popt, _ = curve_fit(poly_func, x_data, y_data)

            # Model performansını değerlendirme
            y_pred = poly_func(x_data, *popt)
            r2 = r_squared(y_data, y_pred)

            # Görselleştirme
            plt.figure(figsize=(12, 8))

            plt.scatter(x_data + 2000, y_data, alpha=0.7, label='Gerçek Veriler', color='blue')

            # Eğri
            x_smooth = np.linspace(min(x_data), max(x_data), 100)
            y_smooth = poly_func(x_smooth, *popt)
            plt.plot(x_smooth + 2000, y_smooth, linewidth=2, label=f'Polinom Model (R²={r2:.3f})', color='red')

            # Tahmin - gelecek 5 yıl
            future_years = np.arange(max(x_data) + 1, max(x_data) + 6)
            future_avg_seasons = poly_func(future_years, *popt)
            plt.plot(future_years + 2000, future_avg_seasons, '--', linewidth=2, label='Tahmin', color='green')

            # Tahmin bölgesini belirtmek için dikey çizgi
            plt.axvline(x=max(x_data) + 2000, color='gray', linestyle='--', alpha=0.7)

            plt.title('Netflix TV Show Ortalama Sezon Sayısı Trend Analizi', fontsize=16)
            plt.xlabel('Yıl', fontsize=14)
            plt.ylabel('Ortalama Sezon Sayısı', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()

            # 2025 yılı için tahmin
            pred_2025 = poly_func(25, *popt)  # 2025 - 2000 = 25
            print(f"2025 yılı için ortalama sezon sayısı tahmini: {pred_2025:.2f}")

            # Grafiği kaydetme
            plt.savefig('graphics/curve_fitting/netflix_tv_show_season_trend.png', dpi=300, bbox_inches='tight')
            plt.close()

        except Exception as e:
            print(f"TV Show sezon trendleri için curve fitting yapılamadı: {e}")

    # Yıllara göre TV Show sayısının analizi
    yearly_tv_shows = tv_shows.groupby('release_year').size().reset_index(name='count')
    yearly_tv_shows = yearly_tv_shows[yearly_tv_shows['release_year'] >= 2000]

    if not yearly_tv_shows.empty:
        x_data = yearly_tv_shows['release_year'].values - 2000
        y_data = yearly_tv_shows['count'].values

        try:
            # Üç farklı model uygulama
            popt_linear, _ = curve_fit(linear_func, x_data, y_data)
            popt_poly, _ = curve_fit(poly_func, x_data, y_data)
            popt_exp, _ = curve_fit(exp_func, x_data, y_data, p0=[1, 0.1, 1])

            # R-kare değerlerini hesaplama
            r2_linear = r_squared(y_data, linear_func(x_data, *popt_linear))
            r2_poly = r_squared(y_data, poly_func(x_data, *popt_poly))
            r2_exp = r_squared(y_data, exp_func(x_data, *popt_exp))

            # Görselleştirme
            plt.figure(figsize=(12, 8))

            plt.scatter(x_data + 2000, y_data, label='Gerçek Veriler', alpha=0.7, color='blue')

            # Modeller
            x_smooth = np.linspace(min(x_data), max(x_data), 100)

            plt.plot(x_smooth + 2000, linear_func(x_smooth, *popt_linear),
                     label=f'Lineer Model (R²={r2_linear:.3f})', linewidth=2, color='red')

            plt.plot(x_smooth + 2000, poly_func(x_smooth, *popt_poly),
                     label=f'Polinom Model (R²={r2_poly:.3f})', linewidth=2, color='green')

            plt.plot(x_smooth + 2000, exp_func(x_smooth, *popt_exp),
                     label=f'Üstel Model (R²={r2_exp:.3f})', linewidth=2, color='purple')

            # Tahminler - gelecek 5 yıl
            future_years = np.arange(max(x_data) + 1, max(x_data) + 6)

            plt.plot(future_years + 2000, linear_func(future_years, *popt_linear), '--', linewidth=2, color='red')
            plt.plot(future_years + 2000, poly_func(future_years, *popt_poly), '--', linewidth=2, color='green')
            plt.plot(future_years + 2000, exp_func(future_years, *popt_exp), '--', linewidth=2, color='purple')

            # Tahmin bölgesini belirtmek için dikey çizgi
            plt.axvline(x=max(x_data) + 2000, color='gray', linestyle='--', alpha=0.7)

            plt.title('Netflix TV Show Sayısı Büyüme Eğrileri ve Tahminler', fontsize=16)
            plt.xlabel('Yıl', fontsize=14)
            plt.ylabel('TV Show Sayısı', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()

            # 2025 yılı için tahmin
            pred_2025_linear = linear_func(25, *popt_linear)
            pred_2025_poly = poly_func(25, *popt_poly)
            pred_2025_exp = exp_func(25, *popt_exp)

            print(f"\n2025 yılı için TV Show sayısı tahminleri:")
            print(f"Lineer Model: {int(pred_2025_linear)} TV Show")
            print(f"Polinom Model: {int(pred_2025_poly)} TV Show")
            print(f"Üstel Model: {int(pred_2025_exp)} TV Show")

            # En iyi modeli belirleme
            best_r2 = max(r2_linear, r2_poly, r2_exp)
            best_model = "Lineer" if best_r2 == r2_linear else "Polinom" if best_r2 == r2_poly else "Üstel"
            print(f"En iyi model: {best_model} Model (R²={best_r2:.4f})")

            # Grafiği kaydetme
            plt.savefig('graphics/curve_fitting/netflix_tv_show_growth_prediction.png', dpi=300, bbox_inches='tight')
            plt.close()

        except Exception as e:
            print(f"TV Show büyüme eğrileri için curve fitting yapılamadı: {e}")


def main():
    # Veri setini yükleme
    netflix_data = load_data()

    if netflix_data is not None:
        print("Netflix veri seti başarıyla yüklendi. Toplam kayıt sayısı:", len(netflix_data))

        # TV Show analizi
        print("\n=== TV Show Sezon Analizi ===")
        analyze_tv_shows(netflix_data)

        print("\nAnalizler tamamlandı. Sonuçları 'graphics/curve_fitting' klasöründe bulabilirsiniz.")


if __name__ == "__main__":
    main()