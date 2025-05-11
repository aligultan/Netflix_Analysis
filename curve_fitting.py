import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import seaborn as sns
import os

# Görsel stili ayarlama
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("muted")
colors = sns.color_palette("muted", 10)

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


# İlk model: Lineer fonksiyon
def linear_func(x, a, b):
    return a * x + b


# İkinci model: Polinom fonksiyon (2. derece)
def poly_func(x, a, b, c):
    return a * x ** 2 + b * x + c


# Üçüncü model: Exponential fonksiyon
def exp_func(x, a, b, c):
    return a * np.exp(b * x) + c


# Curve fitting uygulama ve sonuçları görselleştirme
def apply_curve_fitting(data):
    # Yıla göre içerik sayısını hesaplama
    data['release_year'] = pd.to_numeric(data['release_year'], errors='coerce')
    yearly_content = data.groupby('release_year').size().reset_index(name='content_count')

    # 2000 yılından sonraki verilere odaklanma
    recent_data = yearly_content[yearly_content['release_year'] >= 2000]

    # Verileri hazırlama
    x_data = recent_data['release_year'].values - 2000  # Hesaplamaları kolaylaştırmak için yılları normalize etme
    y_data = recent_data['content_count'].values

    # Eğrileri uydurma
    try:
        # Lineer model
        popt_linear, pcov_linear = curve_fit(linear_func, x_data, y_data)

        # Polinom model
        popt_poly, pcov_poly = curve_fit(poly_func, x_data, y_data)

        # Üstel model
        popt_exp, pcov_exp = curve_fit(exp_func, x_data, y_data, p0=[1, 0.1, 1])  # Başlangıç değerleri

        # Modellerin performansını değerlendirme
        x_line = np.linspace(min(x_data), max(x_data), 100)
        y_linear = linear_func(x_line, *popt_linear)
        y_poly = poly_func(x_line, *popt_poly)
        y_exp = exp_func(x_line, *popt_exp)

        # R-kare değerlerini hesaplama
        def r_squared(y_true, y_pred):
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            return 1 - (ss_res / ss_tot)

        r2_linear = r_squared(y_data, linear_func(x_data, *popt_linear))
        r2_poly = r_squared(y_data, poly_func(x_data, *popt_poly))
        r2_exp = r_squared(y_data, exp_func(x_data, *popt_exp))

        # Sonuçları görselleştirme
        plt.figure(figsize=(12, 8))
        plt.scatter(x_data + 2000, y_data, label='Gerçek Veriler', color=colors[0], alpha=0.7)
        plt.plot(x_line + 2000, y_linear, label=f'Lineer Model (R² = {r2_linear:.3f})', color=colors[1], linewidth=2)
        plt.plot(x_line + 2000, y_poly, label=f'Polinom Model (R² = {r2_poly:.3f})', color=colors[2], linewidth=2)
        plt.plot(x_line + 2000, y_exp, label=f'Üstel Model (R² = {r2_exp:.3f})', color=colors[3], linewidth=2)

        plt.title('Netflix İçerik Sayısının Yıllara Göre Artışı ve Eğri Uydurma Modelleri', fontsize=16)
        plt.xlabel('Yıl', fontsize=14)
        plt.ylabel('İçerik Sayısı', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Grafiği kaydetme
        plt.savefig('graphics/curve_fitting/netflix_content_growth_models.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Gelecek 5 yıl için tahmin
        future_years = np.arange(max(x_data) + 1, max(x_data) + 6)

        # Modellere göre tahminler
        future_linear = linear_func(future_years, *popt_linear)
        future_poly = poly_func(future_years, *popt_poly)
        future_exp = exp_func(future_years, *popt_exp)

        # Tahminleri görselleştirme
        plt.figure(figsize=(12, 8))
        plt.scatter(x_data + 2000, y_data, label='Gerçek Veriler', color=colors[0], alpha=0.7)

        # Mevcut veriler için eğriler
        plt.plot(x_line + 2000, y_linear, color=colors[1], linewidth=2)
        plt.plot(x_line + 2000, y_poly, color=colors[2], linewidth=2)
        plt.plot(x_line + 2000, y_exp, color=colors[3], linewidth=2)

        # Gelecek tahminler
        plt.plot(future_years + 2000, future_linear, '--', color=colors[1], linewidth=2)
        plt.plot(future_years + 2000, future_poly, '--', color=colors[2], linewidth=2)
        plt.plot(future_years + 2000, future_exp, '--', color=colors[3], linewidth=2)

        # Tahmin bölgesini belirtmek için dikey çizgi
        plt.axvline(x=max(x_data) + 2000, color='gray', linestyle='--', alpha=0.7)

        # Etiketler
        plt.text(max(x_data) + 2000 + 1, max(y_data) * 0.9, 'Tahmin', fontsize=12, color='gray')
        plt.text(max(x_data) + 2000 - 4, max(y_data) * 0.9, 'Gerçek Veriler', fontsize=12, color='gray')

        plt.title('Netflix İçerik Sayısı: Gerçek Veriler ve Gelecek 5 Yıl Tahmini', fontsize=16)
        plt.xlabel('Yıl', fontsize=14)
        plt.ylabel('İçerik Sayısı', fontsize=14)
        plt.legend(['Gerçek Veriler', 'Lineer Model', 'Polinom Model', 'Üstel Model'])
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Grafiği kaydetme
        plt.savefig('graphics/curve_fitting/netflix_content_prediction.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Türlere göre curve fitting analizi
        content_types = data['type'].unique()

        plt.figure(figsize=(12, 8))

        for i, content_type in enumerate(content_types):
            type_data = data[data['type'] == content_type]
            yearly_type = type_data.groupby('release_year').size().reset_index(name='count')
            yearly_type = yearly_type[yearly_type['release_year'] >= 2000]

            if not yearly_type.empty:
                x_type = yearly_type['release_year'].values - 2000
                y_type = yearly_type['count'].values

                try:
                    # Polinom modeli uygulama (en iyi performansı genelde bu gösteriyor)
                    popt_type, _ = curve_fit(poly_func, x_type, y_type)

                    # Görselleştirme
                    x_smooth = np.linspace(min(x_type), max(x_type), 100)
                    y_smooth = poly_func(x_smooth, *popt_type)

                    plt.scatter(x_type + 2000, y_type, label=f'{content_type} (Veri)', alpha=0.5, color=colors[i])
                    plt.plot(x_smooth + 2000, y_smooth, label=f'{content_type} (Model)', linewidth=2, color=colors[i])
                except:
                    print(f"{content_type} için curve fitting yapılamadı.")

        plt.title('Netflix İçerik Türlerine Göre Büyüme Eğrileri', fontsize=16)
        plt.xlabel('Yıl', fontsize=14)
        plt.ylabel('İçerik Sayısı', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Grafiği kaydetme
        plt.savefig('graphics/curve_fitting/netflix_content_by_type_growth.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Sonuçları yazdırma
        print("\n=== Curve Fitting Sonuçları ===")
        print(f"Lineer Model Parametreleri: a={popt_linear[0]:.4f}, b={popt_linear[1]:.4f}, R²={r2_linear:.4f}")
        print(
            f"Polinom Model Parametreleri: a={popt_poly[0]:.4f}, b={popt_poly[1]:.4f}, c={popt_poly[2]:.4f}, R²={r2_poly:.4f}")
        print(
            f"Üstel Model Parametreleri: a={popt_exp[0]:.4f}, b={popt_exp[1]:.4f}, c={popt_exp[2]:.4f}, R²={r2_exp:.4f}")

        # En iyi modeli belirleme
        best_r2 = max(r2_linear, r2_poly, r2_exp)
        best_model = "Lineer" if best_r2 == r2_linear else "Polinom" if best_r2 == r2_poly else "Üstel"
        print(f"\nEn iyi model: {best_model} Model (R²={best_r2:.4f})")

        # 2025 yılı için tahmin
        year_2025 = 25  # 2025 - 2000 = 25
        pred_2025_linear = linear_func(year_2025, *popt_linear)
        pred_2025_poly = poly_func(year_2025, *popt_poly)
        pred_2025_exp = exp_func(year_2025, *popt_exp)

        print(f"\n2025 Yılı İçerik Sayısı Tahmini:")
        print(f"Lineer Model: {int(pred_2025_linear)} içerik")
        print(f"Polinom Model: {int(pred_2025_poly)} içerik")
        print(f"Üstel Model: {int(pred_2025_exp)} içerik")

    except Exception as e:
        print(f"Curve fitting işlemi sırasında hata oluştu: {e}")


def main():
    # Veri setini yükleme
    netflix_data = load_data()

    if netflix_data is not None:
        print("Netflix veri seti başarıyla yüklendi. Toplam kayıt sayısı:", len(netflix_data))

        # Curve fitting işlemini uygulama
        apply_curve_fitting(netflix_data)

        print("\nCurve fitting analizleri tamamlandı. Sonuçları 'graphics/curve_fitting' klasöründe bulabilirsiniz.")


if __name__ == "__main__":
    main()