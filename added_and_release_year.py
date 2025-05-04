import pandas as pd
import matplotlib.pyplot as plt
import os


# Grafik klasörü
fig_dir = "graphics"
os.makedirs(fig_dir, exist_ok=True)

# Veriyi oku
df = pd.read_csv("data/netflix1.csv")

# Tarih dönüşümleri
df['date_added'] = pd.to_datetime(df['date_added'], errors='coerce')
df['year_added'] = df['date_added'].dt.year

# Release year ve added year için veri hazırlama
df = df.dropna(subset=['release_year', 'year_added'])
df['release_year'] = df['release_year'].astype(int)

# Netflix'e eklenme yılına göre içerik sayısı
added_counts = df['year_added'].value_counts().sort_index()

# İçeriklerin piyasaya çıkış yılına göre sayısı
release_counts = df['release_year'].value_counts().sort_index()

# Netflix'e eklenme yılına göre içerik sayısı - Film vs Dizi ayrımı
added_counts_by_type = df.groupby(['year_added', 'type']).size().unstack()

# Piyasaya çıkış yılına göre içerik sayısı - Film vs Dizi ayrımı
release_counts_by_type = df.groupby(['release_year', 'type']).size().unstack()

# En çok içerik eklenen yıllar
added_fig = os.path.join(fig_dir, "netflix_added_year_distribution_netflix.png")

if not os.path.exists(added_fig):
    plt.figure(figsize=(12, 6))

    # Netflix teması uygula
    plt.style.use('dark_background')

    # Veriyi çiz - çubuklar için Netflix kırmızısı
    ax = added_counts.plot(kind='bar', color='#E50914', edgecolor='black', width=0.8)

    # Başlık ve etiketler - beyaz renkle
    plt.title("Netflix'e Eklenme Yılına Göre İçerik Sayısı", fontsize=14, color='white')
    plt.xlabel("Eklenme Yılı", fontsize=12, color='white')
    plt.ylabel("İçerik Sayısı", fontsize=12, color='white')

    # Izgara çizgilerini ayarla
    plt.grid(axis='y', linestyle='--', alpha=0.3, color='#555555')

    # Çerçeveleri ayarla
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#333333')
    ax.spines['left'].set_color('#333333')

    # X ve Y ekseni değerlerini beyaz yap
    plt.tick_params(axis='both', colors='white')

    # Her bir çubuğun üstüne değeri ekle
    for i, v in enumerate(added_counts.values):
        ax.text(i, v + 20, str(v), ha='center', va='bottom', color='white', fontweight='bold')

    plt.tight_layout()
    plt.savefig(added_fig, facecolor='black', edgecolor='none')
    print(f"Netflix'e eklenme yılı grafiği kaydedildi: {added_fig}")
    plt.show()
else:
    print(f"Netflix'e eklenme yılı grafiği zaten mevcut: {added_fig}")

# 2000 sonrası içeriklerin piyasaya çıkış yılına göre sayısı
release_counts = df['release_year'].value_counts().sort_index()
release_counts = release_counts[release_counts.index >= 2000]

release_fig = os.path.join(fig_dir, "netflix_release_year_distribution_2000s.png")

if not os.path.exists(release_fig):
    plt.figure(figsize=(12, 6))
    plt.style.use('dark_background')

    # Çizgi grafiği çiz
    plt.plot(release_counts.index, release_counts.values, marker='o', color='#E50914', linewidth=2.5)

    # Etiketler ve başlık
    plt.title("Yıllara Göre Film ve Dizilerin Piyasaya Çıkış Yoğunluğu", fontsize=14, color='white')
    plt.xlabel("Yayın Yılı", fontsize=12, color='white')
    plt.ylabel("İçerik Sayısı", fontsize=12, color='white')

    # Sayı etiketleri
    for x, y in zip(release_counts.index, release_counts.values):
        plt.text(x, y + 20, str(y), ha='center', va='bottom', fontsize=9, color='white')

    # Stil ayarları
    plt.grid(axis='y', linestyle='--', alpha=0.3, color='#555555')
    plt.tick_params(axis='both', colors='white')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_color('#333333')
    plt.gca().spines['left'].set_color('#333333')

    plt.tight_layout()
    plt.savefig(release_fig, facecolor='black', edgecolor='none')
    print(f"Piyasaya çıkış yılı grafiği kaydedildi: {release_fig}")
    plt.show()
else:
    print(f"Piyasaya çıkış yılı grafiği zaten mevcut: {release_fig}")



# Film vs Dizi - Netflix'e eklenme yılı karşılaştırması
added_by_type_fig = os.path.join(fig_dir, "netflix_added_year_by_type_netflix.png")

if not os.path.exists(added_by_type_fig):
    plt.figure(figsize=(12, 6))

    # Netflix teması uygula
    plt.style.use('dark_background')

    # Renkleri tanımla (Ana Netflix kırmızısı ve biraz daha koyu ton)
    colors = ['#E50914', '#831010']

    # Veriyi çiz - çubukları yan yana
    added_counts_by_type.plot(kind='bar', color=colors, edgecolor='black', width=0.8, ax=plt.gca())

    # Başlık ve etiketler
    plt.title("Netflix'e Eklenme Yılına Göre Film ve Dizi Sayısı", fontsize=14, color='white')
    plt.xlabel("Eklenme Yılı", fontsize=12, color='white')
    plt.ylabel("İçerik Sayısı", fontsize=12, color='white')

    # Izgara çizgilerini ayarla
    plt.grid(axis='y', linestyle='--', alpha=0.3, color='#555555')

    # Çerçeveleri ayarla
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_color('#333333')
    plt.gca().spines['left'].set_color('#333333')

    # X ve Y ekseni değerlerini beyaz yap
    plt.tick_params(axis='both', colors='white')

    # Lejantın rengini ayarla
    plt.legend(facecolor='#222222', edgecolor='#444444', labelcolor='white')

    plt.tight_layout()
    plt.savefig(added_by_type_fig, facecolor='black', edgecolor='none')
    print(f"Film vs Dizi eklenme yılı grafiği kaydedildi: {added_by_type_fig}")
    plt.show()
else:
    print(f"Film vs Dizi eklenme yılı grafiği zaten mevcut: {added_by_type_fig}")



# Yıllara göre eklenme gecikmesindeki trend
# Eklenme yılı ile yayın yılı arasındaki fark
df['years_delay'] = df['year_added'] - df['release_year']

# Yıllara göre ortalama gecikme süresi
delay_by_added_year = df.groupby('year_added')['years_delay'].mean()


delay_trend_fig = os.path.join(fig_dir, "netflix_delay_trend_netflix.png")

if not os.path.exists(delay_trend_fig):
    plt.figure(figsize=(12, 6))

    # Netflix teması uygula
    plt.style.use('dark_background')

    # Çizgi grafiği çiz
    plt.plot(delay_by_added_year.index, delay_by_added_year.values, marker='o', linewidth=2.5, color='#E50914')

    # Başlık ve etiketler
    plt.title("Yıllara Göre Netflix'e Eklenme Gecikmesi Trendi", fontsize=14, color='white')
    plt.xlabel("Eklenme Yılı", fontsize=12, color='white')
    plt.ylabel("Ortalama Gecikme (Yıl)", fontsize=12, color='white')

    # Izgara çizgilerini ayarla
    plt.grid(linestyle='--', alpha=0.3, color='#555555')

    # Çerçeveleri ayarla
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_color('#333333')
    plt.gca().spines['left'].set_color('#333333')

    # X ve Y ekseni değerlerini beyaz yap
    plt.tick_params(axis='both', colors='white')

    plt.tight_layout()
    plt.savefig(delay_trend_fig, facecolor='black', edgecolor='none')
    print(f"Gecikme trendi grafiği kaydedildi: {delay_trend_fig}")
    plt.show()
else:
    print(f"Gecikme trendi grafiği zaten mevcut: {delay_trend_fig}")