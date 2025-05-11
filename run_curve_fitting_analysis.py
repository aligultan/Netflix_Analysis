import os
import time
import subprocess
import sys


def create_directories():
    """Gerekli dizinleri oluştur"""
    if not os.path.exists('graphics'):
        os.makedirs('graphics')

    if not os.path.exists('graphics/curve_fitting'):
        os.makedirs('graphics/curve_fitting')


def run_python_script(script_name):
    """Python scriptini çalıştır ve çıktıları yönlendir"""
    print(f"\n{'=' * 50}")
    print(f"Çalıştırılıyor: {script_name}")
    print(f"{'=' * 50}\n")

    try:
        # Python yorumlayıcısını doğru şekilde al
        python_executable = sys.executable

        # Scripti çalıştır
        process = subprocess.Popen(
            [python_executable, script_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )

        # Çıktıları gerçek zamanlı olarak oku ve göster
        while True:
            output = process.stdout.readline()
            error = process.stderr.readline()

            if output == '' and error == '' and process.poll() is not None:
                break

            if output:
                print(output.strip())

            if error:
                print(f"HATA: {error.strip()}", file=sys.stderr)

        return_code = process.poll()

        if return_code == 0:
            print(f"\n{script_name} başarıyla tamamlandı.\n")
        else:
            print(f"\n{script_name} çalıştırılırken hata oluştu (kod: {return_code}).\n")

        return return_code == 0

    except Exception as e:
        print(f"Script çalıştırılırken bir istisna oluştu: {e}")
        return False


def main():
    """Ana işlev - tüm curve fitting analizlerini çalıştır"""
    start_time = time.time()

    print("Netflix Curve Fitting Analizi Başlıyor...")
    print("Versiyon: 1.0.0")
    print(f"Tarih: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Gerekli dizinleri oluştur
    create_directories()

    # Analiz scriptlerini çalıştır
    scripts = [
        "curve_fitting.py",
        "content_type_curve_fitting.py",
        "seasonal_curve_fitting.py"
    ]

    results = {}

    for script in scripts:
        results[script] = run_python_script(script)

    # Sonuçları özetleme
    print("\n")
    print("=" * 50)
    print("ANALİZ SONUÇLARI")
    print("=" * 50)

    all_success = True

    for script, success in results.items():
        status = "✓ Başarılı" if success else "✗ Başarısız"
        print(f"{script}: {status}")

        if not success:
            all_success = False

    # Toplam çalışma süresini hesaplama
    end_time = time.time()
    duration = end_time - start_time

    print("\n")
    print(f"Toplam çalışma süresi: {duration:.2f} saniye")

    if all_success:
        print("\nTüm analizler başarıyla tamamlandı!")
        print("Sonuçlar 'graphics/curve_fitting' klasöründe bulunabilir.")
    else:
        print("\nBazı analizler tamamlanamadı. Lütfen yukarıdaki hata mesajlarını kontrol edin.")


if __name__ == "__main__":
    main()