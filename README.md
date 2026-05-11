# Wykrywanie naczyń krwionośnych dna oka

## Opis projektu
Projekt polega na stworzeniu aplikacji desktopowej w Pythonie do automatycznego wykrywania naczyń krwionośnych na obrazach dna siatkówki oka. Aplikacja wykorzystuje techniki przetwarzania obrazu do klasyfikacji binarnej pikseli jako naczynia lub tło.

## Wymagania
- Python 3.12+
- Biblioteki: opencv-python, numpy, matplotlib, scikit-learn, imbalanced-learn, scikit-image, torch, torchvision, torchaudio, tkinter

## Instalacja
1. Skonfiguruj środowisko Python (3.10–3.13 zalecane).
2. Zainstaluj wymagane pakiety:
   ```
   pip install -r requirements.txt
   ```

### Opcjonalne wsparcie GPU (DirectML)
Aplikacja automatycznie wykrywa GPU przy starcie. Kolejność priorytetów:
**CUDA (NVIDIA) → DirectML (DX12: Intel iGPU / AMD / NVIDIA) → CPU**.

Aby włączyć DirectML (działa na każdej karcie z DirectX 12, w tym Intel UHD,
Iris Xe, AMD Radeon, NVIDIA):

```
pip install torch-directml
```

Wymagania:
- Windows 10/11
- **Python 3.10, 3.11 lub 3.12** (na nowsze wersje brak gotowych wheeli)
- Sterownik GPU z obsługą DirectX 12

Po instalacji uruchom `python main.py` — w konsoli pojawi się
`[device] using directml (...)`. Jeśli `torch-directml` nie jest
zainstalowane, aplikacja po cichu wraca do wielowątkowego CPU, więc kod
działa wszędzie bez zmian.

Aby wymusić CPU mimo zainstalowanego DirectML, odinstaluj pakiet
(`pip uninstall torch-directml`) lub ustaw `CUDA_VISIBLE_DEVICES=""` przed
uruchomieniem.

## Uruchomienie
Uruchom `main.py` za pomocą Pythona. Aplikacja otworzy okno GUI. Po starcie:
1. **Załaduj obraz** – wybierz plik z `all/images/`.
2. **Załaduj maskę ekspercką (manual1/)** – wybierz plik z `all/manual1/`
   (to maska naczyń, *nie* maska FOV z folderu `all/mask/`).
3. **Przetwórz obraz (baseline)** – CLAHE jest już zastosowane w kroku 1,
   więc ten przycisk uruchamia tylko Canny + morfologię.
4. **Trenuj klasyfikator ML** → **Predykcja z ML**.
5. **Trenuj sieć neuronową** → **Predykcja z NN**.
6. **Analizuj wyniki** – metryki względem maski eksperckiej.

## Funkcjonalności
- Ładowanie obrazu dna oka.
- Ładowanie maski ground truth (opcjonalnie dla analizy).
- Przetwarzanie obrazu: wstępne (CLAHE), wykrywanie (Canny), końcowe (morfologia) - baseline.
- Ekstrakcja cech z wycinków (wariancja, momenty Hu).
- Trening klasyfikatora ML (SVM) z undersamplingiem.
- Predykcja z ML.
- Trening głębokiej sieci neuronowej (UNet) na patches.
- Predykcja z NN.
- Wizualizacja wyników.
- Analiza statystyczna: accuracy, sensitivity, specificity, średnia.

## Metody
- Wstępne przetwarzanie: Normalizacja histogramu (CLAHE).
- Wykrywanie naczyń (baseline): Filtr krawędzi Canny.
- Końcowe: Operacje morfologiczne do poprawy maski.
- ML: Ekstrakcja cech z 5x5 px wycinków (filtruje jednolite patches o wariancji <=0), cechy: wariancja, momenty Hu; SVM z RBF, undersampling dla niezrównoważonych danych.
- NN: Głęboka sieć UNet (PyTorch), trening na 64x64 patches, predykcja segmentacyjna.

## Ocena
Aplikacja oblicza metryki na podstawie porównania z maską ekspercką.

## Optymalizacje
- Filtrowanie jednolitych patches (wariancja <=0) w ekstrakcji cech ML, aby uniknąć błędów i poprawić jakość danych treningowych.
- Ignorowanie ostrzeżeń RuntimeWarning z skimage dla stabilności.

## Rozszerzenia
Dla wyższych ocen: sieci neuronowe zamiast SVM.