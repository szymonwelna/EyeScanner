# Wykrywanie naczyń krwionośnych dna oka

## Opis projektu
Projekt polega na stworzeniu aplikacji desktopowej w Pythonie do automatycznego wykrywania naczyń krwionośnych na obrazach dna siatkówki oka. Aplikacja wykorzystuje techniki przetwarzania obrazu do klasyfikacji binarnej pikseli jako naczynia lub tło.

## Wymagania
- Python 3.12+
- Biblioteki: opencv-python, numpy, matplotlib, scikit-learn, imbalanced-learn, scikit-image, torch, torchvision, torchaudio, tkinter

## Instalacja
1. Skonfiguruj środowisko Python.
2. Zainstaluj wymagane pakiety: `pip install opencv-python numpy matplotlib scikit-learn imbalanced-learn scikit-image torch torchvision torchaudio`

## Uruchomienie
Uruchom `main.py` za pomocą Pythona. Aplikacja otworzy okno GUI.

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