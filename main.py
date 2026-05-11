import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from skimage import measure
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="skimage.measure")

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.enc1 = self.conv_block(1, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        self.bottleneck = self.conv_block(512, 1024)
        self.dec4 = self.upconv_block(1024, 512)
        self.dec3 = self.upconv_block(512, 256)
        self.dec2 = self.upconv_block(256, 128)
        self.dec1 = self.upconv_block(128, 64)
        self.final = nn.Conv2d(64, 1, 1)

    def conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.ReLU()
        )

    def upconv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, 2, stride=2),
            nn.ReLU()
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(nn.MaxPool2d(2)(e1))
        e3 = self.enc3(nn.MaxPool2d(2)(e2))
        e4 = self.enc4(nn.MaxPool2d(2)(e3))
        b = self.bottleneck(nn.MaxPool2d(2)(e4))
        d4 = self.dec4(b) + e4
        d3 = self.dec3(d4) + e3
        d2 = self.dec2(d3) + e2
        d1 = self.dec1(d2) + e1
        return torch.sigmoid(self.final(d1))

class RetinaVesselDetector:
    def __init__(self, root):
        self.root = root
        self.root.title("Wykrywanie naczyń dna oka")
        self.image = None
        self.mask = None
        self.predicted_mask = None
        self.classifier = None
        self.features = None
        self.labels = None
        self.model_nn = UNet()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_nn.to(self.device)

        # GUI elements
        self.load_image_btn = tk.Button(root, text="Załaduj obraz", command=self.load_image)
        self.load_image_btn.pack()

        self.load_mask_btn = tk.Button(root, text="Załaduj maskę ground truth", command=self.load_mask)
        self.load_mask_btn.pack()

        self.process_btn = tk.Button(root, text="Przetwórz obraz (baseline)", command=self.process_image)
        self.process_btn.pack()

        self.train_ml_btn = tk.Button(root, text="Trenuj klasyfikator ML", command=self.train_ml)
        self.train_ml_btn.pack()

        self.predict_ml_btn = tk.Button(root, text="Predykcja z ML", command=self.predict_ml)
        self.predict_ml_btn.pack()

        self.train_nn_btn = tk.Button(root, text="Trenuj sieć neuronową", command=self.train_nn)
        self.train_nn_btn.pack()

        self.predict_nn_btn = tk.Button(root, text="Predykcja z NN", command=self.predict_nn)
        self.predict_nn_btn.pack()

        self.analyze_btn = tk.Button(root, text="Analizuj wyniki", command=self.analyze_results)
        self.analyze_btn.pack()

        self.canvas = None

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.tif")])
        if file_path:
            self.image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            messagebox.showinfo("Sukces", "Obraz załadowany")

    def load_mask(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.tif")])
        if file_path:
            self.mask = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            self.mask = (self.mask > 128).astype(np.uint8)  # Binarize
            messagebox.showinfo("Sukces", "Maska załadowana")

    def process_image(self):
        if self.image is None:
            messagebox.showerror("Błąd", "Najpierw załaduj obraz")
            return

        # Wstępne przetwarzanie: CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(self.image)

        # Wykrywanie krawędzi: Filtr Frangi (prosta implementacja)
        # Dla uproszczenia użyjemy Canny jako baseline
        edges = cv2.Canny(enhanced, 50, 150)

        # Końcowe przetwarzanie: Morfologia
        kernel = np.ones((3,3), np.uint8)
        self.predicted_mask = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        self.predicted_mask = (self.predicted_mask > 0).astype(np.uint8)

        # Wizualizacja
        self.visualize()

    def visualize(self):
        if self.canvas:
            self.canvas.get_tk_widget().destroy()

        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].imshow(self.image, cmap='gray')
        axs[0].set_title('Oryginalny obraz')
        axs[1].imshow(self.predicted_mask, cmap='gray')
        axs[1].set_title('Wykryte naczynia')
        if self.mask is not None:
            overlay = cv2.addWeighted(self.image, 0.7, self.predicted_mask*255, 0.3, 0)
            axs[2].imshow(overlay, cmap='gray')
            axs[2].set_title('Nakładka')
        else:
            axs[2].axis('off')

        self.canvas = FigureCanvasTkAgg(fig, master=self.root)
        self.canvas.get_tk_widget().pack()
        self.canvas.draw()

    def analyze_results(self):
        if self.mask is None or self.predicted_mask is None:
            messagebox.showerror("Błąd", "Załaduj obraz, maskę i przetwórz")
            return

        # Oblicz metryki
        tn, fp, fn, tp = confusion_matrix(self.mask.flatten(), self.predicted_mask.flatten()).ravel()
        accuracy = accuracy_score(self.mask.flatten(), self.predicted_mask.flatten())
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        mean_metric = (sensitivity + specificity) / 2

        result = f"Accuracy: {accuracy:.4f}\nSensitivity: {sensitivity:.4f}\nSpecificity: {specificity:.4f}\nŚrednia: {mean_metric:.4f}"
        messagebox.showinfo("Wyniki analizy", result)

    def extract_features(self, patch):
        # Ekstrakcja cech: wariancja, momenty Hu
        variance = np.var(patch)
        moments = measure.moments_central(patch, order=3)
        if moments[0, 0] == 0:
            hu_moments = np.zeros(7)  # Jeśli jednolity patch, ustaw na 0
        else:
            hu_moments = measure.moments_hu(moments)
        features = [variance] + list(hu_moments)
        return features

    def prepare_training_data(self, image, mask, patch_size=5):
        features = []
        labels = []
        h, w = image.shape
        for i in range(patch_size//2, h - patch_size//2):
            for j in range(patch_size//2, w - patch_size//2):
                patch = image[i-patch_size//2:i+patch_size//2+1, j-patch_size//2:j+patch_size//2+1]
                if np.var(patch) > 0:  # Filtrowanie jednolitych patches
                    feat = self.extract_features(patch)
                    label = mask[i, j]
                    features.append(feat)
                    labels.append(label)
        return np.array(features), np.array(labels)

    def train_ml(self):
        if self.image is None or self.mask is None:
            messagebox.showerror("Błąd", "Załaduj obraz i maskę")
            return

        # Przygotuj dane treningowe
        self.features, self.labels = self.prepare_training_data(self.image, self.mask)

        # Undersampling dla niezrównoważonych danych
        rus = RandomUnderSampler(random_state=42)
        X_res, y_res = rus.fit_resample(self.features, self.labels)

        # Podziel na train/test
        X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

        # Trenuj SVM
        self.classifier = SVC(kernel='rbf', C=1.0, random_state=42)
        self.classifier.fit(X_train, y_train)

        # Test na hold-out
        y_pred = self.classifier.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        messagebox.showinfo("Trening zakończony", f"Accuracy na hold-out: {acc:.4f}")

    def predict_ml(self):
        if self.classifier is None or self.image is None:
            messagebox.showerror("Błąd", "Najpierw trenować klasyfikator")
            return

        h, w = self.image.shape
        self.predicted_mask = np.zeros((h, w), dtype=np.uint8)
        patch_size = 5
        for i in range(patch_size//2, h - patch_size//2):
            for j in range(patch_size//2, w - patch_size//2):
                patch = self.image[i-patch_size//2:i+patch_size//2+1, j-patch_size//2:j+patch_size//2+1]
                feat = self.extract_features(patch)
                pred = self.classifier.predict([feat])[0]
                self.predicted_mask[i, j] = pred

        # Wizualizacja
        self.visualize()

    def train_nn(self):
        if self.image is None or self.mask is None:
            messagebox.showerror("Błąd", "Załaduj obraz i maskę")
            return

        # Przygotuj dane: podziel obraz na patches
        h, w = self.image.shape
        patch_size = 64  # Dla NN użyj większych patches
        X = []
        y = []
        for i in range(0, h - patch_size, patch_size//2):
            for j in range(0, w - patch_size, patch_size//2):
                patch_img = self.image[i:i+patch_size, j:j+patch_size]
                patch_mask = self.mask[i:i+patch_size, j:j+patch_size]
                X.append(patch_img)
                y.append(patch_mask)

        X = np.array(X).reshape(-1, 1, patch_size, patch_size)
        y = np.array(y).reshape(-1, 1, patch_size, patch_size)

        # Podziel na train/test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Konwertuj na tensory
        X_train = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_train = torch.tensor(y_train, dtype=torch.float32).to(self.device)
        X_test = torch.tensor(X_test, dtype=torch.float32).to(self.device)
        y_test = torch.tensor(y_test, dtype=torch.float32).to(self.device)

        dataset = TensorDataset(X_train, y_train)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model_nn.parameters(), lr=0.001)

        # Trening
        epochs = 10
        for epoch in range(epochs):
            for inputs, labels in dataloader:
                optimizer.zero_grad()
                outputs = self.model_nn(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        # Test na hold-out
        with torch.no_grad():
            preds = self.model_nn(X_test)
            preds = (preds > 0.5).float()
            acc = (preds == y_test).float().mean()
            messagebox.showinfo("Trening NN zakończony", f"Accuracy na hold-out: {acc:.4f}")

    def predict_nn(self):
        if self.model_nn is None or self.image is None:
            messagebox.showerror("Błąd", "Najpierw trenować NN")
            return

        h, w = self.image.shape
        patch_size = 64
        self.predicted_mask = np.zeros((h, w), dtype=np.uint8)

        for i in range(0, h - patch_size, patch_size//2):
            for j in range(0, w - patch_size, patch_size//2):
                patch = self.image[i:i+patch_size, j:j+patch_size]
                patch_tensor = torch.tensor(patch, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    pred = self.model_nn(patch_tensor)
                    pred = (pred > 0.5).float().squeeze().cpu().numpy()
                self.predicted_mask[i:i+patch_size, j:j+patch_size] = pred

        # Wizualizacja
        self.visualize()

if __name__ == "__main__":
    root = tk.Tk()
    app = RetinaVesselDetector(root)
    root.mainloop()