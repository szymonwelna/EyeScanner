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
from numpy.lib.stride_tricks import sliding_window_view
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

def _log(msg):
    print(msg, flush=True)

def _compute_features_batch(patches):
    """Vectorized features per patch: variance + 7 Hu moments.

    patches: (N, P, P) float array.  Returns (N, 8) feature array.
    """
    p = patches.astype(np.float64)
    n, ph, pw = p.shape
    variance = p.var(axis=(1, 2))

    ys, xs = np.mgrid[0:ph, 0:pw].astype(np.float64)
    m00 = p.sum(axis=(1, 2))
    safe = m00 != 0
    inv = np.zeros_like(m00)
    inv[safe] = 1.0 / m00[safe]
    xbar = (p * xs).sum(axis=(1, 2)) * inv
    ybar = (p * ys).sum(axis=(1, 2)) * inv

    dx = xs[None, :, :] - xbar[:, None, None]
    dy = ys[None, :, :] - ybar[:, None, None]

    def mu(i, j):
        return (p * (dx ** i) * (dy ** j)).sum(axis=(1, 2))

    def nu(i, j):
        m = mu(i, j)
        out = np.zeros_like(m)
        denom = np.ones_like(m00)
        np.power(m00, (i + j) / 2.0 + 1.0, out=denom, where=safe)
        np.divide(m, denom, out=out, where=safe)
        return out

    n20, n02 = nu(2, 0), nu(0, 2)
    n11 = nu(1, 1)
    n30, n03 = nu(3, 0), nu(0, 3)
    n21, n12 = nu(2, 1), nu(1, 2)

    h1 = n20 + n02
    h2 = (n20 - n02) ** 2 + 4 * n11 ** 2
    h3 = (n30 - 3 * n12) ** 2 + (3 * n21 - n03) ** 2
    h4 = (n30 + n12) ** 2 + (n21 + n03) ** 2
    h5 = ((n30 - 3 * n12) * (n30 + n12) *
          ((n30 + n12) ** 2 - 3 * (n21 + n03) ** 2) +
          (3 * n21 - n03) * (n21 + n03) *
          (3 * (n30 + n12) ** 2 - (n21 + n03) ** 2))
    h6 = ((n20 - n02) * ((n30 + n12) ** 2 - (n21 + n03) ** 2) +
          4 * n11 * (n30 + n12) * (n21 + n03))
    h7 = ((3 * n21 - n03) * (n30 + n12) *
          ((n30 + n12) ** 2 - 3 * (n21 + n03) ** 2) -
          (n30 - 3 * n12) * (n21 + n03) *
          (3 * (n30 + n12) ** 2 - (n21 + n03) ** 2))

    return np.stack([variance, h1, h2, h3, h4, h5, h6, h7], axis=1)


def _all_patches(image, patch_size):
    """All (h-p+1)*(w-p+1) patches as a (N, P, P) view of the image."""
    return sliding_window_view(image, (patch_size, patch_size)).reshape(
        -1, patch_size, patch_size)

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
        return self.final(d1)

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

        self.ml_accuracy = tk.IntVar(value=1)
        self.ml_accuracy_scale = tk.Scale(root, label="Dokładność ML (1=wysoka, 5=niska)", from_=1, to=5, orient=tk.HORIZONTAL, variable=self.ml_accuracy, length=260)
        self.ml_accuracy_scale.pack()

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
            _log(f"Obraz załadowany: {file_path}")
            _log(f"Rozmiar obrazu: {self.image.shape}")
            messagebox.showinfo("Sukces", "Obraz załadowany")

    def load_mask(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.tif")])
        if file_path:
            self.mask = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            self.mask = (self.mask > 128).astype(np.uint8)
            _log(f"Maska załadowana: {file_path}")
            _log(f"Rozmiar maski: {self.mask.shape}")
            messagebox.showinfo("Sukces", "Maska załadowana")

    def process_image(self):
        if self.image is None:
            messagebox.showerror("Błąd", "Najpierw załaduj obraz")
            return

        _log("Rozpoczynam przetwarzanie obrazu (baseline)")
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(self.image)
        _log("Zastosowano CLAHE")
        edges = cv2.Canny(enhanced, 50, 150)
        _log("Wykonano detekcję krawędzi Canny")
        kernel = np.ones((3,3), np.uint8)
        self.predicted_mask = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        self.predicted_mask = (self.predicted_mask > 0).astype(np.uint8)
        _log("Zastosowano operacje morfologiczne")
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
        _log("Wynik został zwizualizowany")

    def analyze_results(self):
        if self.mask is None or self.predicted_mask is None:
            messagebox.showerror("Błąd", "Załaduj obraz, maskę i przetwórz")
            return
        if self.mask.shape != self.predicted_mask.shape:
            messagebox.showerror("Błąd", "Maska i predykcja mają różne rozmiary")
            return

        _log("Rozpoczynam analizę wyników")
        y_true = self.mask.flatten()
        y_pred = self.predicted_mask.flatten()
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        accuracy = accuracy_score(y_true, y_pred)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        mean_metric = (sensitivity + specificity) / 2
        _log(f"Accuracy={accuracy:.4f}, Sensitivity={sensitivity:.4f}, Specificity={specificity:.4f}, Średnia={mean_metric:.4f}")

        result = (f"Accuracy: {accuracy:.4f}\nSensitivity: {sensitivity:.4f}\n"
                  f"Specificity: {specificity:.4f}\nŚrednia: {mean_metric:.4f}")
        messagebox.showinfo("Wyniki analizy", result)

    def prepare_training_data(self, image, mask, patch_size=5, samples_per_class=8000):
        h, w = image.shape
        c = patch_size // 2
        rng = np.random.default_rng(42)
        ys, xs = np.mgrid[c:h - c, c:w - c]
        centres = mask[c:h - c, c:w - c]
        pos_idx = np.flatnonzero(centres.ravel() == 1)
        neg_idx = np.flatnonzero(centres.ravel() == 0)
        if pos_idx.size == 0 or neg_idx.size == 0:
            _log("Brak danych pozytywnych lub negatywnych do przygotowania danych treningowych")
            return np.empty((0, 8)), np.empty((0,), dtype=np.int64)
        n = min(samples_per_class, pos_idx.size, neg_idx.size)
        pos_sel = rng.choice(pos_idx, size=n, replace=False)
        neg_sel = rng.choice(neg_idx, size=n, replace=False)
        sel = np.concatenate([pos_sel, neg_sel])
        ys_sel = ys.ravel()[sel]
        xs_sel = xs.ravel()[sel]

        patches = np.stack([
            image[y - c:y + c + 1, x - c:x + c + 1]
            for y, x in zip(ys_sel, xs_sel)
        ])
        labels = np.concatenate([np.ones(n, dtype=np.int64),
                                 np.zeros(n, dtype=np.int64)])
        keep = patches.reshape(patches.shape[0], -1).var(axis=1) > 0
        _log(f"Wybrano {keep.sum()} patchy treningowe z {patches.shape[0]} wycinków")
        feats = _compute_features_batch(patches[keep])
        return feats, labels[keep]

    def train_ml(self):
        if self.image is None or self.mask is None:
            messagebox.showerror("Błąd", "Załaduj obraz i maskę")
            return

        _log("Rozpoczynam przygotowanie danych ML")
        self.features, self.labels = self.prepare_training_data(self.image, self.mask)
        _log(f"Przygotowano dane: {self.features.shape[0]} próbek, {self.features.shape[1]} cech")

        rus = RandomUnderSampler(random_state=42)
        X_res, y_res = rus.fit_resample(self.features, self.labels)
        _log(f"Undersampling: {X_res.shape[0]} próbek po zrównoważeniu")

        X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)
        _log(f"Podział train/test: {X_train.shape[0]}/{X_test.shape[0]}")

        self.classifier = SVC(kernel='rbf', C=1.0, random_state=42)
        _log("Rozpoczynam trening SVM")
        self.classifier.fit(X_train, y_train)

        y_pred = self.classifier.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        _log(f"Trening SVM zakończony, accuracy hold-out: {acc:.4f}")
        messagebox.showinfo("Trening zakończony", f"Accuracy na hold-out: {acc:.4f}")

    def predict_ml(self):
        if self.classifier is None or self.image is None:
            messagebox.showerror("Błąd", "Najpierw trenować klasyfikator")
            return

        step = self.ml_accuracy.get()
        h, w = self.image.shape
        patch_size = 5
        c = patch_size // 2
        self.predicted_mask = np.zeros((h, w), dtype=np.uint8)

        positions_h = list(range(c, h - c, step))
        positions_w = list(range(c, w - c, step))
        total = len(positions_h) * len(positions_w)
        _log(f"Rozpoczynam predykcję ML dla {total} pozycji przy kroku {step}")

        idx = 0
        for y in positions_h:
            row_patches = [self.image[y - c:y + c + 1, x - c:x + c + 1] for x in positions_w]
            feats = _compute_features_batch(np.stack(row_patches))
            preds = self.classifier.predict(feats).astype(np.uint8)
            for x, pred in zip(positions_w, preds):
                self.predicted_mask[y - c:y + c + 1, x - c:x + c + 1] = pred
            idx += len(positions_w)
            if idx % 1000 == 0:
                _log(f"Predykcja ML: przetworzono {idx}/{total} pozycji")

        _log("Predykcja ML zakończona")
        self.visualize()

    def train_nn(self):
        if self.image is None or self.mask is None:
            messagebox.showerror("Błąd", "Załaduj obraz i maskę")
            return

        h, w = self.image.shape
        patch_size = 64
        stride = patch_size // 2
        X, y = [], []
        for i in range(0, h - patch_size + 1, stride):
            for j in range(0, w - patch_size + 1, stride):
                X.append(self.image[i:i + patch_size, j:j + patch_size])
                y.append(self.mask[i:i + patch_size, j:j + patch_size])

        X = np.array(X).reshape(-1, 1, patch_size, patch_size).astype(np.float32) / 255.0
        y = np.array(y).reshape(-1, 1, patch_size, patch_size).astype(np.float32)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
        _log(f"Przygotowano dane NN: {X_train.shape[0]} próbek treningowych, {X_test.shape[0]} próbek testowych")

        X_train_t = torch.tensor(X_train).to(self.device)
        y_train_t = torch.tensor(y_train).to(self.device)
        X_test_t = torch.tensor(X_test).to(self.device)
        y_test_t = torch.tensor(y_test).to(self.device)

        dataset = TensorDataset(X_train_t, y_train_t)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

        pos = float(y_train.sum())
        neg = float(y_train.size - pos)
        pos_weight = torch.tensor([neg / max(pos, 1.0)], device=self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.Adam(self.model_nn.parameters(), lr=1e-3)

        epochs = 10
        self.model_nn.train()
        _log("Rozpoczynam trening NN")
        batch_count = len(dataloader)
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_idx, (inputs, labels) in enumerate(dataloader, start=1):
                optimizer.zero_grad()
                outputs = self.model_nn(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                if batch_idx % 50 == 0:
                    _log(f"Epoch {epoch + 1}/{epochs}, batch {batch_idx}/{batch_count}: loss={loss.item():.4f}")
            _log(f"Epoch {epoch + 1}/{epochs}: loss={epoch_loss:.4f}")

        self.model_nn.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for k in range(0, X_test_t.shape[0], 8):
                logits = self.model_nn(X_test_t[k:k + 8])
                preds = (torch.sigmoid(logits) > 0.5).float()
                correct += (preds == y_test_t[k:k + 8]).float().sum().item()
                total += preds.numel()
        acc = correct / max(total, 1)
        _log(f"Trening NN zakończony, accuracy hold-out: {acc:.4f}")
        messagebox.showinfo("Trening NN zakończony", f"Accuracy na hold-out: {acc:.4f}")

    def predict_nn(self):
        if self.model_nn is None or self.image is None:
            messagebox.showerror("Błąd", "Najpierw trenować NN")
            return

        h, w = self.image.shape
        patch_size = 64
        stride = patch_size // 2
        self.model_nn.eval()
        prob_sum = np.zeros((h, w), dtype=np.float32)
        count = np.zeros((h, w), dtype=np.float32)

        def positions(length):
            steps = list(range(0, max(length - patch_size, 0) + 1, stride))
            if not steps or steps[-1] + patch_size < length:
                steps.append(max(length - patch_size, 0))
            return steps

        ys = positions(h)
        xs = positions(w)
        _log(f"Rozpoczynam predykcję NN dla {len(ys) * len(xs)} patchy")
        with torch.no_grad():
            count_log = 0
            for i in ys:
                for j in xs:
                    patch = self.image[i:i + patch_size, j:j + patch_size]
                    if patch.shape != (patch_size, patch_size):
                        continue
                    t = (torch.tensor(patch, dtype=torch.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(self.device)
                    prob = torch.sigmoid(self.model_nn(t)).squeeze().cpu().numpy()
                    prob_sum[i:i + patch_size, j:j + patch_size] += prob
                    count[i:i + patch_size, j:j + patch_size] += 1.0
                    count_log += 1
                    if count_log % 20 == 0:
                        _log(f"Predykcja NN: przetworzono {count_log} patchy")

        avg = np.divide(prob_sum, count, out=np.zeros_like(prob_sum), where=count > 0)
        self.predicted_mask = (avg > 0.5).astype(np.uint8)
        _log("Predykcja NN zakończona")
        self.visualize()

if __name__ == "__main__":
    root = tk.Tk()
    app = RetinaVesselDetector(root)
    root.mainloop()