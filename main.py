import os
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


def _pick_device():
    """Choose the best available compute device.

    Priority: CUDA (NVIDIA) > DirectML (any DX12 GPU on Windows, incl. Intel
    iGPU and AMD) > CPU. DirectML is enabled when `torch-directml` is
    installed; it ships wheels for Python 3.10–3.12. On Python 3.13+ the
    import will fail silently and we fall back to CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda"), "cuda"
    try:
        import torch_directml  # type: ignore
        if torch_directml.is_available():
            return torch_directml.device(), "directml"
    except Exception:
        pass
    return torch.device("cpu"), "cpu"


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
        # Lean UNet for 64x64 patches; ~50x fewer parameters than the
        # 64/128/256/512/1024 variant, trains in a fraction of the time on CPU
        # without losing accuracy at this resolution.
        self.enc1 = self.conv_block(1, 16)
        self.enc2 = self.conv_block(16, 32)
        self.enc3 = self.conv_block(32, 64)
        self.enc4 = self.conv_block(64, 96)
        self.bottleneck = self.conv_block(96, 128)
        self.dec4 = self.upconv_block(128, 96)
        self.dec3 = self.upconv_block(96, 64)
        self.dec2 = self.upconv_block(64, 32)
        self.dec1 = self.upconv_block(32, 16)
        self.final = nn.Conv2d(16, 1, 1)

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
        # Returns logits; apply sigmoid externally when needed.
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
        self.device, self.device_kind = _pick_device()
        self.model_nn.to(self.device)
        print(f"[device] using {self.device_kind} ({self.device})")
        # Use every CPU core when we are actually running on the CPU.
        if self.device_kind == "cpu":
            try:
                torch.set_num_threads(max(1, (os.cpu_count() or 1) - 1))
            except Exception:
                pass
        self._mask_dir = os.path.join("all", "manual1")
        self._image_dir = os.path.join("all", "images")

        # GUI elements
        self.load_image_btn = tk.Button(root, text="Załaduj obraz", command=self.load_image)
        self.load_image_btn.pack()

        self.load_mask_btn = tk.Button(
            root,
            text="Załaduj maskę ekspercką (manual1/)",
            command=self.load_mask,
        )
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
        initial = self._image_dir if os.path.isdir(self._image_dir) else os.getcwd()
        file_path = filedialog.askopenfilename(
            initialdir=initial,
            filetypes=[("Image files", "*.jpg *.JPG *.png *.tif")],
        )
        if not file_path:
            return
        bgr = cv2.imread(file_path, cv2.IMREAD_COLOR)
        if bgr is None:
            # Single-channel fallback (e.g. already-grey TIFF).
            self.image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        else:
            # Vessels show the highest contrast in the green channel of a
            # fundus image (blue is noisy, red saturates over the retina).
            green = bgr[:, :, 1]
            # Apply CLAHE here so every downstream method (baseline, ML, NN)
            # sees the same contrast-enhanced input.
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            self.image = clahe.apply(green)
        messagebox.showinfo("Sukces", "Obraz załadowany")

    def load_mask(self):
        initial = self._mask_dir if os.path.isdir(self._mask_dir) else os.getcwd()
        file_path = filedialog.askopenfilename(
            initialdir=initial,
            title="Wybierz maskę ekspercką z folderu manual1/",
            filetypes=[("Image files", "*.tif *.png *.jpg")],
        )
        if file_path:
            self.mask = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            self.mask = (self.mask > 128).astype(np.uint8)
            messagebox.showinfo("Sukces", "Maska załadowana")

    def process_image(self):
        if self.image is None:
            messagebox.showerror("Błąd", "Najpierw załaduj obraz")
            return

        # self.image is already CLAHE-enhanced in load_image.
        edges = cv2.Canny(self.image, 50, 150)

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
        if self.mask.shape != self.predicted_mask.shape:
            messagebox.showerror("Błąd", "Maska i predykcja mają różne rozmiary")
            return

        y_true = self.mask.flatten()
        y_pred = self.predicted_mask.flatten()
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        accuracy = accuracy_score(y_true, y_pred)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        mean_metric = (sensitivity + specificity) / 2

        result = (f"Accuracy: {accuracy:.4f}\nSensitivity: {sensitivity:.4f}\n"
                  f"Specificity: {specificity:.4f}\nŚrednia: {mean_metric:.4f}")
        messagebox.showinfo("Wyniki analizy", result)

    def prepare_training_data(self, image, mask, patch_size=5, samples_per_class=8000):
        h, w = image.shape
        c = patch_size // 2
        rng = np.random.default_rng(42)
        # Valid centre coordinates (full patch fits inside the image).
        ys, xs = np.mgrid[c:h - c, c:w - c]
        centres = mask[c:h - c, c:w - c]
        pos_idx = np.flatnonzero(centres.ravel() == 1)
        neg_idx = np.flatnonzero(centres.ravel() == 0)
        if pos_idx.size == 0 or neg_idx.size == 0:
            return np.empty((0, 8)), np.empty((0,), dtype=np.int64)
        n = min(samples_per_class, pos_idx.size, neg_idx.size)
        pos_sel = rng.choice(pos_idx, size=n, replace=False)
        neg_sel = rng.choice(neg_idx, size=n, replace=False)
        sel = np.concatenate([pos_sel, neg_sel])
        ys_sel = ys.ravel()[sel]
        xs_sel = xs.ravel()[sel]

        # Build the patches for the selected centres only.
        patches = np.stack([
            image[y - c:y + c + 1, x - c:x + c + 1]
            for y, x in zip(ys_sel, xs_sel)
        ])
        labels = np.concatenate([np.ones(n, dtype=np.int64),
                                 np.zeros(n, dtype=np.int64)])
        # Drop uniform patches.
        keep = patches.reshape(patches.shape[0], -1).var(axis=1) > 0
        feats = _compute_features_batch(patches[keep])
        return feats, labels[keep]

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
        patch_size = 5
        c = patch_size // 2
        self.predicted_mask = np.zeros((h, w), dtype=np.uint8)

        patches = _all_patches(self.image, patch_size)  # (N, P, P) view
        inner_h, inner_w = h - 2 * c, w - 2 * c
        # Chunk through patches so we don't allocate a giant float64 array.
        preds = np.zeros(patches.shape[0], dtype=np.uint8)
        chunk = 200_000
        for start in range(0, patches.shape[0], chunk):
            block = patches[start:start + chunk]
            feats = _compute_features_batch(block)
            preds[start:start + chunk] = self.classifier.predict(feats).astype(np.uint8)
        self.predicted_mask[c:c + inner_h, c:c + inner_w] = preds.reshape(inner_h, inner_w)

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

        X_train_t = torch.tensor(X_train).to(self.device)
        y_train_t = torch.tensor(y_train).to(self.device)
        X_test_t = torch.tensor(X_test).to(self.device)
        y_test_t = torch.tensor(y_test).to(self.device)

        dataset = TensorDataset(X_train_t, y_train_t)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

        # Vessel pixels are ~5–10% of the image. Half of the raw ratio is
        # enough to keep the model from collapsing to "background everywhere"
        # without over-predicting bright non-vessel regions.
        pos = float(y_train.sum())
        neg = float(y_train.size - pos)
        pw = max(0.5 * (neg / max(pos, 1.0)), 3.0)
        pos_weight = torch.tensor([pw], device=self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.Adam(self.model_nn.parameters(), lr=1e-3)

        epochs = 20
        self.model_nn.train()
        for epoch in range(epochs):
            for inputs, labels in dataloader:
                optimizer.zero_grad()
                outputs = self.model_nn(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        # Hold-out evaluation — batched to avoid materialising preds for the
        # whole test set in one shot.
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

        # Walk patches with stride; clamp the final patch flush to the edge so
        # we cover the whole image when (h, w) aren't multiples of stride.
        def positions(length):
            steps = list(range(0, max(length - patch_size, 0) + 1, stride))
            if not steps or steps[-1] + patch_size < length:
                steps.append(max(length - patch_size, 0))
            return steps

        with torch.no_grad():
            for i in positions(h):
                for j in positions(w):
                    patch = self.image[i:i + patch_size, j:j + patch_size]
                    if patch.shape != (patch_size, patch_size):
                        continue
                    t = (torch.tensor(patch, dtype=torch.float32) / 255.0
                         ).unsqueeze(0).unsqueeze(0).to(self.device)
                    prob = torch.sigmoid(self.model_nn(t)).squeeze().cpu().numpy()
                    prob_sum[i:i + patch_size, j:j + patch_size] += prob
                    count[i:i + patch_size, j:j + patch_size] += 1.0

        avg = np.divide(prob_sum, count, out=np.zeros_like(prob_sum), where=count > 0)
        self.predicted_mask = (avg > 0.5).astype(np.uint8)
        self.visualize()

if __name__ == "__main__":
    root = tk.Tk()
    app = RetinaVesselDetector(root)
    root.mainloop()