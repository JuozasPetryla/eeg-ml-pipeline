import os
import warnings
import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
import joblib
from matplotlib.patches import Patch

from pathlib import Path
from scipy.signal import welch
from scipy.integrate import trapezoid
from scipy.stats import skew, kurtosis

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
warnings.filterwarnings("ignore")

#CASSETTE_PATH = "sleep-cassette"

PLOTS_FOLDER = Path("/Users/edvinaskucys/Library/CloudStorage/GoogleDrive-edvinas.kuc@gmail.com/My Drive/Uni/Produkto vystymo projektas/models/sleep_plots")
os.makedirs(PLOTS_FOLDER, exist_ok=True)

DESC_TO_STAGE = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,
    "Sleep stage R": 4,
}

BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta":  (13.0, 30.0),
}

STAGE_NAMES = {
    0: "Awake",
    1: "N1",
    2: "N2",
    3: "N3",
    4: "REM"
}

stage_labels_lt = {
    0: "Budrumas",
    1: "Lengvas miegas N1",
    2: "Lengvas miegas N2",
    3: "Gilus miegas N3",
    4: "REM miegas",
}

stage_colors_lt = {
    "Budrumas":          "#FF6347",
    "Lengvas miegas N1": "#FFD700",
    "Lengvas miegas N2": "#87CEFA",
    "Gilus miegas N3":   "#1E90FF",
    "REM miegas":        "#32CD32",
}


def extract_band_powers(epoch, sfreq):
    """Extract per-channel band-power features from one epoch (channels x samples)."""
    features = []
    for ch in epoch:
        f, psd = welch(ch, sfreq, nperseg=min(1024, len(ch)))
        for band_name, (fmin, fmax) in BANDS.items():
            mask = (f >= fmin) & (f < fmax)
            features.append(trapezoid(psd[mask], f[mask]))
    return features

#Loading model and scaler
scaler = joblib.load("scaler.pkl")
model = joblib.load("model.pkl");

def load_new_subject(psg_path: str):
    """
    Load a single new PSG EDF file and extract features for prediction.
    No hypnogram needed — returns X_scaled, time_hours, and raw_eeg for plotting.

    Args:
        psg_path: path to the PSG .edf file

    Returns:
        X_scaled: feature array ready for model.predict()
        time_hours: timestamp for each epoch (for plotting)
        raw_eeg: MNE Raw object with only EEG channels
    """
    psg_path = Path(psg_path)
    print(f"Loading {psg_path.name}...")

    try:
        raw = mne.io.read_raw_edf(psg_path, preload=True, verbose=False)
        # Keep only EEG channels
        raw_eeg = raw.copy().pick_types(eeg=True, meg=False, stim=False)
    except Exception as e:
        raise RuntimeError(f"Failed to load {psg_path}: {e}")

    sfreq = raw_eeg.info["sfreq"]
    epoch_len = 30.0
    epoch_samples = int(epoch_len * sfreq)

    data = raw_eeg.get_data()  # shape: (n_channels, n_samples) — EEG only
    n_epochs = data.shape[1] // epoch_samples

    print(f"  EEG channels: {len(raw_eeg.ch_names)}, Sfreq: {sfreq} Hz")
    print(f"  Total epochs (30s): {n_epochs}")

    X = []
    for i in range(n_epochs):
        start = i * epoch_samples
        end   = start + epoch_samples
        epoch = data[:, start:end]  # (n_channels, epoch_samples)
        X.append(extract_band_powers(epoch, sfreq))

    X = np.array(X)
    X_scaled = scaler.transform(X)  # use the scaler fitted on training data
    time_hours = np.arange(n_epochs) / 120

    print(f"  Feature shape: {X.shape}")
    return X_scaled, time_hours, raw_eeg


# additional code to load uploaded pdg and predict sleep stages
X_new, time_hours, raw_eeg = load_new_subject(Path("/Users/edvinaskucys/Library/CloudStorage/GoogleDrive-edvinas.kuc@gmail.com/My Drive/Uni/Produkto vystymo projektas/physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/SC4191E0-PSG.edf"))
y = model.predict(X_new)  # use test set for display/plots
total = len(y)
# sleep_eff   = np.sum(y > 1) / total
# transitions = sum(y[i] != y[i-1] for i in range(1, total))
# sleep_latency = 110
# rem_latency   = 132
# waso          = 0

stage_percentages = {
    STAGE_NAMES.get(s, str(s)): np.sum(y == s) / total
    for s in np.unique(y)
}


# Scatter hypnogram
plt.figure(figsize=(15, 4))
for code, label in stage_labels_lt.items():
    idx = [i for i, s in enumerate(y) if s == code]
    if idx:
        plt.scatter(time_hours[idx], [label] * len(idx),
                    color=stage_colors_lt[label], s=20, label=label)
plt.xlabel("Laikas (valandos)")
plt.ylabel("Miego stadija")
plt.title("Hipnograma (scatter plot)")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{PLOTS_FOLDER}/hypnogram_scatter.png")
plt.close()


y_idx = np.array([list(stage_labels_lt.keys()).index(s)
                  if s in stage_labels_lt else 0 for s in y])
plt.figure(figsize=(15, 4))
cmap = mcolors.ListedColormap([stage_colors_lt[label]
                                for label in stage_labels_lt.values()])
plt.imshow([y_idx], aspect='auto', cmap=cmap)
plt.yticks([])
plt.xticks(
    np.arange(0, len(time_hours), 120),
    [f"{int(h)}:00" for h in np.arange(0, len(time_hours) / 120, 1)],
)
plt.xlabel("Laikas (valandos)")
plt.title("Hipnograma (heatmap)")
handles = [Patch(color=color, label=label)
           for label, color in stage_colors_lt.items()]
plt.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(f"{PLOTS_FOLDER}/hypnogram_heatmap.png")
plt.close()


#traditional hypnogram with plotly

import plotly.graph_objects as go

stage_list = list(stage_labels_lt.values())

y_pos = [stage_list.index(stage_labels_lt[s]) for s in y]

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=time_hours,
    y=y_pos,
    mode="lines",
    line=dict(shape="hv", color="purple"),
    name="Hipnograma",
))

fig.update_yaxes(
    tickmode="array",
    tickvals=list(range(len(stage_list))),
    ticktext=stage_list,
    autorange="reversed",
)

fig.update_layout(
    title="Hipnograma",
    xaxis_title="Laikas (valandos)",
    yaxis_title="Miego stadija",
    hovermode="x",
)

fig.write_image(f"{PLOTS_FOLDER}/hypnogram_plotly.png")

# Bar chart of stage percentages
plt.figure(figsize=(6, 4))
colors_list = sns.color_palette("pastel", len(stage_percentages))
plt.bar(stage_percentages.keys(),
        [v * 100 for v in stage_percentages.values()],
        color=colors_list)
plt.ylabel("Procentai (%)")
plt.title("Miego stadijų pasiskirstymas")
plt.tight_layout()
plt.savefig(f"{PLOTS_FOLDER}/stages.png")
plt.close()
