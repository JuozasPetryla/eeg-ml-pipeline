#%%
import os
import warnings
import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
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
#%%
FONT_PATH = "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf"
pdfmetrics.registerFont(TTFont('DejaVu', FONT_PATH))
#%%
#CASSETTE_PATH = "sleep-cassette"
CASSETTE_PATH = Path("/Users/edvinaskucys/Library/CloudStorage/GoogleDrive-edvinas.kuc@gmail.com/My Drive/Uni/Produkto vystymo projektas/physionet.org/files/sleep-edfx/1.0.0/sleep-cassette")


PLOTS_FOLDER = Path("/Users/edvinaskucys/Library/CloudStorage/GoogleDrive-edvinas.kuc@gmail.com/My Drive/Uni/Produkto vystymo projektas/models/sleep_plots")
os.makedirs(PLOTS_FOLDER, exist_ok=True)
#%%
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

TRAIN_SUFFIXES = ["EC", "EJ", "EV", "EU", "FM"]
TEST_SUFFIXES  = ["EP", "EH", "EA", "FC"]
#%%
def extract_band_powers(epoch, sfreq):
    """Extract per-channel band-power features from one epoch (channels x samples)."""
    features = []
    for ch in epoch:
        f, psd = welch(ch, sfreq, nperseg=min(1024, len(ch)))
        for band_name, (fmin, fmax) in BANDS.items():
            mask = (f >= fmin) & (f < fmax)
            features.append(trapezoid(psd[mask], f[mask]))
    return features
#%%
def load_dataset(base_path: Path):
    """
    Load sleep-cassette data, split into train/test by subject suffix.
    Returns X_train, y_train, X_test, y_test as numpy arrays.
    """
    psg_files = sorted(base_path.glob("*PSG.edf"))
    hyp_files = sorted(base_path.glob("*Hypnogram.edf"))

    hyp_map = {}
    for h in hyp_files:
        root = h.name.split("-Hypnogram.edf")[0]
        hyp_map[root] = h

    X_train_all, y_train_all = [], []
    X_test_all,  y_test_all  = [], []

    warnings.filterwarnings("ignore")

    for p in psg_files:
        psg_id = p.name.split("-PSG.edf")[0]
        hyp_file = None
        is_test = False

        for suffix in TRAIN_SUFFIXES + TEST_SUFFIXES:
            candidate = psg_id[:-2] + suffix
            if candidate in hyp_map:
                hyp_file = hyp_map[candidate]
                is_test = (suffix in TEST_SUFFIXES)
                break

        if hyp_file is None:
            print(f"Skipping {p.name} (no hypnogram found)")
            continue

        split = "TEST" if is_test else "TRAIN"
        print(f"Processing {p.name} -> {hyp_file.name} [{split}]")

        try:
            raw = mne.io.read_raw_edf(p, preload=True, verbose=False)
            ann = mne.read_annotations(hyp_file)
            raw.set_annotations(ann)
        except Exception as e:
            print(f"  Error loading {p.name}: {e}")
            continue

        try:
            events, event_id_raw = mne.events_from_annotations(raw, chunk_duration=30.0)
        except Exception as e:
            print(f"  Error extracting events from {p.name}: {e}")
            continue

        y = events[:, 2].copy()
        event_id = {}
        for label_str, class_id in DESC_TO_STAGE.items():
            if label_str in event_id_raw:
                raw_id = event_id_raw[label_str]
                y[y == raw_id] = class_id
                event_id[label_str] = class_id

        mask_valid = np.isin(y, list(DESC_TO_STAGE.values()))
        if not mask_valid.any():
            print(f"  No valid sleep stages in {p.name}, skipping.")
            continue

        y = y[mask_valid]
        events = events[mask_valid].copy()
        events[:, 2] = y

        try:
            epochs = mne.Epochs(
                raw, events, event_id=event_id,
                tmin=0, tmax=30.0 - 1 / raw.info["sfreq"],
                baseline=None, preload=True, verbose=False
            )
        except Exception as e:
            print(f"  Error creating epochs for {p.name}: {e}")
            continue

        X = epochs.get_data()
        sfreq = raw.info["sfreq"]

        target_X = X_test_all  if is_test else X_train_all
        target_y = y_test_all  if is_test else y_train_all

        for ep in X:
            target_X.append(extract_band_powers(ep, sfreq))
        target_y.append(y)

    X_train = np.array(X_train_all)
    y_train = np.concatenate(y_train_all) if y_train_all else np.array([])
    X_test  = np.array(X_test_all)
    y_test  = np.concatenate(y_test_all)  if y_test_all  else np.array([])

    print(f"\n--- Data Split ---")
    print(f"Training epochs : {len(y_train)} | Feature shape: {X_train.shape}")
    print(f"Testing  epochs : {len(y_test)}  | Feature shape: {X_test.shape}")

    return X_train, y_train, X_test, y_test
#%%
X_train, y_train, X_test, y_test = load_dataset(CASSETTE_PATH)
#%%
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    class_weight="balanced",
    random_state=42,
)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)

print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred,
      target_names=[STAGE_NAMES.get(i, str(i)) for i in sorted(STAGE_NAMES)]))
#%%
import joblib

# Save after training
joblib.dump(scaler, "scaler.pkl");
joblib.dump(model, "model.pkl");
#%%
def load_new_subject(psg_path: str):
    """
    Load a single new PSG EDF file and extract features for prediction.
    No hypnogram needed — returns X (features) and epoch timestamps.

    Args:
        psg_path: path to the PSG .edf file

    Returns:
        X_scaled: feature array ready for model.predict()
        time_hours: timestamp for each epoch (for plotting)
    """
    psg_path = Path(psg_path)
    print(f"Loading {psg_path.name}...")

    try:
        raw = mne.io.read_raw_edf(psg_path, preload=True, verbose=False)
    except Exception as e:
        raise RuntimeError(f"Failed to load {psg_path}: {e}")

    sfreq = raw.info["sfreq"]
    epoch_len = 30.0
    epoch_samples = int(epoch_len * sfreq)

    data = raw.get_data()  # shape: (n_channels, n_samples)
    n_epochs = data.shape[1] // epoch_samples

    print(f"  Channels: {len(raw.ch_names)}, Sfreq: {sfreq} Hz")
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
    return X_scaled, time_hours

#%%
x_new, time_hours = load_new_subject(Path("/Users/edvinaskucys/Library/CloudStorage/GoogleDrive-edvinas.kuc@gmail.com/My Drive/Uni/Produkto vystymo projektas/physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/SC4012E0-PSG.edf"))
x_new_scaled = scaler.transform(x_new)
y = model.predict(x_new_scaled)  # use test set for display/plots
total = len(y)
sleep_eff   = np.sum(y > 1) / total
transitions = sum(y[i] != y[i-1] for i in range(1, total))
sleep_latency = 110
rem_latency   = 132
waso          = 0

stage_percentages = {
    STAGE_NAMES.get(s, str(s)): np.sum(y == s) / total
    for s in np.unique(y)
}

score = 55
ai_comments = [
    "Low sleep efficiency (possible insomnia)",
    "Delayed REM sleep",
    "High sleep fragmentation",
]
#%%
stage_labels_lt = {
    0: "Budrumas",
    1: "Lengvas miegas N1",
    2: "Lengvas miegas N2",
    3: "Gilus miegas N3",
    4: "REM miegas",
    6: "Nežinoma",
}

stage_colors_lt = {
    "Budrumas":          "#FF6347",
    "Lengvas miegas N1": "#FFD700",
    "Lengvas miegas N2": "#87CEFA",
    "Gilus miegas N3":   "#1E90FF",
    "REM miegas":        "#32CD32",
    "Nežinoma":          "#D3D3D3",
}

#time_hours = np.arange(len(y)) / 120

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
plt.show()
plt.close()

#%%
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
plt.show()
plt.savefig(f"{PLOTS_FOLDER}/hypnogram_heatmap.png")
plt.close()

#%%
plt.figure(figsize=(15, 4))

stage_list = list(stage_labels_lt.values())

for code, label in stage_labels_lt.items():
    mask = np.array([1 if s == code else 0 for s in y])
    y_pos = stage_list.index(label)
    plt.fill_between(
        time_hours,
        y_pos - 0.4,
        y_pos - 0.4 + mask * 0.8,
        step="mid",
        color=stage_colors_lt[label],
        alpha=0.8,
        label=label,
    )

plt.yticks(range(len(stage_list)), stage_list)
plt.xlabel("Laikas (valandos)")
plt.ylabel("Miego stadija")
plt.title("Hipnograma (filled area)")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{PLOTS_FOLDER}/hypnogram_filled.png")
plt.show()
plt.close()

#%%
plt.figure(figsize=(12,4))
plt.plot(y, color="purple")
plt.fill_between(range(len(y[:500])), y[:500], color="purple", alpha=0.3)
plt.gca().invert_yaxis()
plt.title("Miego stadijos per laiką")
plt.savefig(f"{PLOTS_FOLDER}/hypnogram.png")
plt.show()
plt.close()
#%%
plt.figure(figsize=(6, 4))
colors_list = sns.color_palette("pastel", len(stage_percentages))
plt.bar(stage_percentages.keys(),
        [v * 100 for v in stage_percentages.values()],
        color=colors_list)
plt.ylabel("Procentai (%)")
plt.title("Miego stadijų pasiskirstymas")
plt.tight_layout()
plt.show()
plt.savefig(f"{PLOTS_FOLDER}/stages.png")
plt.close()

#%%
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, cmap="coolwarm")
plt.title("Klasifikacijos matrica")
plt.tight_layout()
plt.savefig(f"{PLOTS_FOLDER}/cm.png")
plt.show()
plt.close()
#%%
# import xgboost as xgb
# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# from sklearn.model_selection import GridSearchCV
#
# ############################
# # MODEL
# ############################
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled  = scaler.transform(X_test)
#
# # Parameter grid to test
# param_grid = {
#     "n_estimators":    [100, 300],
#     "max_depth":       [4, 6, 8],
#     "learning_rate":   [0.05, 0.1, 0.2],
#     "subsample":       [0.7, 1.0],
#     "colsample_bytree":[0.7, 1.0],
# }
#
# xgb_base = xgb.XGBClassifier(
#     objective="multi:softmax",
#     num_class=5,
#     eval_metric="mlogloss",
#     use_label_encoder=False,
#     random_state=42,
#     n_jobs=-1,
# )
#
# search = GridSearchCV(
#     xgb_base,
#     param_grid,
#     cv=3,
#     scoring="f1_macro",   # macro because classes are imbalanced
#     verbose=2,
#     n_jobs=-1,
# )
#
# search.fit(X_train_scaled, y_train)
#
# print("Best params:", search.best_params_)
# print("Best CV f1_macro:", search.best_score_)
#
# model = search.best_estimator_
# y_pred = model.predict(X_test_scaled)
# acc = accuracy_score(y_test, y_pred)
#
# print("\n--- Classification Report ---")
# print(classification_report(y_test, y_pred,
#       target_names=[STAGE_NAMES.get(i, str(i)) for i in sorted(STAGE_NAMES)]))

#%%
# from sklearn.metrics import confusion_matrix
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# cm = confusion_matrix(y_test, y_pred)
#
# labels = [STAGE_NAMES.get(i, str(i)) for i in sorted(STAGE_NAMES)]
#
# plt.figure(figsize=(8, 6))
# sns.heatmap(
#     cm,
#     annot=True,
#     fmt="d",
#     cmap="coolwarm",
#     xticklabels=labels,
#     yticklabels=labels,
# )
# plt.xlabel("Predicted")
# plt.ylabel("Actual")
# plt.title(f"Confusion Matrix — XGBoost (acc: {acc:.4f})")
# plt.tight_layout()
# plt.savefig(f"{PLOTS_FOLDER}/cm_xgb.png")
# plt.show()

#%%
