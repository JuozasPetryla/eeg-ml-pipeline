from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import mne
import numpy as np
from matplotlib.patches import Patch
from scipy.integrate import trapezoid
from scipy.signal import welch

from ml.db import get_db
from ml.file_storage import download_file, ensure_bucket_exists, upload_file
from ml.job_repository import get_object_storage_key_by_job_id
from ml.result_writer import (
    mark_analysis_job_failed,
    store_analysis_result,
)

MODEL_VERSION = "sleep-v1"

BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
}

STAGE_NAMES_LT = {
    0: "Budrumas",
    1: "Lengvas miegas N1",
    2: "Lengvas miegas N2",
    3: "Gilus miegas N3",
    4: "REM miegas",
}

STAGE_COLORS_LT = {
    "Budrumas": "#FF6347",
    "Lengvas miegas N1": "#FFD700",
    "Lengvas miegas N2": "#87CEFA",
    "Gilus miegas N3": "#1E90FF",
    "REM miegas": "#32CD32",
}

MODEL_DIR = Path(__file__).resolve().parent
SCALER = joblib.load(MODEL_DIR / "scaler.pkl")
MODEL = joblib.load(MODEL_DIR / "model.pkl")

ASSET_CONTENT_TYPE = "image/png"


def extract_band_powers(epoch: np.ndarray, sfreq: float) -> list[float]:
    features = []
    for ch in epoch:
        freqs, psd = welch(ch, sfreq, nperseg=min(1024, len(ch)))
        for fmin, fmax in BANDS.values():
            mask = (freqs >= fmin) & (freqs < fmax)
            features.append(trapezoid(psd[mask], freqs[mask]))
    return features


def load_subject(psg_path: Path):
    raw = mne.io.read_raw_edf(psg_path, preload=True, verbose=False)
    raw_eeg = raw.copy().pick_types(eeg=True, meg=False, stim=False)

    sfreq = raw_eeg.info["sfreq"]
    epoch_len = 30.0
    epoch_samples = int(epoch_len * sfreq)

    data = raw_eeg.get_data()
    n_epochs = data.shape[1] // epoch_samples

    features = []
    for i in range(n_epochs):
        start = i * epoch_samples
        end = start + epoch_samples
        epoch = data[:, start:end]
        features.append(extract_band_powers(epoch, sfreq))

    x = np.array(features)
    x_scaled = SCALER.transform(x)
    time_hours = np.arange(n_epochs) / 120
    return x_scaled, time_hours


def save_scatter(y: np.ndarray, time_hours: np.ndarray, output_path: Path) -> None:
    plt.figure(figsize=(15, 4))
    for code, label in STAGE_NAMES_LT.items():
        idx = np.where(y == code)[0]
        if len(idx) > 0:
            plt.scatter(
                time_hours[idx],
                [label] * len(idx),
                color=STAGE_COLORS_LT[label],
                s=20,
                label=label,
            )
    plt.xlabel("Laikas (valandos)")
    plt.ylabel("Miego stadija")
    plt.title("Hipnograma")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_heatmap(y: np.ndarray, time_hours: np.ndarray, output_path: Path) -> None:
    y_idx = np.array([code for code in y])
    cmap = mcolors.ListedColormap([STAGE_COLORS_LT[label] for label in STAGE_NAMES_LT.values()])

    plt.figure(figsize=(15, 4))
    plt.imshow([y_idx], aspect="auto", cmap=cmap)
    plt.yticks([])
    plt.xticks(
        np.arange(0, len(time_hours), 120),
        [f"{int(h)}:00" for h in np.arange(0, len(time_hours) / 120, 1)],
    )
    plt.xlabel("Laikas (valandos)")
    plt.title("Hipnograma (heatmap)")
    handles = [Patch(color=color, label=label) for label, color in STAGE_COLORS_LT.items()]
    plt.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_classic(y: np.ndarray, time_hours: np.ndarray, output_path: Path) -> None:
    stage_labels = [STAGE_NAMES_LT[s] for s in y]
    stage_order = list(STAGE_NAMES_LT.values())
    y_pos = [stage_order.index(label) for label in stage_labels]

    plt.figure(figsize=(15, 4))
    plt.step(time_hours, y_pos, where="post", color="#203d63")
    plt.yticks(range(len(stage_order)), stage_order)
    plt.gca().invert_yaxis()
    plt.xlabel("Laikas (valandos)")
    plt.ylabel("Miego stadija")
    plt.title("Klasikinė hipnograma")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_stage_distribution(y: np.ndarray, output_path: Path) -> None:
    total = len(y)
    percentages = {
        STAGE_NAMES_LT[code]: float(np.sum(y == code) / total * 100)
        for code in sorted(set(y.tolist()))
    }

    colors = [STAGE_COLORS_LT[label] for label in percentages.keys()]

    plt.figure(figsize=(6, 4))
    plt.bar(percentages.keys(), percentages.values(), color=colors)
    plt.ylabel("Procentai (%)")
    plt.title("Miego stadijų pasiskirstymas")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_eeg_with_stages(raw_eeg: mne.io.BaseRaw, y: np.ndarray, output_path: Path) -> list[str]:
    sfreq = raw_eeg.info["sfreq"]
    data = raw_eeg.get_data()
    n_samples = data.shape[1]

    t = np.arange(n_samples) / sfreq / 3600.0
    epoch_len = 30.0
    samples_per_epoch = int(epoch_len * sfreq)
    n_epochs = len(y)

    if n_epochs * samples_per_epoch > n_samples:
        raise RuntimeError("Predicted epochs exceed available EEG samples")

    stage_per_sample = np.repeat(y, samples_per_epoch)[:n_samples]

    fig, (ax1, ax2, ax_stage) = plt.subplots(
        3,
        1,
        figsize=(15, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 1, 0.5]},
    )

    ch_names = raw_eeg.ch_names
    fpz_idx = ch_names.index("Fpz-Cz") if "Fpz-Cz" in ch_names else 0
    pf_idx = ch_names.index("Pf-Cz") if "Pf-Cz" in ch_names else min(1, len(ch_names) - 1)

    ax1.plot(t, data[fpz_idx] * 1e6, color="blue", linewidth=0.8)
    ax1.set_ylabel(f"{ch_names[fpz_idx]} (µV)")
    ax1.set_title("EEG su miego stadijomis")
    ax1.grid(alpha=0.3)

    ax2.plot(t, data[pf_idx] * 1e6, color="orange", linewidth=0.8)
    ax2.set_ylabel(f"{ch_names[pf_idx]} (µV)")
    ax2.grid(alpha=0.3)

    stage_list = list(stage_labels_to_codes().keys())
    ax_stage.imshow(
        [stage_per_sample],
        aspect="auto",
        extent=[t[0], t[-1], 0, 1],
        cmap=mcolors.ListedColormap(
            [STAGE_COLORS_LT[STAGE_NAMES_LT[s]] for s in stage_list]
        ),
    )

    handles = [
        Patch(color=STAGE_COLORS_LT[STAGE_NAMES_LT[s]], label=STAGE_NAMES_LT[s])
        for s in stage_list
    ]
    ax_stage.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc="upper left")
    ax_stage.set_yticks([])
    ax_stage.set_xlabel("Laikas (valandos)")
    ax_stage.set_title("Modelio prognozuotos miego stadijos")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    return [ch_names[fpz_idx], ch_names[pf_idx]]


def stage_labels_to_codes() -> dict[int, str]:
    return {code: label for code, label in STAGE_NAMES_LT.items()}


def process_night_analysis_job(analysis_job_id: int) -> dict[str, str]:
    local_file_path = None
    output_dir = None

    try:
        with get_db() as db:
            object_name = get_object_storage_key_by_job_id(db, analysis_job_id)

        local_file_path = download_file(
            object_name,
            f"/tmp/data/night_job_{analysis_job_id}.edf",
        )
        output_dir = Path(f"/tmp/night_results/job_{analysis_job_id}")
        output_dir.mkdir(parents=True, exist_ok=True)

        x_new, time_hours = load_subject(local_file_path)
        y = MODEL.predict(x_new)

        raw = mne.io.read_raw_edf(local_file_path, preload=True, verbose=False)
        raw_eeg = raw.copy().pick_types(eeg=True, meg=False, stim=False)

        scatter_path = output_dir / "hypnogram_scatter.png"
        classic_path = output_dir / "hypnogram_classic.png"
        heatmap_path = output_dir / "hypnogram_heatmap.png"
        stages_path = output_dir / "stages.png"
        eeg_with_stages_path = output_dir / "eeg_with_stages.png"

        save_scatter(y, time_hours, scatter_path)
        save_classic(y, time_hours, classic_path)
        save_heatmap(y, time_hours, heatmap_path)
        save_stage_distribution(y, stages_path)
        save_eeg_with_stages(raw_eeg, y, eeg_with_stages_path)

        ensure_bucket_exists()
        asset_paths = {
            "scatter": scatter_path,
            "classic": classic_path,
            "heatmap": heatmap_path,
            "stages": stages_path,
            "eeg_with_stages": eeg_with_stages_path,
        }
        uploaded_assets = {}
        for asset_name, asset_path in asset_paths.items():
            object_key = f"analysis-results/{analysis_job_id}/{asset_path.name}"
            uploaded_assets[asset_name] = upload_file(
                asset_path,
                object_key,
                ASSET_CONTENT_TYPE,
            )

        result_payload = uploaded_assets

        with get_db() as db:
            store_analysis_result(
                db,
                analysis_job_id,
                result_payload,
                model_version=MODEL_VERSION,
            )

        return result_payload
    except Exception as e:
        with get_db() as db:
            mark_analysis_job_failed(db, analysis_job_id, str(e))
        raise
    finally:
        if local_file_path is not None and local_file_path.exists():
            local_file_path.unlink()
