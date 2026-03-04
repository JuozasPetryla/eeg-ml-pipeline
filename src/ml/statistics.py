import mne
import numpy as np
import os
from scipy.integrate import simpson
from typing import Dict, Any


def analyze_eeg_clinical(file_path: str) -> Dict[str, Any]:
    """
    Reads an EEG file, filters noise, extracts frequency bands
    """
    if not os.path.exists(file_path):
        print(f"Klaida: Failas {file_path} nerastas.")
        return {}

    try:
        # 1. Load the raw data
        raw = mne.io.read_raw(file_path, preload=True, verbose=False)

        # 2. Triukšmo filtravimas (Noise Filtering)
        raw.filter(l_freq=0.5, h_freq=50.0, verbose=False)
        raw.notch_filter(freqs=np.arange(50, 51), verbose=False)

        # 3. Dažnių juostų išskyrimas (PSD)
        spectrum = raw.compute_psd(method='welch', fmin=0.5, fmax=45.0, verbose=False)
        psds, freqs = spectrum.get_data(return_freqs=True)
        avg_psd = psds.mean(axis=0)  # Vidurkis per visus kanalus

        bands = {
            'Delta': (0.5, 4),
            'Theta': (4,   8),
            'Alpha': (8,  13),
            'Beta':  (13, 30),
            'Gamma': (30, 45),
        }

        # ── Bendra galia (visų juostų suma) — santykinei galiai skaičiuoti ──
        total_power = sum(
            simpson(y=avg_psd[np.logical_and(freqs >= lo, freqs <= hi)],
                    x=freqs[np.logical_and(freqs >= lo, freqs <= hi)])
            for lo, hi in bands.values()
        )

        band_results = {}
        for band, (fmin, fmax) in bands.items():
            idx = np.logical_and(freqs >= fmin, freqs <= fmax)

            # Absoliuti galia
            power = simpson(y=avg_psd[idx], x=freqs[idx])

            # Santykinė galia (%)
            relative_power = (power / total_power * 100) if total_power > 0 else 0.0

            # Amplitudės statistika iš laiko srities
            band_raw  = raw.copy().filter(l_freq=fmin, h_freq=fmax, verbose=False)
            band_data = band_raw.get_data()

            band_results[band] = {
                "galia":              float(power),
                "santykine_galia_%":  round(float(relative_power), 2),
                "vidurine_amplitude": float(np.mean(np.abs(band_data))),
                "nuokrypis":          float(np.std(band_data)),
                "max_amplitude":      float(np.max(np.abs(band_data))),
            }

        analysis = {
            "informacija": {
                "failas":      file_path,
                "trukme_sek":  raw.times[-1],
                "sfreq":       raw.info['sfreq'],
            },
            "rezultatai": band_results,
        }
        return analysis

    except Exception as e:
        print(f"Klaida: {e}")
        return {}


def power_bar(pct: float, width: int = 20) -> str:
    """ASCII juosta santykinei galiai vizualizuoti."""
    filled = int(round(pct / 100 * width))
    return "█" * filled + "░" * (width - filled)


def main():
    target_file = "tmp_data/random_testuks.edf"

    print(f"\n--- EEG Signalų Analizė: {target_file} ---")
    results = analyze_eeg_clinical(target_file)

    if not results:
        print("Nepavyko paskaičiuoti metrikų.")
        return

    info = results["informacija"]
    print(f"\n[1] Metaduomenys:")
    print(f"    Trukmė : {info['trukme_sek']:.2f} s")
    print(f"    sfreq  : {info['sfreq']} Hz")

    print(f"\n[2] Dažnių juostų metrikos:")
    header = f"{'Juosta':<7} | {'Galia (mokslinė)':<16} | {'Santykinė %':<11} | {'Juosta':<22} | {'Vid. Amp.':<12} | {'Nuokrypis':<12} | {'Max Amp.'}"
    print(header)
    print("-" * len(header))
    # istrinsim situs printus, dbr kad graziau butu.
    for band, s in results["rezultatai"].items():
        bar = power_bar(s["santykine_galia_%"])
        print(
            f"{band:<7} | "
            f"{s['galia']:<16.4e} | "
            f"{s['santykine_galia_%']:>6.2f} %    | "
            f"{bar:<22} | "
            f"{s['vidurine_amplitude']:.4e}   | "
            f"{s['nuokrypis']:.4e}   | "
            f"{s['max_amplitude']:.4e}"
        )


if __name__ == '__main__':
    main()