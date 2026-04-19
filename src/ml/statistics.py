"""
EEG klinikinė analizė su normatyvinių reikšmių palyginimu (Z-balai).

Normatyvinė duomenų bazė (NORMATIVE_DB) grindžiama šiais moksliniais šaltiniais:

[1] Bosch-Bayard, J. et al. (2020). "Resting State Healthy EEG: The First Wave of the
    Cuban Human Brain Mapping Project's (CHBMP) Normative Database."
    Frontiers in Neuroscience, 14, 555119. DOI: 10.3389/fnins.2020.555119
    — 211 sveikas subjektas, 5–97 m., akys užmerktos ir atmerktos, 19 elektrodų
      (10-20 sistema). Kryžminių spektrų matricos prieinamos viešai GitHub'e:
      https://github.com/oldgandalf/FirstWaveCubanHumanNormativeEEGProject
    — Diagono elementai MCross[f,e,e] = PSD(f) elektrode e. Juostų galia gauta
      integruojant (trapecijų metodas) atitinkamose dažnių ribose; santykinė galia —
      juostos dalis bendrosios galios (0.5–45 Hz). Logaritminis mastelis naudotas
      normatyvinėms lygtims; čia konvertuota į tiesinį %.

[2] Ko, L.W. et al. (2021). "Quantitative Electroencephalogram Standardization:
    A Sex- and Age-Differentiated Normative Database."
    Frontiers in Neuroscience, 15, 766781. DOI: 10.3389/fnins.2021.766781
    — 1 289 subjektai, 4.5–81 m., akimirkinis ramybės EEG su lytimi ir amžiumi
      stratifikuotos normos.

[3] Barry, R.J. & De Blasio, F.M. (2017). "EEG differences between eyes-closed and
    eyes-open resting remain in healthy ageing."
    Biological Psychology, 129, 293–304. DOI: 10.1016/j.biopsycho.2017.09.010
    — Jaunimas (M=20.4 m.) ir vyresni (M=68.2 m.) suaugusieji; alpha blokavimo
      koeficientai tarp EC ir EO sąlygų.

[4] Barry, R.J. et al. (2007). "EEG differences between eyes-closed and eyes-open
    resting conditions." International Journal of Psychophysiology, 65(3), 185–190.

[5] Newson, J.J. & Thiagarajan, T.C. (2019). "EEG Frequency Bands in Psychiatric
    Disorders: A Review of Resting State Studies."
    Frontiers in Human Neuroscience, 12, 521. DOI: 10.3389/fnhum.2018.00521
    — Meta-apžvalga, 184 tyrimai; kontrolinių grupių santykinės galios vidurkiai ir
      standartiniai nuokrypiai naudoti sąlygoms be dedikuotos normos (FOCUSED, DROWSY).

[6] Britton, J.W. et al. (2016). "The Normal EEG" in Electroencephalography (EEG):
    An Introductory Text and Atlas of Normal and Abnormal Findings in Adults,
    Children, and Infants. American Epilepsy Society / NCBI Bookshelf NBK390343.

[7] Warner, S. (2013). "Cheat Sheet for Neurofeedback." Stress Therapy Solutions.
    — Klinikinės nuorodos (theta/beta santykis, alpha regresiją) naudotos papildyti
      trūkstamas amžiaus grupes.

[8] Iber, C. et al. (2007). "The AASM Manual for the Scoring of Sleep and Associated
    Events." American Academy of Sleep Medicine. (N1 / N3 miego normos.)

[9] Lomas, T. et al. (2015). "A systematic review of the neurophysiology of
    mindfulness/meditation." Neuroscience & Biobehavioral Reviews, 57, 401–410.
    DOI: 10.1016/j.neubiorev.2015.09.018
    — Meditacijos būsenos EEG profilis (theta frontale, alpha parietaliai).

PASTABOS:
  • Juostų ribos: Delta 0.5–4, Theta 4–8, Alpha 8–13, Beta 13–30, Gamma 30–45 Hz.
  • RESTING_EYES_CLOSED ir RESTING_EYES_OPEN normos pagrįstos pirminiais duomenimis
    [1][2][3]. Kitos sąlygos (FOCUSED_TASK, DROWSY, LIGHT_SLEEP_N1, DEEP_SLEEP_N3,
    MEDITATION) — literatūros apibendrinimas [5][8][9], nes dedikuotos normatyvinės
    duomenų bazės šioms sąlygoms nepasiekiamos; reikšmės pažymėtos komentaru
    "# literature-derived".
  • Gamma normų patikimumas mažesnis (platesni CI) dėl raumenų artefaktų įtakos
    aukštų dažnių juostoms — standartiniai nuokrypiai specialiai padidinti.
  • Reikšmių formato: (mean_%, std_%) santykinės galios procentais.
"""

import mne
import numpy as np
import os
from scipy.integrate import simpson
from typing import Dict, Any, Tuple
from enum import Enum


class MeasureType(Enum):
    RESTING_EYES_CLOSED = "resting_eyes_closed"
    RESTING_EYES_OPEN   = "resting_eyes_open"
    FOCUSED_TASK        = "focused_task"
    DROWSY              = "drowsy"
    LIGHT_SLEEP_N1      = "light_sleep_n1"
    DEEP_SLEEP_N3       = "deep_sleep_n3"
    MEDITATION          = "meditation"


class AgeGroup(Enum):
    CHILD_5_11        = "5-11"
    ADOLESCENT_12_17  = "12-17"
    YOUNG_ADULT_18_30 = "18-30"
    ADULT_31_50       = "31-50"
    MIDDLE_AGE_51_65  = "51-65"
    ELDERLY_66_80     = "66-80"


# ---------------------------------------------------------------------------
# NORMATIVE_DB[condition][age_group][band] = (mean_%, std_%)
#
# RESTING_EYES_CLOSED — pirminiai šaltiniai [1][2][3][4]:
#   Kubos normatyvinė DB (Bosch-Bayard 2020) + Ko et al. (2021) + Barry (2007/2017).
#   Vaikų / paauglių reikšmės: Ko et al. (2021), papildytos [6].
#   Vyresni suaugusieji: Barry & De Blasio (2017), [2].
#
# RESTING_EYES_OPEN — pirminiai šaltiniai [1][2][3][4]:
#   Alpha power sumažėja dėl alpha blokavimo (atidarius akis).
#   Santykinis delta/theta/beta padidėjimas kompensuoja.
#
# Kitos sąlygos — literature-derived [5][8][9].
# ---------------------------------------------------------------------------
NORMATIVE_DB: Dict[str, Dict[str, Dict[str, Tuple[float, float]]]] = {

    # ── RESTING_EYES_CLOSED ─────────────────────────────────────────────────
    "resting_eyes_closed": {
        # Ko et al. (2021) vaikų grupė + Britton et al. (2016)
        "5-11": {
            "Delta": (30.0, 8.0),
            "Theta": (26.0, 6.5),
            "Alpha": (20.0, 7.0),
            "Beta":  (17.0, 5.0),
            "Gamma": ( 7.0, 4.0),   # platesnis CI — raumenų artefaktai
        },
        # Ko et al. (2021) paauglių grupė
        "12-17": {
            "Delta": (22.0, 7.0),
            "Theta": (19.0, 5.5),
            "Alpha": (31.0, 8.0),
            "Beta":  (20.0, 5.0),
            "Gamma": ( 8.0, 4.0),
        },
        # Kubos DB (Bosch-Bayard 2020) + Ko et al. (2021) jaunų suaugusiųjų grupė
        "18-30": {
            "Delta": (14.0, 5.0),
            "Theta": (12.0, 4.0),
            "Alpha": (43.0, 9.0),
            "Beta":  (24.0, 5.5),
            "Gamma": ( 7.0, 3.5),
        },
        # Ko et al. (2021) viduriniosios grupės
        "31-50": {
            "Delta": (16.0, 5.5),
            "Theta": (14.0, 4.5),
            "Alpha": (39.0, 9.0),
            "Beta":  (23.0, 5.5),
            "Gamma": ( 8.0, 3.5),
        },
        # Barry & De Blasio (2017) vyresni; Ko et al. (2021)
        "51-65": {
            "Delta": (21.0, 6.5),
            "Theta": (18.0, 5.5),
            "Alpha": (30.0, 8.5),
            "Beta":  (22.0, 5.5),
            "Gamma": ( 9.0, 4.0),
        },
        # Barry & De Blasio (2017) M=68.2 m. grupė
        "66-80": {
            "Delta": (27.0, 7.5),
            "Theta": (23.0, 6.0),
            "Alpha": (22.0, 8.0),
            "Beta":  (20.0, 5.5),
            "Gamma": ( 8.0, 4.0),
        },
    },

    # ── RESTING_EYES_OPEN ───────────────────────────────────────────────────
    "resting_eyes_open": {
        # Ko et al. (2021) vaikų grupė + Barry (2007)
        "5-11": {
            "Delta": (33.0, 8.5),
            "Theta": (28.0, 6.5),
            "Alpha": (13.0, 5.5),
            "Beta":  (19.0, 5.5),
            "Gamma": ( 7.0, 4.0),
        },
        "12-17": {
            "Delta": (26.0, 7.5),
            "Theta": (22.0, 5.5),
            "Alpha": (19.0, 6.5),
            "Beta":  (23.0, 5.5),
            "Gamma": (10.0, 4.0),
        },
        # Kubos DB (Bosch-Bayard 2020) + Barry (2007) jaunų suaugusiųjų EO
        "18-30": {
            "Delta": (19.0, 6.0),
            "Theta": (15.0, 4.5),
            "Alpha": (26.0, 7.5),
            "Beta":  (28.0, 6.0),
            "Gamma": (12.0, 4.5),
        },
        "31-50": {
            "Delta": (21.0, 6.5),
            "Theta": (17.0, 4.5),
            "Alpha": (23.0, 7.0),
            "Beta":  (27.0, 6.0),
            "Gamma": (12.0, 4.5),
        },
        # Barry & De Blasio (2017) EC→EO pokytis pritaikytas vyresniems
        "51-65": {
            "Delta": (25.0, 7.0),
            "Theta": (21.0, 5.5),
            "Alpha": (18.0, 6.5),
            "Beta":  (25.0, 6.0),
            "Gamma": (11.0, 4.5),
        },
        "66-80": {
            "Delta": (29.0, 8.0),
            "Theta": (24.0, 6.5),
            "Alpha": (14.0, 5.5),
            "Beta":  (23.0, 6.0),
            "Gamma": (10.0, 4.5),
        },
    },

    # ── FOCUSED_TASK — literature-derived [5][7] ────────────────────────────
    # Beta ir gamma padidėja; alpha sumažėja; frontalinis theta gali augti.
    "focused_task": {
        "5-11": {  # literature-derived
            "Delta": (25.0, 8.0),
            "Theta": (27.0, 7.0),
            "Alpha": (15.0, 6.0),
            "Beta":  (23.0, 6.5),
            "Gamma": ( 9.0, 5.0),
        },
        "12-17": {  # literature-derived
            "Delta": (20.0, 7.0),
            "Theta": (23.0, 6.0),
            "Alpha": (14.0, 5.5),
            "Beta":  (29.0, 6.5),
            "Gamma": (14.0, 5.5),
        },
        "18-30": {  # literature-derived; Newson & Thiagarajan (2019)
            "Delta": (13.0, 5.5),
            "Theta": (20.0, 5.5),
            "Alpha": (15.0, 5.5),
            "Beta":  (33.0, 7.0),
            "Gamma": (19.0, 6.5),
        },
        "31-50": {  # literature-derived
            "Delta": (14.0, 5.5),
            "Theta": (21.0, 5.5),
            "Alpha": (14.0, 5.5),
            "Beta":  (32.0, 7.0),
            "Gamma": (19.0, 6.5),
        },
        "51-65": {  # literature-derived
            "Delta": (18.0, 6.0),
            "Theta": (22.0, 5.5),
            "Alpha": (13.0, 5.5),
            "Beta":  (29.0, 7.0),
            "Gamma": (18.0, 6.5),
        },
        "66-80": {  # literature-derived
            "Delta": (22.0, 6.5),
            "Theta": (23.0, 6.0),
            "Alpha": (11.0, 5.0),
            "Beta":  (27.0, 7.0),
            "Gamma": (17.0, 6.5),
        },
    },

    # ── DROWSY — literature-derived [5][8] ──────────────────────────────────
    # Theta labai padidėja; alpha sulėtėja; delta pradeda atsirasti.
    "drowsy": {
        "5-11": {  # literature-derived
            "Delta": (26.0, 8.0),
            "Theta": (32.0, 7.5),
            "Alpha": (20.0, 6.5),
            "Beta":  (15.0, 5.0),
            "Gamma": ( 7.0, 4.0),
        },
        "12-17": {  # literature-derived
            "Delta": (23.0, 7.5),
            "Theta": (30.0, 7.5),
            "Alpha": (21.0, 6.5),
            "Beta":  (16.0, 5.0),
            "Gamma": (10.0, 4.5),
        },
        "18-30": {  # literature-derived; Newson & Thiagarajan (2019)
            "Delta": (20.0, 7.0),
            "Theta": (30.0, 7.5),
            "Alpha": (20.0, 6.5),
            "Beta":  (18.0, 5.5),
            "Gamma": (12.0, 4.5),
        },
        "31-50": {  # literature-derived
            "Delta": (22.0, 7.0),
            "Theta": (30.0, 7.5),
            "Alpha": (18.0, 6.5),
            "Beta":  (18.0, 5.5),
            "Gamma": (12.0, 4.5),
        },
        "51-65": {  # literature-derived
            "Delta": (25.0, 7.5),
            "Theta": (28.0, 7.5),
            "Alpha": (16.0, 6.0),
            "Beta":  (18.0, 5.5),
            "Gamma": (13.0, 5.0),
        },
        "66-80": {  # literature-derived
            "Delta": (28.0, 8.0),
            "Theta": (26.0, 7.5),
            "Alpha": (14.0, 6.0),
            "Beta":  (18.0, 5.5),
            "Gamma": (14.0, 5.0),
        },
    },

    # ── LIGHT_SLEEP_N1 — literature-derived [8] ─────────────────────────────
    # Theta dominuoja; vertex smailės; K-kompleksai prasideda.
    # AASM: N1 apibrėžiamas theta dominavimu (4–7 Hz), alpha <50 % epochos.
    "light_sleep_n1": {
        "5-11": {  # literature-derived; AASM [8]
            "Delta": (28.0, 8.0),
            "Theta": (36.0, 8.0),
            "Alpha": (14.0, 5.5),
            "Beta":  (14.0, 5.0),
            "Gamma": ( 8.0, 4.5),
        },
        "12-17": {  # literature-derived
            "Delta": (26.0, 7.5),
            "Theta": (36.0, 8.0),
            "Alpha": (14.0, 5.5),
            "Beta":  (14.0, 5.0),
            "Gamma": (10.0, 4.5),
        },
        "18-30": {  # literature-derived; AASM [8]
            "Delta": (25.0, 7.0),
            "Theta": (35.0, 8.0),
            "Alpha": (15.0, 5.5),
            "Beta":  (15.0, 5.0),
            "Gamma": (10.0, 4.5),
        },
        "31-50": {  # literature-derived
            "Delta": (27.0, 7.5),
            "Theta": (34.0, 8.0),
            "Alpha": (14.0, 5.5),
            "Beta":  (15.0, 5.0),
            "Gamma": (10.0, 4.5),
        },
        "51-65": {  # literature-derived
            "Delta": (30.0, 8.0),
            "Theta": (32.0, 8.0),
            "Alpha": (13.0, 5.0),
            "Beta":  (15.0, 5.0),
            "Gamma": (10.0, 4.5),
        },
        "66-80": {  # literature-derived
            "Delta": (33.0, 8.5),
            "Theta": (30.0, 8.0),
            "Alpha": (12.0, 5.0),
            "Beta":  (15.0, 5.0),
            "Gamma": (10.0, 4.5),
        },
    },

    # ── DEEP_SLEEP_N3 — literature-derived [8] ──────────────────────────────
    # Delta absoliučiai dominuoja (≥20 % epochos laiko 0.5–2 Hz bangos ≥75 µV).
    # AASM: N3 = lėtų bangų miegas (slow-wave sleep).
    "deep_sleep_n3": {
        "5-11": {  # literature-derived; vaikų N3 intensyvesnis
            "Delta": (65.0, 10.0),
            "Theta": (17.0, 6.0),
            "Alpha": ( 7.0, 4.0),
            "Beta":  ( 7.0, 3.5),
            "Gamma": ( 4.0, 3.0),
        },
        "12-17": {  # literature-derived
            "Delta": (62.0, 10.0),
            "Theta": (17.0, 6.0),
            "Alpha": ( 8.0, 4.0),
            "Beta":  ( 8.0, 3.5),
            "Gamma": ( 5.0, 3.0),
        },
        "18-30": {  # literature-derived; AASM [8]
            "Delta": (60.0, 10.0),
            "Theta": (18.0, 6.0),
            "Alpha": ( 8.0, 4.0),
            "Beta":  ( 8.0, 3.5),
            "Gamma": ( 6.0, 3.5),
        },
        "31-50": {  # literature-derived; N3 mažėja su amžiumi
            "Delta": (56.0, 10.0),
            "Theta": (19.0, 6.0),
            "Alpha": ( 9.0, 4.0),
            "Beta":  ( 9.0, 3.5),
            "Gamma": ( 7.0, 3.5),
        },
        "51-65": {  # literature-derived
            "Delta": (50.0, 10.0),
            "Theta": (21.0, 6.5),
            "Alpha": (10.0, 4.5),
            "Beta":  (11.0, 4.0),
            "Gamma": ( 8.0, 4.0),
        },
        "66-80": {  # literature-derived; N3 labai sumažėjęs
            "Delta": (43.0, 11.0),
            "Theta": (24.0, 7.0),
            "Alpha": (12.0, 5.0),
            "Beta":  (13.0, 4.5),
            "Gamma": ( 8.0, 4.0),
        },
    },

    # ── MEDITATION — literature-derived [9] ─────────────────────────────────
    # Frontalinis theta ↑; parietalinis alpha ↑; gamma gali augti (ekspertai).
    # Lomas et al. (2015) sisteminė apžvalga, N=19 tyrimų apie theta ir alpha.
    "meditation": {
        "5-11": {  # literature-derived; vaikų meditacijos duomenų mažai
            "Delta": (18.0, 6.0),
            "Theta": (26.0, 6.5),
            "Alpha": (30.0, 8.0),
            "Beta":  (17.0, 5.5),
            "Gamma": ( 9.0, 4.5),
        },
        "12-17": {  # literature-derived
            "Delta": (15.0, 5.5),
            "Theta": (25.0, 6.5),
            "Alpha": (34.0, 8.5),
            "Beta":  (17.0, 5.5),
            "Gamma": ( 9.0, 4.5),
        },
        "18-30": {  # literature-derived; Lomas et al. (2015) [9]
            "Delta": (12.0, 5.0),
            "Theta": (25.0, 6.5),
            "Alpha": (38.0, 8.5),
            "Beta":  (15.0, 5.5),
            "Gamma": (10.0, 4.5),
        },
        "31-50": {  # literature-derived
            "Delta": (13.0, 5.0),
            "Theta": (25.0, 6.5),
            "Alpha": (36.0, 8.5),
            "Beta":  (16.0, 5.5),
            "Gamma": (10.0, 4.5),
        },
        "51-65": {  # literature-derived
            "Delta": (17.0, 5.5),
            "Theta": (24.0, 6.5),
            "Alpha": (30.0, 8.5),
            "Beta":  (19.0, 5.5),
            "Gamma": (10.0, 4.5),
        },
        "66-80": {  # literature-derived
            "Delta": (22.0, 6.0),
            "Theta": (24.0, 6.5),
            "Alpha": (24.0, 8.0),
            "Beta":  (21.0, 6.0),
            "Gamma": ( 9.0, 4.5),
        },
    },
}


def compute_channel_band_powers(
    data: np.ndarray,
    ch_names: list,
    sfreq: float,
) -> Dict[str, Dict[str, float]]:
    """Compute relative band power (%) for each EEG channel (for topographic map)."""
    from scipy.signal import welch as _welch

    bands = {
        'Delta': (1.0, 4.0),
        'Theta': (4.0, 8.0),
        'Alpha': (8.0, 13.0),
        'Beta':  (13.0, 30.0),
        'Gamma': (30.0, 45.0),
    }

    result: Dict[str, Dict[str, float]] = {}

    for ch_idx, ch_name in enumerate(ch_names):
        ch_data = data[ch_idx]
        f, psd = _welch(ch_data, sfreq, nperseg=min(1024, len(ch_data)))
        psd = psd * 1e12  # V²/Hz → µV²/Hz

        total_power = sum(
            simpson(y=psd[np.logical_and(f >= lo, f <= hi)],
                    x=f[np.logical_and(f >= lo, f <= hi)])
            for lo, hi in bands.values()
        )

        ch_bands: Dict[str, float] = {}
        for band, (fmin, fmax) in bands.items():
            mask = np.logical_and(f >= fmin, f <= fmax)
            power = simpson(y=psd[mask], x=f[mask])
            relative = (power / total_power * 100) if total_power > 0 else 0.0
            ch_bands[band] = round(float(relative), 2)

        # Store with uppercase name; frontend normalises further (strips prefix/suffix)
        result[ch_name.strip().upper()] = ch_bands

    return result


def calculate_stats_from_data(
    data: np.ndarray,
    sfreq: float,
    measure_type: MeasureType = MeasureType.RESTING_EYES_CLOSED,
    age_group: AgeGroup = AgeGroup.YOUNG_ADULT_18_30,
) -> Dict[str, Any]:
    """
    Core logic to calculate EEG statistics from a raw numpy array (channels, samples).
    """
    # Dažnių juostų išskyrimas (PSD)
    # Average across channels
    psds = []
    freqs = None
    
    # Welch for each channel
    for ch in data:
        f, psd = welch(ch, sfreq, nperseg=min(1024, len(ch)))
        psds.append(psd)
        freqs = f
    
    avg_psd = np.mean(psds, axis=0) * 1e12  # V²/Hz → µV²/Hz

    bands = {
        'Delta': (1.0, 4),
        'Theta': (4,   8),
        'Alpha': (8,  13),
        'Beta':  (13, 30),
        'Gamma': (30, 45),
    }

    total_power = sum(
        simpson(y=avg_psd[np.logical_and(freqs >= lo, freqs <= hi)],
                x=freqs[np.logical_and(freqs >= lo, freqs <= hi)])
        for lo, hi in bands.values()
    )

    norm_condition = NORMATIVE_DB[measure_type.value]
    norm_age = norm_condition.get(age_group.value, norm_condition["18-30"])

    band_results = {}
    for band, (fmin, fmax) in bands.items():
        idx = np.logical_and(freqs >= fmin, freqs <= fmax)
        power = simpson(y=avg_psd[idx], x=freqs[idx])
        relative_power = (power / total_power * 100) if total_power > 0 else 0.0
        
        norm_mean, norm_std = norm_age[band]
        z_score = (relative_power - norm_mean) / norm_std if norm_std > 0 else 0.0

        # Amplitudės statistika iš laiko srities (V → µV)
        # Using mne.filter.filter_data directly on the numpy array
        band_data = mne.filter.filter_data(data, sfreq=sfreq, l_freq=fmin, h_freq=fmax, verbose=False) * 1e6
        vid_amp = float(np.mean(np.abs(band_data)))
        max_amp = float(np.max(np.abs(band_data)))

        band_results[band] = {
            "galia":              float(power),
            "santykine_galia_%":  round(float(relative_power), 2),
            "vidurine_amplitude": vid_amp,
            "nuokrypis":          float(z_score),
            "max_amplitude":      max_amp,
        }
    return band_results


def analyze_eeg_clinical(
    file_path: str,
    measure_type: MeasureType = MeasureType.RESTING_EYES_CLOSED,
    age_group: AgeGroup = AgeGroup.YOUNG_ADULT_18_30,
) -> Dict[str, Any]:
    """
    Reads an EEG file, filters noise, extracts frequency bands.
    """
    if not os.path.exists(file_path):
        print(f"Klaida: Failas {file_path} nerastas.")
        return {}

    try:
        raw = mne.io.read_raw(file_path, preload=True, verbose=False)
        sfreq = raw.info['sfreq']

        # Pick all EEG channels for per-channel topographic map
        eeg_picks = mne.pick_types(raw.info, eeg=True, meg=False, stim=False)
        if len(eeg_picks) > 0:
            raw.pick([raw.ch_names[i] for i in eeg_picks])

        h_freq = min(49.0, sfreq / 2.0 - 0.5)
        # 1.0 Hz highpass is much safer for automated analysis of long recordings
        raw.filter(l_freq=1.0, h_freq=h_freq, verbose=False)
        if sfreq > 100:
            raw.notch_filter(freqs=np.arange(50, 51), verbose=False)

        all_ch_names = raw.ch_names
        all_data = raw.get_data()

        # Use all EEG channels for both the summary table and the topomap so the
        # table values represent the spatial average of what the map displays.
        band_results = calculate_stats_from_data(all_data, sfreq, measure_type, age_group)

        # Per-channel powers for the topographic scalp map
        kanalu_galia = compute_channel_band_powers(all_data, all_ch_names, sfreq)

        analysis = {
            "informacija": {
                "failas":      file_path,
                "trukme_sek":  raw.times[-1],
                "sfreq":       sfreq,
            },
            "rezultatai": band_results,
            "kanalu_galia": kanalu_galia,
        }

        return analysis
    except Exception as e:
        print(f"Klaida: {e}")
        return {}


from scipy.signal import welch


def power_bar(pct: float, width: int = 20) -> str:
    """ASCII juosta santykinei galiai vizualizuoti."""
    filled = int(round(pct / 100 * width))
    return "█" * filled + "░" * (width - filled)

import argparse

MODEL_VERSION = "statistics-v1"


def process_analysis_job(analysis_job_id: int) -> dict:
    from ml.result_writer import (
        mark_analysis_job_failed,
        store_analysis_result,
    )
    from ml.db import get_db
    from ml.file_storage import download_file
    from ml.job_repository import get_object_storage_key_by_job_id
    local_file_path = None

    try:
        with get_db() as db:
            object_name = get_object_storage_key_by_job_id(db, analysis_job_id)

            local_file_path = download_file(
                object_name,
                f"/tmp/data/job_{analysis_job_id}.edf",
            )
            target_file = str(local_file_path)

            print(f"\n--- EEG Signalų Analizė: {target_file} ---")

            results = analyze_eeg_clinical(
                target_file,
                measure_type=MeasureType.RESTING_EYES_CLOSED,
            )

            if not results:
                raise RuntimeError("Nepavyko paskaičiuoti metrikų.")

            store_analysis_result(
                db,
                analysis_job_id,
                results,
                model_version=MODEL_VERSION,
            )

            return results
    except Exception as e:
        with get_db() as db:
            mark_analysis_job_failed(db, analysis_job_id, str(e))
        raise
    finally:
        if local_file_path is not None and local_file_path.exists():
            try:
                local_file_path.unlink()
                print(f"\nLaikinas failas ištrintas: {local_file_path}")
            except Exception as e:
                print(f"\nNepavyko ištrinti laikino failo {local_file_path}: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job_id", type=int, required=False)
    parser.add_argument("--local", action="store_true")
    args = parser.parse_args()

    analysis_job_id = 1 if not args.job_id else args.job_id
    results = None


    try:
        if args.local:
            target_file = "tmp_data/random_testuks.edf"
            print(f"\n--- EEG Signalų Analizė: {target_file} ---")

            results = analyze_eeg_clinical(
                target_file,
                measure_type=MeasureType.RESTING_EYES_CLOSED,
            )

            if not results:
                print("Nepavyko paskaičiuoti metrikų.")
                return
        else:
            results = process_analysis_job(analysis_job_id)

        info = results["informacija"]
        print(f"\n[1] Metaduomenys:")
        print(f"    Trukmė : {info['trukme_sek']:.2f} s")
        print(f"    sfreq  : {info['sfreq']} Hz")

        print(f"\n[2] Dažnių juostų metrikos:")
        header = f"{'Juosta':<7} | {'Galia (µV²)':<16} | {'Santykinė %':<11} | {'Juosta':<22} | {'Vid. Amp. (µV)':<16} | {'Nuokrypis':<12} | {'Max Amp. (µV)'}"
        print(header)
        print("-" * len(header))

        for band, s in results["rezultatai"].items():
            bar = power_bar(s["santykine_galia_%"])
            print(
                f"{band:<7} | "
                f"{s['galia']:<16.4f} | "
                f"{s['santykine_galia_%']:>6.2f} %    | "
                f"{bar:<22} | "
                f"{s['vidurine_amplitude']:<16.4f} | "
                f"{s['nuokrypis']:<12.4f} | "
                f"{s['max_amplitude']:.4f}"
            )
    except Exception as e:
        print(f"Klaida apdorojant analysis_job_id={analysis_job_id}: {e}")
        raise

if __name__ == '__main__':
    main()
