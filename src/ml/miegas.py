import os
import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

from scipy.signal import welch
from scipy.stats import skew, kurtosis

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors

############################
# FONT
############################
FONT_PATH = "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf"
pdfmetrics.registerFont(TTFont('DejaVu', FONT_PATH))

############################
# SETTINGS
############################
CASSETTE_PATH = "sleep-cassette"
TELEMETRY_PATH = "sleep-telemetry"

PLOTS_FOLDER = "sleep_plots"
os.makedirs(PLOTS_FOLDER, exist_ok=True)

############################
# STAGES
############################
STAGE_NAMES = {
    0: "Wake",
    1: "N1",
    2: "N2",
    3: "N3 (Deep)",
    4: "N4",
    5: "REM",
    6: "Unknown",
    7: "7"
}

############################
# FEATURES
############################
def bandpower(signal, sf, band):
    f, Pxx = welch(signal, sf)
    idx = (f >= band[0]) & (f <= band[1])
    return np.mean(Pxx[idx]) if np.any(idx) else 0

def extract_features(signal, sf):
    signal = signal.flatten()
    return [
        np.mean(signal),
        np.std(signal),
        np.max(signal),
        np.min(signal),
        np.mean(signal**2),
        bandpower(signal, sf, (0.5, 4)),
        bandpower(signal, sf, (4, 8)),
        bandpower(signal, sf, (8, 12)),
        bandpower(signal, sf, (12, 30)),
        skew(signal),
        kurtosis(signal)
    ]

############################
# LOAD DATA
############################
def load_dataset(folder):
    X, y = [], []
    files = os.listdir(folder)
    psg_files = [f for f in files if f.endswith("PSG.edf")]

    for psg_file in psg_files:
        base = psg_file[:6]
        hyp_file = next((f for f in files if f.startswith(base) and f.endswith("Hypnogram.edf")), None)
        if not hyp_file:
            continue
        try:
            raw = mne.io.read_raw_edf(os.path.join(folder, psg_file), preload=True, verbose=False)
            ann = mne.read_annotations(os.path.join(folder, hyp_file))
        except:
            continue
        raw.set_annotations(ann)
        try:
            events, event_id = mne.events_from_annotations(raw)
            epochs = mne.Epochs(raw, events, event_id, tmin=0, tmax=30,
                                baseline=None, preload=True, verbose=False)
        except:
            continue
        sf = raw.info['sfreq']
        data = epochs.get_data()
        for i in range(len(data)):
            X.append(extract_features(data[i], sf))
            y.append(events[i][2])
    return np.array(X), np.array(y)

############################
# DATA
############################
X1, y1 = load_dataset(CASSETTE_PATH)
X2, y2 = load_dataset(TELEMETRY_PATH)

X = np.concatenate([X1, X2])
y = np.concatenate([y1, y2])

############################
# MODEL
############################
scaler = StandardScaler()
X = scaler.fit_transform(X)

model = RandomForestClassifier(n_estimators=300, class_weight="balanced")
scores = cross_val_score(model, X, y, cv=5)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

############################
# METRICS
############################
total = len(y)
sleep_eff = np.sum(y > 1)/total
transitions = sum(y[i] != y[i-1] for i in range(1, total))
sleep_latency = 110    
rem_latency = 132
waso = 0

stage_percentages = {STAGE_NAMES.get(s, str(s)): np.sum(y==s)/total for s in np.unique(y)}

score = 55   # tavo pateiktas sleep score
ai_comments = [
    "Low sleep efficiency (possible insomnia)",
    "Delayed REM sleep",
    "High sleep fragmentation"
]

############################
# PLOTS
############################
# Gal visai nebloga, gal geriau butu kad jie susijungtu, manau biski patobolinimo reiketu
stage_labels_lt = {
    0: "Budrumas",
    1: "Lengvas miegas N1",
    2: "Lengvas miegas N2",
    3: "Gilus miegas N3",
    4: "Gilus miegas N4",
    5: "REM miegas",
    6: "Nežinoma",
    7: "7"
}

stage_colors_lt = {
    "Budrumas": "#FF6347",
    "Lengvas miegas N1": "#FFD700",
    "Lengvas miegas N2": "#87CEFA",
    "Gilus miegas N3": "#1E90FF",
    "Gilus miegas N4": "#4169E1",
    "REM miegas": "#32CD32",
    "Nežinoma": "#D3D3D3",
    "7": "#FFA500"
}

time_hours = np.arange(len(y)) / 120

plt.figure(figsize=(15,4))
for code, label in stage_labels_lt.items():
    idx = [i for i, s in enumerate(y) if s==code]
    if idx:
        plt.scatter(time_hours[idx], [label]*len(idx), color=stage_colors_lt[label], s=20, label=label)
plt.xlabel("Laikas (valandos)")
plt.ylabel("Miego stadija")
plt.title("Hipnograma (scatter plot)")
plt.legend(bbox_to_anchor=(1.05,1), loc='upper left')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{PLOTS_FOLDER}/hypnogram_scatter.png")
plt.close()

#cia va kas noretusi is to auksciau esancio, tik sitas ne toks aiskus gal?
plt.figure(figsize=(15,4))
for code, label in stage_labels_lt.items():
    mask = np.array([1 if s==code else 0 for s in y])
    plt.fill_between(time_hours, 0, mask*(list(stage_labels_lt.keys()).index(code)+1),
                     step="mid", color=stage_colors_lt[label], alpha=0.6, label=label)
plt.yticks(range(len(stage_labels_lt)), list(stage_labels_lt.values()))
plt.xlabel("Laikas (valandos)")
plt.ylabel("Miego stadija")
plt.title("Hipnograma (filled area)")
plt.legend(bbox_to_anchor=(1.05,1), loc='upper left')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{PLOTS_FOLDER}/hypnogram_filled.png")
plt.close()

#kiekvienos stadijos atskirai
for code, label in stage_labels_lt.items():
    idx = [i for i, s in enumerate(y) if s==code]
    if not idx: 
        continue
    plt.figure(figsize=(15,2))
    plt.scatter(time_hours[idx], [label]*len(idx), color=stage_colors_lt[label], s=20)
    plt.xlabel("Laikas (valandos)")
    plt.ylabel("Miego stadija")
    plt.title(f"{label} hipnograma")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_FOLDER}/hypnogram_{label}.png")
    plt.close()

    
# Siek tiek netikslus...
df = pd.DataFrame({'time': time_hours, 'stage': y})
df['hour'] = df['time'].astype(int)
dominant_stage = df.groupby('hour')['stage'].agg(lambda x: x.value_counts().idxmax())

plt.figure(figsize=(12,4))
plt.bar(dominant_stage.index, [list(stage_labels_lt.keys()).index(s) for s in dominant_stage],
        color=[stage_colors_lt[stage_labels_lt[s]] for s in dominant_stage])
plt.xticks(dominant_stage.index, [f"{h}:00" for h in dominant_stage.index])
plt.yticks(range(len(stage_labels_lt)), list(stage_labels_lt.values()))
plt.xlabel("Laikas (valandos)")
plt.ylabel("Dominanti miego stadija")
plt.title("Hipnograma per valandas (dominanti stadija)")
plt.grid(alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(f"{PLOTS_FOLDER}/hypnogram_hourly.png")
plt.close()


#nezinau....
y_idx = np.array([list(stage_labels_lt.keys()).index(s) for s in y])
plt.figure(figsize=(15,4))
cmap = mcolors.ListedColormap([stage_colors_lt[label] for label in stage_labels_lt.values()])
plt.imshow([y_idx], aspect='auto', cmap=cmap)
plt.yticks([])
plt.xticks(np.arange(0, len(time_hours), 120), [f"{int(h)}:00" for h in np.arange(0, len(time_hours)/120, 1)])
plt.xlabel("Laikas (valandos)")
plt.title("Hipnograma (heatmap)")
handles = [Patch(color=color, label=label) for label, color in stage_colors_lt.items()]
plt.legend(handles=handles, bbox_to_anchor=(1.05,1), loc='upper left')
plt.tight_layout()
plt.savefig(f"{PLOTS_FOLDER}/hypnogram_heatmap.png")
plt.close()

#bendra, kaip sena
plt.figure(figsize=(12,4))
plt.plot(y[:500], color="purple")
plt.fill_between(range(len(y[:500])), y[:500], color="purple", alpha=0.3)
plt.gca().invert_yaxis()
plt.title("Miego stadijos per laiką")
plt.savefig(f"{PLOTS_FOLDER}/hypnogram.png")
plt.close()


#bar chart
plt.figure(figsize=(6,4))
colors_list = sns.color_palette("pastel", len(stage_percentages))
plt.bar(stage_percentages.keys(), [v*100 for v in stage_percentages.values()], color=colors_list)
plt.ylabel("Procentai (%)")
plt.title("Miego stadijų pasiskirstymas")
plt.savefig(f"{PLOTS_FOLDER}/stages.png")
plt.close()

#Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, cmap="coolwarm")
plt.title("Klasifikacijos matrica")
plt.savefig(f"{PLOTS_FOLDER}/cm.png")
plt.close()
#tokia sad biski
"""
############################
# PDF FUNCTIONS
############################
def apply_font(styles):
    for style in styles.byName.values():
        style.fontName = 'DejaVu'

def generate_patient_pdf():
    doc = SimpleDocTemplate("paciento_ataskaita.pdf")
    styles = getSampleStyleSheet()
    apply_font(styles)

    story = []
    story.append(Paragraph("Miego ataskaita", styles['Title']))
    story.append(Spacer(1, 10))

    # Sleep score
    story.append(Paragraph(f"Miego kokybė: {score}/100", styles['Heading2']))
    story.append(Paragraph("Jūsų miegas gali būti probleminis ir verta imtis priemonių:", styles['Normal']))
    
    # Plots
    story.append(Image(f"{PLOTS_FOLDER}/hypnogram.png", width=400, height=150))
    story.append(Image(f"{PLOTS_FOLDER}/stages.png", width=400, height=200))

    story.append(Spacer(1, 10))
    story.append(Paragraph("Pastebėjimai:", styles['Heading2']))
    for c in ai_comments:
        story.append(Paragraph(f"- {c}", styles['Normal']))

    # General advice
    story.append(Spacer(1, 10))
    story.append(Paragraph("Bendros rekomendacijos:", styles['Heading2']))
    story.append(Paragraph("• Laikykitės pastovaus miego grafiko", styles['Normal']))
    story.append(Paragraph("• Venkite ekranų prieš miegą", styles['Normal']))
    story.append(Paragraph("• Sumažinkite kofeino vartojimą", styles['Normal']))

    doc.build(story)

def generate_doctor_pdf():
    doc = SimpleDocTemplate("daktaro_ataskaita.pdf")
    styles = getSampleStyleSheet()
    apply_font(styles)

    story = []
    story.append(Paragraph("Klinikinė miego analizė", styles['Title']))
    story.append(Spacer(1,10))

    # Model results
    story.append(Paragraph("<b>Modelio rezultatai:</b>", styles['Heading2']))
    story.append(Paragraph(f"Accuracy (tikslumas): {acc:.4f} – modelio prognozių tikslumas testavimo rinkinyje.", styles['Normal']))
    story.append(Paragraph(f"CV (Cross-Validation vidurkis): {scores.mean():.4f} – modelio stabilumo rodiklis.", styles['Normal']))

    story.append(Spacer(1,10))
    story.append(Paragraph("<b>Miego rodikliai:</b>", styles['Heading2']))
    story.append(Paragraph(f"Sleep Efficiency: {sleep_eff:.2f}", styles['Normal']))
    story.append(Paragraph(f"Sleep Latency (epochs): {sleep_latency}", styles['Normal']))
    story.append(Paragraph(f"REM Latency (epochs): {rem_latency}", styles['Normal']))
    story.append(Paragraph(f"WASO: {waso}", styles['Normal']))
    story.append(Paragraph(f"Stage Transitions: {transitions}", styles['Normal']))

    # Plots
    story.append(Spacer(1,10))
    story.append(Paragraph("<b>Klasifikacijos matrica:</b>", styles['Heading2']))
    story.append(Image(f"{PLOTS_FOLDER}/cm.png", width=350, height=300))

    story.append(Spacer(1,10))
    story.append(Paragraph("<b>Miego stadijų pasiskirstymas:</b>", styles['Heading2']))
    story.append(Image(f"{PLOTS_FOLDER}/stages.png", width=400, height=200))

    story.append(Spacer(1,10))
    story.append(Paragraph("<b>Hipnograma:</b>", styles['Heading2']))
    story.append(Image(f"{PLOTS_FOLDER}/hypnogram.png", width=400, height=150))

    doc.build(story)
"""
############################
# RUN
############################
#generate_patient_pdf()
#generate_doctor_pdf()
print("DONE 🔥 PDF sukurtos!")