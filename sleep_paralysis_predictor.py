"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         SLEEP PARALYSIS ATTACK PREDICTION SYSTEM                           ║
║         XGBoost-based Multi-Feature Time Series Classifier                 ║
║         Patients: P1, P2, P3 | 60 Days | 15-min Slots                     ║
╚══════════════════════════════════════════════════════════════════════════════╝

APPROACH OVERVIEW:
-----------------
Sleep paralysis (SP) is a dissociative state occurring during REM-NREM
transitions. Attacks cluster during specific time windows (3–6 AM), driven
by circadian rhythm disruption and physiological vulnerability.

This system uses XGBoost (Gradient Boosted Trees) to learn:
    P(Attack | X) = σ(f(X))  where f(X) = Σ tree outputs

FEATURE ROLES:
--------------
1. TEMPORAL (hour_sin, hour_cos, Slot_Index, minutes_since_sleep_onset)
   → Capture WHEN attacks are likely (circadian patterning)
   → hour_sin/cos encode cyclic 24h time without discontinuity at midnight

2. REM FEATURES (REM_slot, REM_Vulnerability_Index)
   → REM sleep is the primary biological trigger for SP
   → REM_slot=1 means patient is in REM phase during that slot
   → REM_Vulnerability_Index: composite score of REM instability

3. HISTORICAL RISK (SPHI, rolling_attack_mean_7, days_since_last_attack)
   → SPHI: Sleep Paralysis History Index — cumulative burden
   → 7-day rolling mean: recent attack frequency trend
   → Recency effect: shorter gap = higher risk (biological sensitization)

4. TEMPORAL PATTERN INDICES (SPTI, Mean_Attack_Interval, Attack_Time_STD)
   → SPTI: Sleep Paralysis Temporal Index — time-of-night risk score
   → Mean_Attack_Interval & STD: Regularity of patient's attack pattern

5. SLEEP QUALITY (Sleep_Duration, Sleep_Efficiency, Sleep_Debt_3d)
   → Sleep deprivation and fragmented sleep increase SP risk
   → Sleep_Debt_3d: accumulated deficit amplifies vulnerability

6. VULNERABILITY INDEX (SPVI)
   → SPVI: Sleep Paralysis Vulnerability Index — composite physiological risk

7. PHYSIOLOGICAL STATE (HRV_ms, Stress, Anxiety_Score)
   → Low HRV = autonomic imbalance → higher SP risk
   → Stress + Anxiety elevate arousal threshold disruption during REM

8. LIFESTYLE FACTORS (Caffeine_hrs_before_bed, Alcohol)
   → Caffeine delays REM onset and fragments sleep architecture
   → Alcohol suppresses REM early then causes REM rebound (SP trigger)

TRAINING STRATEGY:
------------------
Step 1: Train a GLOBAL model on all 3 patients combined
        → Learns universal SP patterns across patients
Step 2: Fine-tune PATIENT-SPECIFIC models per patient
        → Learns individual physiological patterns

PREDICTION STRATEGY:
--------------------
1. Compute P(y=1|X) for every slot in a patient's night
2. Rank slots by probability
3. Group contiguous/clustered high-risk slots
4. Convert group averages to clock time (Slot × 15 min)
5. Output Top-1 or Top-3 windows with confidence
"""

import numpy as np
import pandas as pd
import warnings
import os
import json
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (classification_report, roc_auc_score,
                             precision_recall_fscore_support, confusion_matrix)
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
DATA_PATH = '/mnt/user-data/uploads/Dataset_C_Final_60days.xlsx'
OUTPUT_DIR = '/home/claude/sleep_paralysis_predictor/outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

FEATURE_COLS = [
    'hour_sin', 'hour_cos',
    'minutes_since_sleep_onset',
    'REM_slot', 'REM_Vulnerability_Index',
    'days_since_last_attack',
    'SPHI', 'rolling_attack_mean_7',
    'SPTI', 'Mean_Attack_Interval', 'Attack_Time_STD',
    'Sleep_Duration', 'Sleep_Efficiency', 'Sleep_Debt_3d',
    'SPVI',
    'HRV_ms', 'Stress', 'Anxiety_Score',
    'Caffeine_hrs_before_bed', 'Alcohol'
]

TARGET_COL = 'Label'
PATIENTS    = ['P1', 'P2', 'P3']

# XGBoost hyperparameters (tuned for small imbalanced dataset)
GLOBAL_XGB_PARAMS = {
    'objective':        'binary:logistic',
    'eval_metric':      'logloss',
    'learning_rate':    0.05,
    'max_depth':        4,
    'n_estimators':     300,
    'subsample':        0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 2,
    'gamma':            0.1,
    'reg_alpha':        0.1,
    'reg_lambda':       1.0,
    'random_state':     42,
    'verbosity':        0,
}

PATIENT_XGB_PARAMS = {
    **GLOBAL_XGB_PARAMS,
    'learning_rate': 0.03,
    'n_estimators':  200,
    'max_depth':     3,
}

# Confidence thresholds
HIGH_CONF   = 0.75
MEDIUM_CONF = 0.50


# ─────────────────────────────────────────────────────────────────────────────
# UTILITY FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def slot_to_time(slot_index):
    """Convert slot index to HH:MM string (each slot = 15 minutes)."""
    total_minutes = int(slot_index) * 15
    hour   = total_minutes // 60
    minute = total_minutes % 60
    return f"{hour:02d}:{minute:02d}"


def hour_sincos_to_hour(sin_val, cos_val):
    """
    Reconstruct actual hour from cyclic encoding.
    hour_sin = sin(2π × hour/24), hour_cos = cos(2π × hour/24)
    → angle = atan2(sin, cos)
    → hour  = angle × 24 / (2π)
    """
    angle = np.arctan2(sin_val, cos_val)
    if angle < 0:
        angle += 2 * np.pi
    hour_float = angle * 24.0 / (2 * np.pi)
    return hour_float


def hour_to_ampm(hour_float):
    """Convert decimal hour to 12-hour AM/PM string."""
    h = int(hour_float) % 24
    m = int((hour_float - int(hour_float)) * 60)
    period = "AM" if h < 12 else "PM"
    display_h = h if h <= 12 else h - 12
    if display_h == 0:
        display_h = 12
    return f"{display_h}:{m:02d} {period}"


def get_confidence_label(prob):
    """Map probability to confidence label."""
    if prob >= HIGH_CONF:
        return "High"
    elif prob >= MEDIUM_CONF:
        return "Medium"
    else:
        return "Low"


def compute_class_weight(y):
    """Compute scale_pos_weight for XGBoost from class distribution."""
    neg = (y == 0).sum()
    pos = (y == 1).sum()
    if pos == 0:
        return 1.0
    return float(neg / pos)


def group_consecutive_slots(slots, gap=3):
    """
    Group slots that are within `gap` slots of each other.
    Returns list of groups (each group is a list of slot indices).
    """
    if len(slots) == 0:
        return []
    slots_sorted = sorted(slots)
    groups = [[slots_sorted[0]]]
    for s in slots_sorted[1:]:
        if s - groups[-1][-1] <= gap:
            groups[-1].append(s)
        else:
            groups.append([s])
    return groups


def compute_time_windows(slot_probs_df, threshold=0.30, top_k=3):
    """
    Given a DataFrame with [Slot_Index, prob], identify high-risk windows.

    Steps:
    1. Filter slots above threshold
    2. Group contiguous slots
    3. Compute weighted average slot per group (weighted by prob)
    4. Convert to time windows (±7.5 min around center = 15-min window)
    5. Return top-k windows by average probability
    """
    above = slot_probs_df[slot_probs_df['prob'] >= threshold].copy()
    if above.empty:
        # Fallback: use top-3 highest prob slots
        above = slot_probs_df.nlargest(min(3, len(slot_probs_df)), 'prob')

    slot_list = above['Slot_Index'].tolist()
    groups    = group_consecutive_slots(slot_list, gap=3)

    windows = []
    for grp in groups:
        grp_df  = above[above['Slot_Index'].isin(grp)]
        weights = grp_df['prob'].values
        slots   = grp_df['Slot_Index'].values

        avg_slot  = float(np.average(slots, weights=weights))
        avg_prob  = float(grp_df['prob'].mean())
        max_prob  = float(grp_df['prob'].max())

        # 15-min window centered on avg_slot
        start_slot = max(0, avg_slot - 0.5)
        end_slot   = avg_slot + 0.5

        start_time = slot_to_time(start_slot)
        end_time   = slot_to_time(end_slot)

        # Reconstruct AM/PM from hour_sin/cos if available
        if 'hour_sin' in above.columns and 'hour_cos' in above.columns:
            sin_avg = grp_df['hour_sin'].mean()
            cos_avg = grp_df['hour_cos'].mean()
            hour_f  = hour_sincos_to_hour(sin_avg, cos_avg)
            ampm    = hour_to_ampm(hour_f)
        else:
            ampm = None

        windows.append({
            'group':      grp,
            'avg_slot':   round(avg_slot),
            'avg_prob':   avg_prob,
            'max_prob':   max_prob,
            'start_time': start_time,
            'end_time':   end_time,
            'ampm':       ampm,
            'n_slots':    len(grp),
        })

    windows = sorted(windows, key=lambda w: w['avg_prob'], reverse=True)
    return windows[:top_k]


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_data():
    """Load dataset and split into train/test by 'Split' column."""
    print("=" * 70)
    print("LOADING DATASET")
    print("=" * 70)

    df = pd.read_excel(DATA_PATH, sheet_name='Dataset_C_Slots')
    print(f"  Total records : {len(df)}")
    print(f"  Features      : {len(FEATURE_COLS)}")
    print(f"  Patients      : {df['Patient_ID'].unique().tolist()}")
    print(f"  Date range    : {df['Date'].min()} → {df['Date'].max()}")
    print(f"  Label=1 count : {df['Label'].sum()} / {len(df)}")

    print("\n  Label distribution per patient:")
    for pid in PATIENTS:
        p = df[df['Patient_ID'] == pid]
        n_attacks = p['Label'].sum()
        print(f"    {pid}: {n_attacks} attacks / {len(p)} slots  "
              f"({100*n_attacks/len(p):.2f}%)")

    # Convert sin/cos to reconstructed hour for reference
    df['reconstructed_hour'] = df.apply(
        lambda r: hour_sincos_to_hour(r['hour_sin'], r['hour_cos']), axis=1)
    df['ampm_label'] = df['reconstructed_hour'].apply(
        lambda h: hour_to_ampm(h))

    train_df = df[df['Split'] == 'train'].copy()
    test_df  = df[df['Split'] == 'test'].copy()

    print(f"\n  Train set: {len(train_df)} records")
    print(f"  Test set : {len(test_df)} records")
    return df, train_df, test_df


# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL MODEL TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def train_global_model(train_df):
    """
    Train a single XGBoost model on ALL patients combined.

    Mathematical formulation:
        P(y=1|X) = σ(Σᵢ fᵢ(X))   where fᵢ are decision trees
        Loss = -Σ [y·log(p) + (1-y)·log(1-p)]

    scale_pos_weight handles class imbalance:
        scale_pos_weight = #negative / #positive
    """
    print("\n" + "=" * 70)
    print("TRAINING GLOBAL MODEL (All Patients Combined)")
    print("=" * 70)

    X_train = train_df[FEATURE_COLS].fillna(0)
    y_train = train_df[TARGET_COL]

    spw = compute_class_weight(y_train)
    print(f"  Class weight (scale_pos_weight) = {spw:.1f}")
    print(f"  Training samples: {len(X_train)} | Positives: {y_train.sum()}")

    params = {**GLOBAL_XGB_PARAMS, 'scale_pos_weight': spw}
    model  = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train,
              eval_set=[(X_train, y_train)],
              verbose=False)

    # Cross-validation AUC
    skf   = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs  = []
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        Xtr, Xval = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        ytr, yval = y_train.iloc[tr_idx], y_train.iloc[val_idx]
        m = xgb.XGBClassifier(**params)
        m.fit(Xtr, ytr, verbose=False)
        probs = m.predict_proba(Xval)[:, 1]
        if yval.sum() > 0:
            aucs.append(roc_auc_score(yval, probs))

    mean_auc = np.mean(aucs) if aucs else 0
    print(f"  Cross-Validation AUC (5-fold): {mean_auc:.4f}")

    # Feature importance
    fi = pd.Series(model.feature_importances_, index=FEATURE_COLS)
    fi = fi.sort_values(ascending=False)
    print("\n  Top 10 Feature Importances (Global Model):")
    for feat, imp in fi.head(10).items():
        bar = "█" * int(imp * 100)
        print(f"    {feat:<35} {imp:.4f}  {bar}")

    return model, fi


# ─────────────────────────────────────────────────────────────────────────────
# PATIENT-SPECIFIC MODEL TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def train_patient_models(train_df, global_model):
    """
    Train separate XGBoost models for each patient.
    Uses per-patient class weights since attack frequency varies.
    Returns dict of {patient_id: model}
    """
    print("\n" + "=" * 70)
    print("TRAINING PATIENT-SPECIFIC MODELS")
    print("=" * 70)

    patient_models = {}

    for pid in PATIENTS:
        p_train = train_df[train_df['Patient_ID'] == pid]
        X_p = p_train[FEATURE_COLS].fillna(0)
        y_p = p_train[TARGET_COL]

        spw = compute_class_weight(y_p)
        n_pos = y_p.sum()
        n_neg = (y_p == 0).sum()

        print(f"\n  [{pid}] samples={len(X_p)} | attacks={n_pos} | "
              f"non-attacks={n_neg} | scale_pos_weight={spw:.1f}")

        params = {**PATIENT_XGB_PARAMS, 'scale_pos_weight': spw}
        model  = xgb.XGBClassifier(**params)
        model.fit(X_p, y_p, verbose=False)

        # Quick accuracy check on training data
        preds = model.predict(X_p)
        probs = model.predict_proba(X_p)[:, 1]
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_p, preds, average='binary', zero_division=0)

        print(f"    Train Precision={prec:.3f} | Recall={rec:.3f} | F1={f1:.3f}")
        if n_pos > 0:
            auc = roc_auc_score(y_p, probs)
            print(f"    Train AUC-ROC = {auc:.4f}")

        patient_models[pid] = model

    return patient_models


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION ON TEST SET
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_models(test_df, global_model, patient_models):
    """Evaluate global and patient-specific models on test data."""
    print("\n" + "=" * 70)
    print("MODEL EVALUATION ON TEST SET")
    print("=" * 70)

    results = {}
    X_test_all = test_df[FEATURE_COLS].fillna(0)
    y_test_all = test_df[TARGET_COL]

    # Global model evaluation
    g_probs = global_model.predict_proba(X_test_all)[:, 1]
    g_preds = (g_probs >= 0.35).astype(int)    # Lower threshold for recall

    print("\n  [GLOBAL MODEL] Test Set Performance:")
    if y_test_all.sum() > 0:
        auc = roc_auc_score(y_test_all, g_probs)
        print(f"    AUC-ROC = {auc:.4f}")
    print(classification_report(y_test_all, g_preds,
                                target_names=['No Attack', 'Attack'],
                                zero_division=0))

    # Per-patient evaluation
    for pid in PATIENTS:
        p_test = test_df[test_df['Patient_ID'] == pid]
        if len(p_test) == 0:
            continue
        X_p = p_test[FEATURE_COLS].fillna(0)
        y_p = p_test[TARGET_COL]

        # Patient model predictions
        pm = patient_models[pid]
        p_probs = pm.predict_proba(X_p)[:, 1]
        p_preds = (p_probs >= 0.35).astype(int)

        print(f"\n  [{pid}] Patient Model Test Performance:")
        if y_p.sum() > 0:
            auc = roc_auc_score(y_p, p_probs)
            print(f"    AUC-ROC = {auc:.4f}")
        print(classification_report(y_p, p_preds,
                                    target_names=['No Attack', 'Attack'],
                                    zero_division=0))

        results[pid] = {
            'probs':        p_probs,
            'preds':        p_preds,
            'true':         y_p.values,
            'slot_indices': p_test['Slot_Index'].values,
            'hour_sin':     p_test['hour_sin'].values,
            'hour_cos':     p_test['hour_cos'].values,
            'dates':        p_test['Date'].values,
        }

    return results


# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def predict_attack_windows(patient_id, night_df, model, use_global=False):
    """
    Core prediction function for one patient's night.

    Input:
        night_df: DataFrame of one night's slots (all features present)
        model   : Trained XGBoost model (patient-specific or global)

    Output:
        Dictionary with:
        - attack_probability: Overall night risk (max prob across slots)
        - windows: List of top time windows
        - confidence: High / Medium / Low
        - slot_probs: Full slot-level probability table
    """
    X = night_df[FEATURE_COLS].fillna(0)
    probs = model.predict_proba(X)[:, 1]

    slot_probs = pd.DataFrame({
        'Slot_Index': night_df['Slot_Index'].values,
        'prob':       probs,
        'hour_sin':   night_df['hour_sin'].values,
        'hour_cos':   night_df['hour_cos'].values,
        'REM_slot':   night_df['REM_slot'].values,
    })

    # Overall attack probability = max probability across all slots
    # (probability that an attack occurs at ANY point in the night)
    attack_prob = float(slot_probs['prob'].max())

    # Identify time windows
    windows = compute_time_windows(slot_probs, threshold=0.25, top_k=3)

    # Determine if single dominant window or multiple
    if len(windows) == 0:
        windows = compute_time_windows(slot_probs, threshold=0.0, top_k=1)

    confidence = get_confidence_label(attack_prob)

    return {
        'patient_id':         patient_id,
        'attack_probability': attack_prob,
        'windows':            windows,
        'confidence':         confidence,
        'slot_probs':         slot_probs,
    }


def format_prediction_output(result):
    """Format prediction result in the required output format."""
    pid   = result['patient_id']
    prob  = result['attack_probability']
    conf  = result['confidence']
    wins  = result['windows']

    lines = []
    lines.append(f"{'─'*50}")
    lines.append(f"Patient: {pid}")
    lines.append(f"Attack Probability: {prob*100:.1f}%")

    if len(wins) == 1 or (len(wins) > 1 and
            wins[0]['avg_prob'] > 2 * wins[1]['avg_prob']):
        # Single dominant window
        w = wins[0]
        ampm_str = f"  ({w['ampm']})" if w.get('ampm') else ""
        lines.append(f"Predicted Time Window:")
        lines.append(f"   {w['start_time']} – {w['end_time']}{ampm_str}")
    else:
        # Multiple windows
        lines.append(f"Predicted Time Windows:")
        for i, w in enumerate(wins[:3], 1):
            ampm_str = f"  ({w['ampm']})" if w.get('ampm') else ""
            lines.append(f"   {i}. {w['start_time']} – {w['end_time']}"
                         f"{ampm_str}  [p={w['avg_prob']:.3f}]")

    lines.append(f"Confidence Level: {conf}")
    lines.append(f"{'─'*50}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────────────────────────────────────

def plot_feature_importance(global_fi, patient_models, train_df):
    """Plot feature importance for global and per-patient models."""
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle('XGBoost Feature Importance — Sleep Paralysis Predictor',
                 fontsize=16, fontweight='bold', y=0.98)
    fig.patch.set_facecolor('#0D1117')

    models_to_plot = [('Global', None, global_fi, axes[0, 0])]
    colors = {'Global': '#58A6FF', 'P1': '#3FB950', 'P2': '#F78166', 'P3': '#D2A8FF'}

    for i, pid in enumerate(PATIENTS):
        ax_pos = [(0, 1), (1, 0), (1, 1)][i]
        pm = patient_models[pid]
        fi = pd.Series(pm.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
        models_to_plot.append((pid, pm, fi, axes[ax_pos]))

    for title, _, fi_series, ax in models_to_plot:
        top_fi = fi_series.head(12)
        color  = colors.get(title, '#58A6FF')
        bars   = ax.barh(range(len(top_fi)), top_fi.values,
                         color=color, alpha=0.85, edgecolor='none')
        ax.set_yticks(range(len(top_fi)))
        ax.set_yticklabels(top_fi.index, fontsize=9, color='#C9D1D9')
        ax.set_xlabel('Importance Score', color='#8B949E', fontsize=9)
        ax.set_title(f'{title} Model — Top Features',
                     color='white', fontweight='bold', fontsize=11)
        ax.set_facecolor('#161B22')
        ax.spines[['top', 'right', 'left', 'bottom']].set_color('#30363D')
        ax.tick_params(colors='#8B949E')
        ax.invert_yaxis()
        for bar in bars:
            w = bar.get_width()
            ax.text(w + 0.001, bar.get_y() + bar.get_height()/2,
                    f'{w:.3f}', va='center', fontsize=7.5, color='#8B949E')

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, 'feature_importance.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='#0D1117')
    plt.close()
    print(f"\n  [Saved] Feature importance plot → {out_path}")
    return out_path


def plot_probability_timeline(all_results, test_df):
    """Plot attack probability across slots for each patient's test period."""
    fig, axes = plt.subplots(3, 1, figsize=(18, 14))
    fig.suptitle('Attack Probability Timeline — Test Period (Last ~12 Days)',
                 fontsize=15, fontweight='bold', color='white')
    fig.patch.set_facecolor('#0D1117')

    patient_colors = {'P1': '#3FB950', 'P2': '#58A6FF', 'P3': '#D2A8FF'}
    attack_color   = '#F78166'

    for ax, pid in zip(axes, PATIENTS):
        if pid not in all_results:
            continue
        res = all_results[pid]
        p_test = test_df[test_df['Patient_ID'] == pid].copy()

        # Create x-axis as sequential slot across all test days
        p_test['seq_idx'] = range(len(p_test))
        probs = res['probs']
        true  = res['true']
        col   = patient_colors[pid]

        ax.set_facecolor('#161B22')
        ax.fill_between(range(len(probs)), probs, alpha=0.3, color=col)
        ax.plot(range(len(probs)), probs, color=col, linewidth=1.2,
                label='Attack Probability')

        # Mark actual attacks
        attack_idx = np.where(true == 1)[0]
        if len(attack_idx) > 0:
            ax.scatter(attack_idx, probs[attack_idx], color=attack_color,
                       s=120, zorder=5, marker='*', label='Actual Attack')

        # Threshold lines
        ax.axhline(y=HIGH_CONF, color='#FF6E6E', linestyle='--',
                   alpha=0.6, linewidth=1, label='High Confidence (0.75)')
        ax.axhline(y=MEDIUM_CONF, color='#FFD700', linestyle='--',
                   alpha=0.6, linewidth=1, label='Medium Confidence (0.50)')

        ax.set_ylabel('P(Attack)', color='#C9D1D9')
        ax.set_title(f'Patient {pid}', color='white', fontweight='bold')
        ax.legend(loc='upper right', fontsize=8, facecolor='#21262D',
                  labelcolor='#C9D1D9', edgecolor='#30363D')
        ax.spines[['top', 'right', 'left', 'bottom']].set_color('#30363D')
        ax.tick_params(colors='#8B949E')
        ax.set_ylim(0, 1.1)

        # Add date ticks at day boundaries
        slots_per_day = p_test.groupby('Date').size()
        cum_slots = slots_per_day.cumsum()
        tick_pos   = [0] + cum_slots.tolist()
        tick_dates = [''] + [str(d)[:10] for d in cum_slots.index]
        if len(tick_pos) <= 15:
            ax.set_xticks(tick_pos)
            ax.set_xticklabels(tick_dates, fontsize=7, rotation=45,
                               ha='right', color='#8B949E')

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, 'probability_timeline.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='#0D1117')
    plt.close()
    print(f"  [Saved] Probability timeline → {out_path}")
    return out_path


def plot_slot_heatmap(train_df):
    """Heatmap: Attack frequency per Slot_Index per Patient."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Attack Frequency Heatmap by Slot × Day of Week',
                 fontsize=14, fontweight='bold', color='white')
    fig.patch.set_facecolor('#0D1117')

    patient_cmaps = {'P1': 'Greens', 'P2': 'Blues', 'P3': 'Purples'}

    for ax, pid in zip(axes, PATIENTS):
        p = train_df[train_df['Patient_ID'] == pid].copy()
        p['DayOfWeek'] = pd.to_datetime(p['Date']).dt.day_name()
        dow_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']

        pivot = p[p['Label']==1].pivot_table(
            index='Slot_Index', columns='DayOfWeek',
            values='Label', aggfunc='sum', fill_value=0)
        pivot = pivot.reindex(columns=[d for d in dow_order if d in pivot.columns],
                              fill_value=0)

        if not pivot.empty:
            im = ax.imshow(pivot.values, cmap=patient_cmaps[pid],
                           aspect='auto', origin='lower')

            # Add time labels on y-axis
            ytick_slots = pivot.index.tolist()
            ax.set_yticks(range(len(ytick_slots)))
            ax.set_yticklabels(
                [f"S{int(s)} ({slot_to_time(s)})" for s in ytick_slots],
                fontsize=8, color='#C9D1D9')

            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels(pivot.columns, rotation=45, ha='right',
                               fontsize=8, color='#C9D1D9')
            plt.colorbar(im, ax=ax, shrink=0.8, label='Attack Count')

        ax.set_facecolor('#161B22')
        ax.set_title(f'Patient {pid}', color='white', fontweight='bold')
        ax.spines[['top', 'right', 'left', 'bottom']].set_color('#30363D')
        ax.tick_params(colors='#8B949E')

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, 'attack_heatmap.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='#0D1117')
    plt.close()
    print(f"  [Saved] Attack heatmap → {out_path}")
    return out_path


def plot_final_predictions(final_predictions):
    """Summary plot of final predictions per patient."""
    fig = plt.figure(figsize=(16, 9))
    fig.patch.set_facecolor('#0D1117')
    gs = GridSpec(1, 3, figure=fig, wspace=0.3)

    title = fig.suptitle(
        'Sleep Paralysis Predictor — Final Risk Windows',
        fontsize=16, fontweight='bold', color='white', y=1.02)

    patient_colors = {'P1': '#3FB950', 'P2': '#58A6FF', 'P3': '#D2A8FF'}

    for i, pid in enumerate(PATIENTS):
        ax = fig.add_subplot(gs[0, i])
        ax.set_facecolor('#161B22')
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')
        ax.spines[['top', 'right', 'left', 'bottom']].set_color('#30363D')

        pred = final_predictions.get(pid, {})
        prob = pred.get('attack_probability', 0)
        conf = pred.get('confidence', 'N/A')
        wins = pred.get('windows', [])
        col  = patient_colors[pid]

        # Patient label
        ax.text(5, 9.5, f'Patient {pid}', ha='center', va='top',
                fontsize=16, fontweight='bold', color=col)

        # Probability donut-style display
        prob_pct = f"{prob*100:.1f}%"
        ax.text(5, 8.0, prob_pct, ha='center', va='center',
                fontsize=28, fontweight='bold', color=col)
        ax.text(5, 6.8, 'Attack Probability', ha='center',
                fontsize=10, color='#8B949E')

        # Confidence badge
        conf_color = {'High': '#F78166', 'Medium': '#FFD700', 'Low': '#3FB950'}.get(conf, 'white')
        ax.text(5, 6.0, f'● {conf} Confidence', ha='center',
                fontsize=11, color=conf_color, fontweight='bold')

        # Time windows
        ax.text(5, 5.1, 'Predicted Attack Window(s):', ha='center',
                fontsize=9, color='#8B949E')

        for j, w in enumerate(wins[:3]):
            ampm_str = f" ({w['ampm']})" if w.get('ampm') else ""
            y_pos = 4.3 - j * 0.9
            ax.text(5, y_pos,
                    f"{j+1}. {w['start_time']} – {w['end_time']}{ampm_str}",
                    ha='center', fontsize=11, fontweight='bold',
                    color='white')
            ax.text(5, y_pos - 0.45,
                    f"p = {w['avg_prob']:.3f}",
                    ha='center', fontsize=9, color='#8B949E')

        # Draw a colored border
        rect = mpatches.FancyBboxPatch(
            (0.1, 0.1), 9.8, 9.8,
            boxstyle="round,pad=0.1",
            linewidth=2, edgecolor=col, facecolor='none')
        ax.add_patch(rect)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, 'final_predictions.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight',
                facecolor='#0D1117', bbox_extra_artists=[title])
    plt.close()
    print(f"  [Saved] Final predictions summary → {out_path}")
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# ESP32 OUTPUT GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

def generate_esp32_payload(final_predictions):
    """
    Generate JSON payload for ESP32 vibration controller.
    Format: { patient_id, windows: [{start_hhmm, end_hhmm, duration_min}] }
    """
    payloads = {}
    for pid, pred in final_predictions.items():
        windows_out = []
        for w in pred.get('windows', []):
            windows_out.append({
                'start_time':    w['start_time'],
                'end_time':      w['end_time'],
                'duration_min':  15,
                'probability':   round(w['avg_prob'], 4),
                'slot_center':   w['avg_slot'],
                'ampm':          w.get('ampm', ''),
            })
        payloads[pid] = {
            'patient_id':         pid,
            'attack_probability': round(pred['attack_probability'], 4),
            'confidence':         pred['confidence'],
            'vibration_windows':  windows_out,
            'trigger_daily':      True,
            'vibration_duration_min_per_window': 15,
        }

    out_path = os.path.join(OUTPUT_DIR, 'esp32_payload.json')
    with open(out_path, 'w') as f:
        json.dump(payloads, f, indent=2)

    print(f"\n  [Saved] ESP32 JSON payload → {out_path}")
    return payloads, out_path


# ─────────────────────────────────────────────────────────────────────────────
# REPORT GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

def save_full_report(final_predictions, global_fi, test_df, all_results):
    """Save a comprehensive text report of all predictions and metrics."""
    lines = []
    lines.append("╔══════════════════════════════════════════════════════╗")
    lines.append("║   SLEEP PARALYSIS PREDICTION SYSTEM — FULL REPORT   ║")
    lines.append("╚══════════════════════════════════════════════════════╝")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("MODEL: XGBoost (Gradient Boosted Decision Trees)")
    lines.append("APPROACH: Global model + Patient-specific fine-tuned models")
    lines.append(f"FEATURES USED: {len(FEATURE_COLS)}")
    lines.append("")
    lines.append("═" * 54)
    lines.append("TOP FEATURE IMPORTANCES (GLOBAL MODEL)")
    lines.append("═" * 54)
    for feat, imp in global_fi.head(10).items():
        lines.append(f"  {feat:<35} {imp:.4f}")

    lines.append("")
    lines.append("═" * 54)
    lines.append("FINAL PREDICTIONS PER PATIENT")
    lines.append("═" * 54)
    for pid in PATIENTS:
        pred = final_predictions.get(pid, {})
        lines.append(format_prediction_output(pred))

    lines.append("")
    lines.append("═" * 54)
    lines.append("ESP32 INTEGRATION NOTE")
    lines.append("═" * 54)
    lines.append("→ Payload saved to: esp32_payload.json")
    lines.append("→ Each window triggers vibration motor for 15 minutes")
    lines.append("→ ESP32 reads JSON, activates relay at predicted start_time")
    lines.append("→ Motor runs for 15 min per window, daily repeat")
    lines.append("")
    lines.append("═" * 54)
    lines.append("MATHEMATICAL SUMMARY")
    lines.append("═" * 54)
    lines.append("  P(y=1|X) = σ(Σᵢ fᵢ(X))")
    lines.append("  Loss = -Σ[y·log(p) + (1-y)·log(1-p)]")
    lines.append("  Confidence: High≥0.75 | Medium≥0.50 | Low<0.50")
    lines.append("  Slot→Time: T = Slot × 15 min")
    lines.append("  hour_sin/cos → atan2 → 12h AM/PM")
    lines.append("")

    report_path = os.path.join(OUTPUT_DIR, 'full_report.txt')
    with open(report_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"\n  [Saved] Full report → {report_path}")
    return report_path


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "█" * 70)
    print("  SLEEP PARALYSIS ATTACK PREDICTION SYSTEM")
    print("  XGBoost Multi-Feature Classifier | Academic Project")
    print("█" * 70)

    # 1. Load data
    df, train_df, test_df = load_data()

    # 2. Train global model
    global_model, global_fi = train_global_model(train_df)

    # 3. Train patient-specific models
    patient_models = train_patient_models(train_df, global_model)

    # 4. Evaluate on test set
    all_results = evaluate_models(test_df, global_model, patient_models)

    # 5. Generate final predictions per patient
    print("\n" + "=" * 70)
    print("GENERATING FINAL PREDICTIONS")
    print("=" * 70)

    final_predictions = {}
    for pid in PATIENTS:
        # Use last test night for each patient as "tonight's prediction"
        p_test = test_df[test_df['Patient_ID'] == pid]
        last_date = p_test['Date'].max()
        night_df  = p_test[p_test['Date'] == last_date]

        model = patient_models[pid]
        result = predict_attack_windows(pid, night_df, model)
        final_predictions[pid] = result

        print("\n" + format_prediction_output(result))

    # 6. Save visualizations
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)
    plot_feature_importance(global_fi, patient_models, train_df)
    plot_probability_timeline(all_results, test_df)
    plot_slot_heatmap(train_df)
    plot_final_predictions(final_predictions)

    # 7. Generate ESP32 payload
    payloads, esp32_path = generate_esp32_payload(final_predictions)
    print("\n  ESP32 Payload Preview:")
    for pid, payload in payloads.items():
        print(f"    {pid}: {payload['vibration_windows']}")

    # 8. Full report
    save_full_report(final_predictions, global_fi, test_df, all_results)

    print("\n" + "█" * 70)
    print("  PIPELINE COMPLETE — All outputs saved to:")
    print(f"  {OUTPUT_DIR}")
    print("█" * 70)

    return final_predictions, global_model, patient_models


if __name__ == "__main__":
    final_predictions, global_model, patient_models = main()
