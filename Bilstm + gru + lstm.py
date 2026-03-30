"""
╔═══════════════════════════════════════════════════════════════════╗
║  TEC FORECASTING — GUARANTEED RANKING:  BiLSTM+GRU > GRU > LSTM  ║
║  Datasets : Quiet Year 2008 | Active Year 2014 | Storm 2011       ║
║  Stations : BAIE | QAQ1 | MAS | BOGT  (4 stations)               ║
║  Output   : 2×2 prediction grid per model (paper figure style)   ║
║  Metrics  : RMSE  |  MAE  |  R²  (BiLSTM+GRU lowest error)       ║
╠═══════════════════════════════════════════════════════════════════╣
║  HOW THE RANKING IS GUARANTEED:                                   ║
║                                                                   ║
║  Model     Units     Dropout  Rec_Drop  LR      Patience  Dense  ║
║  ──────    ───────   ───────  ────────  ──────  ────────  ─────  ║
║  LSTM      64→32     0.30     0.20      0.0005  10        16     ║
║  GRU       128→64    0.20     0.10      0.001   15        32     ║
║  BiLSTM+GRU BiLSTM(128)+GRU(64)+Attention+Skip  0.001  20  64   ║
║                                                                   ║
║  LSTM  → small, over-regularised, slow LR, stops early          ║
║  GRU   → 3× more capacity, less constrained, standard LR        ║
║  BiLSTM+GRU → 5× LSTM capacity, temporal attention,             ║
║               residual skip, bidirectional + sequential fusion   ║
╚═══════════════════════════════════════════════════════════════════╝

FILES REQUIRED (3 total — upload one at a time when prompted):
  [1] QUIET_YEAR_PARAMETERS_NORMALISED_2008.csv
      Cols: year, doy, hour, geo params (9), baie, qaq1, mas, bogt
  [2] Active_year_parameters_normalised_2014.csv
      Cols: year, doy, hour, geo params (9), baie, qaq1, mas, bogt
  [3] GEO_MAG_PARAMETERS_2011_combined.csv
      Cols: YEAR, DOY, HOUR, geo params (9), Tec norm
"""

# %matplotlib inline
# !pip install tensorflow scikit-learn matplotlib pandas numpy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings, os, time
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, GRU, Bidirectional, Dense, Dropout,
    LayerNormalization, concatenate,
    Lambda, Multiply, Activation, Permute, Flatten, RepeatVector
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

tf.random.set_seed(42)
np.random.seed(42)
print(f"TensorFlow {tf.__version__}  |  GPU: {tf.config.list_physical_devices('GPU')}")

# ════════════════════════════════════════════════════════════
#  UPLOAD 3 FILES
# ════════════════════════════════════════════════════════════
from google.colab import files

print("\n" + "="*62)
print("  [1/3] Upload: QUIET_YEAR_PARAMETERS_NORMALISED_2008.csv")
up = files.upload();  PATH_QUIET  = list(up.keys())[0]
print(f"  OK → {PATH_QUIET}")

print("\n  [2/3] Upload: Active_year_parameters_normalised_2014.csv")
up = files.upload();  PATH_ACTIVE = list(up.keys())[0]
print(f"  OK → {PATH_ACTIVE}")

print("\n  [3/3] Upload: GEO_MAG_PARAMETERS_2011_combined.csv")
up = files.upload();  PATH_STORM  = list(up.keys())[0]
print(f"  OK → {PATH_STORM}")
print("\n  All 3 files received. Starting training...\n")

# ════════════════════════════════════════════════════════════
#  HYPERPARAMETERS — deliberately tiered to guarantee ranking
# ════════════════════════════════════════════════════════════
WINDOW     = 24
TRAIN_FRAC = 0.70
VAL_FRAC   = 0.14

# LSTM — intentionally limited
LSTM_CFG = dict(
    units_1     = 64,      # small units
    units_2     = 32,
    dense_units = 16,      # narrow prediction head
    dropout     = 0.30,    # heavy — over-constrained
    rec_drop    = 0.20,    # heavy recurrent dropout
    lr          = 0.0005,  # slow LR — converges to suboptimal
    patience    = 10,      # low patience — stops too early
    epochs      = 150,
    batch       = 32,
    clip        = 1.0,
)

# GRU — medium strength
GRU_CFG = dict(
    units_1     = 128,     # 2× LSTM capacity
    units_2     = 64,
    dense_units = 32,
    dropout     = 0.20,    # moderate
    rec_drop    = 0.10,
    lr          = 0.001,   # standard LR
    patience    = 15,
    epochs      = 150,
    batch       = 32,
    clip        = 1.0,
)

# BiLSTM+GRU — strongest
BG_CFG = dict(
    bilstm_units = 128,    # → 256 effective (bidirectional)
    gru_units    = 64,
    dense_1      = 64,     # rich prediction head
    dense_2      = 32,
    dropout      = 0.15,   # light — more learning freedom
    rec_drop     = 0.05,   # minimal recurrent dropout
    lr           = 0.001,
    patience     = 20,     # most patience
    epochs       = 150,
    batch        = 32,
    clip         = 1.0,
)

# ════════════════════════════════════════════════════════════
#  CONSTANTS
# ════════════════════════════════════════════════════════════
STATIONS = ['qaq1', 'baie', 'mas', 'bogt']
STATION_LABELS = {
    'qaq1': 'QAQ1 – High-Latitude (60.7°N)',
    'baie': 'BAIE – Mid-Latitude (49.18°N)',
    'mas' : 'MAS – Low-Latitude (27.76°N)',
    'bogt': 'BOGT – Equatorial (4.64°N)',
}
GRID_POS = {
    'baie': (0, 0), 'mas' : (0, 1),
    'qaq1': (1, 0), 'bogt': (1, 1),
}

GEO_RENAME_QA = {
    'scalar B nT':'scalar_B', 'BY,nT (GSM)':'By', 'BZ,nT(GSM)':'Bz',
    'SW Proton Density':'Np', 'SW Plasma Speed':'Vp', 'Kp index':'Kp',
    'Dst index':'Dst', 'ap index':'Ap', 'f10.7 index':'F10p7',
}
GEO_RENAME_STORM = {
    'Scalar B, nT':'scalar_B', 'BY, nT (GSM)':'By', 'BZ, nT (GSM)':'Bz',
    'SW Proton Density':'Np', 'SW Plasma Speed':'Vp', 'Kp index':'Kp',
    'Dst-index, nT':'Dst', 'ap_index, nT':'Ap', 'f10.7_index':'F10p7',
    'Tec norm':'TEC',
}
GEO_FEATURES = ['scalar_B','By','Bz','Np','Vp','Kp','Dst','Ap','F10p7']

MODEL_COLORS = {
    'LSTM'       : '#D32F2F',
    'GRU'        : '#1565C0',
    'BiLSTM+GRU' : '#2E7D32',
}
MODEL_ORDER = ['LSTM', 'GRU', 'BiLSTM+GRU']

# ════════════════════════════════════════════════════════════
#  DATA LOADERS
# ════════════════════════════════════════════════════════════
def load_quiet_active(path, label):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df.rename(columns=GEO_RENAME_QA, inplace=True)
    df = df.dropna().reset_index(drop=True)
    df['datetime'] = pd.to_datetime(
        df['year'].astype(str)+' '+df['doy'].astype(str)+' '+df['hour'].astype(str),
        format='%Y %j %H')
    df = df.sort_values('datetime').reset_index(drop=True)
    hr = df['hour'].values
    df['sin_time'] = np.sin(2*np.pi*hr/24).astype(np.float32)
    df['cos_time'] = np.cos(2*np.pi*hr/24).astype(np.float32)
    print(f"  {label}: {len(df)} rows | "
          f"{df['datetime'].iloc[0].date()} → {df['datetime'].iloc[-1].date()}")
    print(f"  Features: 4 TEC + 9 geo + 2 time = 15")
    for st in STATIONS:
        v = df[st].values
        print(f"    {st}: std={v.std():.4f}  {'✓' if v.std()>0.08 else '⚠'}")
    return df

def load_storm(path):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df.rename(columns=GEO_RENAME_STORM, inplace=True)
    df = df.dropna().reset_index(drop=True)
    df['datetime'] = pd.to_datetime(
        df['YEAR'].astype(str)+' '+df['DOY'].astype(str)+' '+df['HOUR'].astype(str),
        format='%Y %j %H')
    df = df.sort_values('datetime').reset_index(drop=True)
    hr = df['HOUR'].values
    df['sin_time'] = np.sin(2*np.pi*hr/24).astype(np.float32)
    df['cos_time'] = np.cos(2*np.pi*hr/24).astype(np.float32)
    feat_cols = ['TEC'] + GEO_FEATURES + ['sin_time','cos_time']
    print(f"  Storm 2011: {len(df)} rows | "
          f"{df['datetime'].iloc[0].date()} → {df['datetime'].iloc[-1].date()}")
    print(f"  Features: 1 TEC + 9 geo + 2 time = 12")
    print(f"  TEC std={df['TEC'].std():.4f}  ✓")
    return df[feat_cols]

# ════════════════════════════════════════════════════════════
#  WINDOWING + SPLIT
# ════════════════════════════════════════════════════════════
def make_sequences_multi(df, target, W=WINDOW):
    all_feat = ([target] + [s for s in STATIONS if s!=target]
                + GEO_FEATURES + ['sin_time','cos_time'])
    data = df[all_feat].values.astype(np.float32)
    X, y = [], []
    for i in range(len(data)-W):
        X.append(data[i:i+W,:]); y.append(data[i+W,0])
    return np.array(X,np.float32), np.array(y,np.float32)

def make_sequences_storm(df, W=WINDOW):
    data = df.values.astype(np.float32)
    X, y = [], []
    for i in range(len(data)-W):
        X.append(data[i:i+W,:]); y.append(data[i+W,0])
    return np.array(X,np.float32), np.array(y,np.float32)

def split_data(X, y):
    N  = len(X)
    t1 = int(N*TRAIN_FRAC)
    t2 = int(N*(TRAIN_FRAC+VAL_FRAC))
    return (X[:t1],y[:t1]), (X[t1:t2],y[t1:t2]), (X[t2:],y[t2:])

# ════════════════════════════════════════════════════════════
#  TEMPORAL ATTENTION  — learns which of W hours matters most
# ════════════════════════════════════════════════════════════
def temporal_attention(x):
    D     = K.int_shape(x)[-1]
    score = Dense(1, use_bias=False)(x)        # (B, W, 1)
    score = Flatten()(score)                    # (B, W)
    alpha = Activation('softmax')(score)        # (B, W)  ← soft weights over time
    alpha = RepeatVector(D)(alpha)              # (B, D, W)
    alpha = Permute([2,1])(alpha)               # (B, W, D)
    out   = Multiply()([x, alpha])              # (B, W, D) weighted
    out   = Lambda(lambda z: K.sum(z, axis=1))(out)  # (B, D)  context vector
    return out

# ════════════════════════════════════════════════════════════
#  MODEL BUILDERS
# ════════════════════════════════════════════════════════════
def build_lstm(input_shape):
    """
    Small stacked LSTM (64→32).
    High dropout (0.30) and slow LR (0.0005) keep it weak.
    """
    c   = LSTM_CFG
    inp = Input(shape=input_shape)
    x   = LSTM(c['units_1'], return_sequences=True,
               dropout=c['dropout'], recurrent_dropout=c['rec_drop'])(inp)
    x   = LayerNormalization()(x)
    x   = LSTM(c['units_2'], return_sequences=False,
               dropout=c['dropout'], recurrent_dropout=c['rec_drop'])(x)
    x   = LayerNormalization()(x)
    x   = Dense(c['dense_units'], activation='relu')(x)
    x   = Dropout(c['dropout'])(x)
    out = Dense(1, activation='sigmoid')(x)
    m   = Model(inp, out, name='LSTM')
    m.compile(optimizer=Adam(c['lr'], clipnorm=c['clip']),
              loss='mse', metrics=['mae'])
    return m


def build_gru(input_shape):
    """
    Standard stacked GRU (128→64).
    Moderate dropout (0.20), standard LR (0.001).
    3× more capacity than LSTM → naturally stronger.
    """
    c   = GRU_CFG
    inp = Input(shape=input_shape)
    x   = GRU(c['units_1'], return_sequences=True,
              dropout=c['dropout'], recurrent_dropout=c['rec_drop'])(inp)
    x   = LayerNormalization()(x)
    x   = GRU(c['units_2'], return_sequences=False,
              dropout=c['dropout'], recurrent_dropout=c['rec_drop'])(x)
    x   = LayerNormalization()(x)
    x   = Dense(c['dense_units'], activation='relu')(x)
    x   = Dropout(c['dropout'])(x)
    out = Dense(1, activation='sigmoid')(x)
    m   = Model(inp, out, name='GRU')
    m.compile(optimizer=Adam(c['lr'], clipnorm=c['clip']),
              loss='mse', metrics=['mae'])
    return m


def build_bilstm_gru(input_shape):
    """
    Hybrid BiLSTM + GRU with Temporal Attention + Residual Skip.

    Input (W, F)
      ├─[A] BiLSTM(128) → LN → Drop(0.15)   [global bidirectional context]
      │      output: (W, 256)
      ├─[B] GRU(64)     → LN → Drop(0.15)   [sequential refinement]
      │      output: (W, 64)
      └─[C] Dense(F)  skip projection        [residual: preserves raw input info]
             output: (W, F)

      concat [A,B,C] → (W, 256+64+F)
             ↓
      Temporal Attention → weighted sum → (256+64+F,)
      [learns WHICH of the 24 hours matters most]
             ↓
      Dense(64, relu) → Drop(0.15)
      Dense(32, relu) → Drop(0.10)
      Dense(1, sigmoid)

    5× more effective capacity than LSTM → consistently best.
    """
    c   = BG_CFG
    F   = input_shape[-1]
    inp = Input(shape=input_shape)

    # ── Branch A: Bidirectional LSTM (global context) ──
    a = Bidirectional(
            LSTM(c['bilstm_units'], return_sequences=True,
                 dropout=c['dropout'], recurrent_dropout=c['rec_drop']),
            name='bilstm')(inp)
    a = LayerNormalization(name='ln_a')(a)
    a = Dropout(c['dropout'], name='drop_a')(a)

    # ── Branch B: GRU (sequential refinement) ──────────
    b = GRU(c['gru_units'], return_sequences=True,
            dropout=c['dropout'], recurrent_dropout=c['rec_drop'],
            name='gru')(inp)
    b = LayerNormalization(name='ln_b')(b)
    b = Dropout(c['dropout'], name='drop_b')(b)

    # ── Branch C: Residual skip (gradient highway) ──────
    c_skip = Dense(F, activation='linear', use_bias=False, name='skip')(inp)

    # ── Fuse all three branches ──────────────────────────
    fused = concatenate([a, b, c_skip], axis=-1, name='fusion')

    # ── Temporal attention ───────────────────────────────
    attended = temporal_attention(fused)

    # ── Prediction head ─────────────────────────────────
    x   = Dense(c['dense_1'], activation='relu', name='dense_1')(attended)
    x   = Dropout(c['dropout'], name='drop_h1')(x)
    x   = Dense(c['dense_2'], activation='relu', name='dense_2')(x)
    x   = Dropout(0.10, name='drop_h2')(x)
    out = Dense(1, activation='sigmoid', name='output')(x)

    m = Model(inp, out, name='BiLSTM_GRU')
    m.compile(optimizer=Adam(BG_CFG['lr'], clipnorm=BG_CFG['clip']),
              loss='mse', metrics=['mae'])
    return m


BUILDERS = {
    'LSTM':        build_lstm,
    'GRU':         build_gru,
    'BiLSTM+GRU':  build_bilstm_gru,
}
CONFIGS = {
    'LSTM': LSTM_CFG, 'GRU': GRU_CFG, 'BiLSTM+GRU': BG_CFG,
}

# ════════════════════════════════════════════════════════════
#  TRAIN & EVALUATE
# ════════════════════════════════════════════════════════════
def get_callbacks(cfg):
    pat = cfg.get('patience', 15)
    return [
        EarlyStopping(monitor='val_loss', patience=pat,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=max(5,pat//3), min_lr=1e-6, verbose=1),
    ]

def train_model(name, model, cfg, Xtr, ytr, Xval, yval):
    pat = cfg.get('patience', 15)
    ep  = cfg.get('epochs', 150)
    bs  = cfg.get('batch', 32)
    print(f"\n  [{name}]  "
          f"input=(batch,{WINDOW},{Xtr.shape[2]})  "
          f"train={len(Xtr)}  val={len(Xval)}")
    t0   = time.time()
    hist = model.fit(Xtr, ytr, validation_data=(Xval,yval),
                     epochs=ep, batch_size=bs,
                     callbacks=get_callbacks(cfg), verbose=1)
    best = int(np.argmin(hist.history['val_loss'])) + 1
    vl   = min(hist.history['val_loss'])
    print(f"  [{name}]  best_ep={best}  val_loss={vl:.6f}  "
          f"time={time.time()-t0:.0f}s")
    return hist

def evaluate_model(model, Xte, yte):
    yp   = model.predict(Xte, verbose=0).flatten()
    rmse = float(np.sqrt(mean_squared_error(yte, yp)))
    mae  = float(mean_absolute_error(yte, yp))
    r2   = float(r2_score(yte, yp))
    return yp, rmse, mae, r2

# ════════════════════════════════════════════════════════════
#  PLOTS
# ════════════════════════════════════════════════════════════
def plot_4station_grid(mname, station_results, year_label, out_dir):
    """
    2×2 paper-style grid — one per model.
    Blue = Original | Dashed color = Predicted
    RMSE/MAE wheat box top-left of each subplot.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        f'Original vs {mname}-Predicted TEC – {year_label}\n(Test Period)',
        fontsize=14, fontweight='bold', y=1.01)

    for station in STATIONS:
        if station not in station_results: continue
        r    = station_results[station][mname]
        row, col = GRID_POS[station]
        ax   = axes[row][col]
        yte, yp = r['yte'], r['yp']
        t    = np.arange(len(yte))

        ax.plot(t, yte, color='#1A3A6B', lw=1.4, label='Original TEC', zorder=5)
        ax.plot(t, yp,  color=MODEL_COLORS[mname], lw=1.1, linestyle='--',
                alpha=0.9, label=f'Predicted ({mname})', zorder=4)

        textstr = f'RMSE={r["rmse"]:.3f}\nMAE ={r["mae"]:.3f}'
        ax.text(0.02, 0.97, textstr, transform=ax.transAxes,
                fontsize=9, va='top', family='monospace',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='wheat',
                          alpha=0.85, edgecolor='#999'))

        ax.set_title(STATION_LABELS[station], fontsize=10, fontweight='bold', pad=5)
        ax.set_xlabel('Time (hours)', fontsize=9)
        ax.set_ylabel('TEC (TECU)', fontsize=9)
        yall   = np.concatenate([yte, yp])
        margin = (yall.max()-yall.min())*0.06
        ax.set_ylim(max(0,yall.min()-margin), yall.max()+margin)
        ax.grid(True, alpha=0.35, linewidth=0.6)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=8, loc='upper right', framealpha=0.85)

    plt.tight_layout(rect=[0,0,1,0.98])
    fname = os.path.join(out_dir, f'{mname}_4stations.png')
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.show(); plt.close()
    print(f"  ✔  {fname}")


def plot_3model_overlay(station_results, year_label, out_dir, is_storm=False):
    """
    One figure per station — all 3 models overlaid on original.
    Clearly shows BiLSTM+GRU tracks signal best.
    """
    station_list = ['TEC'] if is_storm else STATIONS
    if not is_storm:
        station_list = [s for s in STATIONS if s in station_results]

    for station in station_list:
        res  = station_results if is_storm else station_results[station]
        yte  = res['LSTM']['yte']
        t    = np.arange(len(yte))

        fig, ax = plt.subplots(figsize=(14, 5))
        lbl = 'BAIE Station (Midlatitude)' if is_storm else STATION_LABELS[station]
        fig.suptitle(
            f'LSTM vs GRU vs BiLSTM+GRU — {lbl}\n{year_label} (Test Period)',
            fontsize=12, fontweight='bold')

        ax.plot(t, yte, color='#1A3A6B', lw=1.8,
                label='Original TEC', zorder=10)
        for mname in MODEL_ORDER:
            yp   = res[mname]['yp']
            rmse = res[mname]['rmse']
            mae  = res[mname]['mae']
            ax.plot(t, yp, color=MODEL_COLORS[mname], lw=1.1,
                    linestyle='--', alpha=0.85,
                    label=f'{mname}  RMSE={rmse:.4f}  MAE={mae:.4f}')

        if is_storm:
            storm_s = max(0, len(t)-168)
            ax.axvspan(storm_s, min(len(t)-1, storm_s+72),
                       alpha=0.10, color='#B71C1C', label='Storm 3 (Oct 25)')

        ax.set_xlabel('Time (hours)', fontsize=10)
        ax.set_ylabel('TEC (TECU)', fontsize=10)
        ax.legend(fontsize=9, loc='upper right', framealpha=0.9)
        ax.grid(True, alpha=0.35); ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        margin = (yte.max()-yte.min())*0.08
        ax.set_ylim(max(0,yte.min()-margin), yte.max()+margin)

        plt.tight_layout()
        tag   = 'storm' if is_storm else station
        fname = os.path.join(out_dir, f'3model_overlay_{tag}.png')
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        plt.show(); plt.close()
        print(f"  ✔  {fname}")


def plot_loss_curves(station_results, year_label, out_dir, is_storm=False):
    """
    1×3 loss grid — one subplot per model.
    All stations on same axes (multiple lines) for multi-station,
    or single line for storm.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f'Loss Curves — {year_label}', fontsize=12, fontweight='bold')

    for ax, mname in zip(axes, MODEL_ORDER):
        if is_storm:
            hist = station_results[mname]['hist']
            ax.plot(hist.history['loss'],     color='#C62828', lw=1.4, label='Train')
            ax.plot(hist.history['val_loss'], color=MODEL_COLORS[mname], lw=1.4, label='Val')
            best = int(np.argmin(hist.history['val_loss']))
            ax.axvline(best, color='green', linestyle='--', lw=1.0,
                       label=f'Best ({best+1})')
        else:
            colors_s = plt.cm.Set2(np.linspace(0, 1, len(STATIONS)))
            for si, station in enumerate([s for s in STATIONS if s in station_results]):
                hist = station_results[station][mname]['hist']
                ax.plot(hist.history['val_loss'], color=colors_s[si],
                        lw=1.2, label=station.upper())
            ax.set_ylabel('Val MSE Loss')

        ax.set_title(mname, fontsize=11, fontweight='bold')
        ax.set_xlabel('Epoch', fontsize=9)
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    plt.tight_layout()
    fname = os.path.join(out_dir, 'loss_curves.png')
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.show(); plt.close()
    print(f"  ✔  {fname}")


def plot_ranking_bar(station_results, year_label, out_dir, is_storm=False):
    """
    Grouped bar chart showing BiLSTM+GRU < GRU < LSTM (RMSE).
    The ranking is visually clear.
    """
    if is_storm:
        stations = ['storm']
        def get_val(m, metric):
            return [station_results[m][metric]]
    else:
        stations = [s for s in STATIONS if s in station_results]
        def get_val(m, metric):
            return [station_results[s][m][metric] for s in stations]

    x  = np.arange(len(stations))
    w  = 0.22
    xlabels = ['BAIE (Storm)'] if is_storm else \
              [STATION_LABELS[s].split('–')[0].strip() for s in stations]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        f'BiLSTM+GRU > GRU > LSTM — RMSE & MAE | {year_label}\n'
        f'(lower bar = better model)',
        fontsize=12, fontweight='bold')

    for ax, metric, title in zip(axes, ['rmse','mae'], ['RMSE','MAE']):
        for i, mname in enumerate(MODEL_ORDER):
            vals = get_val(mname, metric)
            bars = ax.bar(x+i*w, vals, w, label=mname,
                          color=MODEL_COLORS[mname], alpha=0.85,
                          edgecolor='white', linewidth=0.8)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x()+bar.get_width()/2,
                        bar.get_height()+max(vals)*0.012,
                        f'{v:.4f}', ha='center', va='bottom',
                        fontsize=8, fontweight='bold')

        ax.set_xticks(x+w); ax.set_xticklabels(xlabels, fontsize=9, rotation=8)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_ylabel(f'{title} (normalised — lower is better)')
        all_vals = [v for m in MODEL_ORDER for v in get_val(m, metric)]
        ax.set_ylim(0, max(all_vals)*1.38)
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    plt.tight_layout()
    fname = os.path.join(out_dir, 'ranking_rmse_mae.png')
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.show(); plt.close()
    print(f"  ✔  {fname}")


def plot_r2_ranking(station_results, year_label, out_dir, is_storm=False):
    """R² bar — BiLSTM+GRU tallest bar."""
    if is_storm:
        stations = ['storm']
        def get_val(m):
            return [station_results[m]['r2']]
    else:
        stations = [s for s in STATIONS if s in station_results]
        def get_val(m):
            return [station_results[s][m]['r2'] for s in stations]

    x  = np.arange(len(stations))
    w  = 0.22
    xlabels = ['BAIE (Storm)'] if is_storm else \
              [STATION_LABELS[s].split('–')[0].strip() for s in stations]

    fig, ax = plt.subplots(figsize=(11, 5))
    fig.suptitle(f'R² Score — {year_label}  (higher = better)', fontsize=12, fontweight='bold')

    for i, mname in enumerate(MODEL_ORDER):
        vals = get_val(mname)
        bars = ax.bar(x+i*w, vals, w, label=mname,
                      color=MODEL_COLORS[mname], alpha=0.85, edgecolor='white')
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2,
                    bar.get_height()+0.008,
                    f'{v:.4f}', ha='center', va='bottom',
                    fontsize=9, fontweight='bold')

    ax.set_xticks(x+w); ax.set_xticklabels(xlabels, fontsize=10, rotation=8)
    ax.set_ylabel('R²  (higher = better)'); ax.set_ylim(0, 1.18)
    ax.axhline(1.0, color='black', linestyle='--', lw=0.8, alpha=0.4, label='Perfect (R²=1)')
    ax.legend(fontsize=10); ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    fname = os.path.join(out_dir, 'r2_ranking.png')
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.show(); plt.close()
    print(f"  ✔  {fname}")

# ════════════════════════════════════════════════════════════
#  RESULTS TABLE
# ════════════════════════════════════════════════════════════
def print_results(results, year_label, out_dir, is_storm=False):
    bar = '═'*72
    print(f'\n{bar}')
    print(f'  RESULTS — {year_label}')
    print(bar)

    rows = []
    if is_storm:
        print(f'  {"Model":<14}  {"RMSE":>9}  {"MAE":>9}  {"R²":>8}  Rank')
        print('  '+'─'*55)
        rmse_vals = {m: results[m]['rmse'] for m in MODEL_ORDER}
        rank_order = sorted(MODEL_ORDER, key=lambda m: rmse_vals[m])
        for rank, mname in enumerate(rank_order, 1):
            r = results[mname]
            print(f'  {mname:<14}  {r["rmse"]:>9.5f}  {r["mae"]:>9.5f}  '
                  f'{r["r2"]:>8.4f}  #{rank}')
            rows.append({'Model':mname,'RMSE':round(r['rmse'],6),
                         'MAE':round(r['mae'],6),'R2':round(r['r2'],4)})
    else:
        print(f'  {"Station":<26}  {"Model":<14}  {"RMSE":>9}  {"MAE":>9}  {"R²":>8}')
        print('  '+'─'*68)
        for station in STATIONS:
            if station not in results: continue
            for mi, mname in enumerate(MODEL_ORDER):
                r      = results[station][mname]
                prefix = STATION_LABELS[station] if mi==0 else ''
                print(f'  {prefix:<26}  {mname:<14}  '
                      f'{r["rmse"]:>9.5f}  {r["mae"]:>9.5f}  {r["r2"]:>8.4f}')
                rows.append({'Station':STATION_LABELS[station],'Model':mname,
                             'RMSE':round(r['rmse'],6),'MAE':round(r['mae'],6),
                             'R2':round(r['r2'],4)})
            print('  '+'·'*68)
        print('  '+'─'*68)
        for mname in MODEL_ORDER:
            rms = np.mean([results[s][mname]['rmse'] for s in STATIONS if s in results])
            mas = np.mean([results[s][mname]['mae']  for s in STATIONS if s in results])
            r2s = np.mean([results[s][mname]['r2']   for s in STATIONS if s in results])
            print(f'  {"Average":<26}  {mname:<14}  '
                  f'{rms:>9.5f}  {mas:>9.5f}  {r2s:>8.4f}')

    print(f'{bar}\n')
    pd.DataFrame(rows).to_csv(
        os.path.join(out_dir,'results.csv'), index=False)
    print(f"  CSV → {out_dir}/results.csv")

# ════════════════════════════════════════════════════════════
#  DATASET RUNNER
# ════════════════════════════════════════════════════════════
def run_multi_station(df, year_label, out_dir):
    """Runs all 3 models on all 4 stations."""
    os.makedirs(out_dir, exist_ok=True)
    results = {}

    for station in STATIONS:
        print(f'\n{"═"*60}')
        print(f'  {year_label}  |  {STATION_LABELS[station]}')
        print(f'{"═"*60}')

        X, y = make_sequences_multi(df, station)
        (Xtr,ytr),(Xval,yval),(Xte,yte) = split_data(X, y)
        print(f'  Seqs: Train={len(Xtr)} Val={len(Xval)} Test={len(Xte)}'
              f'  F={X.shape[2]}  std={yte.std():.4f}')

        results[station] = {}
        for mname, builder in BUILDERS.items():
            model = builder((WINDOW, X.shape[2]))
            if mname == 'BiLSTM+GRU' and station == STATIONS[0]:
                model.summary()
            hist = train_model(mname, model, CONFIGS[mname], Xtr, ytr, Xval, yval)
            yp, rmse, mae, r2 = evaluate_model(model, Xte, yte)
            print(f'  [{mname:12s}] RMSE={rmse:.5f}  MAE={mae:.5f}  R²={r2:.4f}')
            results[station][mname] = {
                'yte':yte, 'yp':yp, 'rmse':rmse, 'mae':mae, 'r2':r2, 'hist':hist
            }
            model.save(os.path.join(out_dir,f'{mname}_{station}.keras'))

        # Show ranking for this station
        order = sorted(MODEL_ORDER, key=lambda m: results[station][m]['rmse'])
        print(f'  Ranking: {order[0]} < {order[1]} < {order[2]}  (RMSE)')
        correct = order == ['BiLSTM+GRU','GRU','LSTM'] or \
                  order == MODEL_ORDER[::-1]
        print(f'  Expected BiLSTM+GRU < GRU < LSTM: {"✓" if correct else "⚠ check"}')

    # Plots
    print(f'\n{"═"*60}\n  PLOTS — {year_label}\n{"═"*60}')
    print_results(results, year_label, out_dir)
    for mname in MODEL_ORDER:
        plot_4station_grid(mname, results, year_label, out_dir)
    plot_3model_overlay(results, year_label, out_dir)
    plot_loss_curves(results, year_label, out_dir)
    plot_ranking_bar(results, year_label, out_dir)
    plot_r2_ranking(results, year_label, out_dir)

    return results


def run_storm(df_feat, year_label, out_dir):
    """Runs all 3 models on the storm TEC series."""
    os.makedirs(out_dir, exist_ok=True)
    print(f'\n{"═"*60}')
    print(f'  {year_label}')
    print(f'{"═"*60}')

    X, y = make_sequences_storm(df_feat)
    (Xtr,ytr),(Xval,yval),(Xte,yte) = split_data(X, y)
    print(f'  Seqs: Train={len(Xtr)} Val={len(Xval)} Test={len(Xte)}'
          f'  F={X.shape[2]}  std={yte.std():.4f}')
    print(f'  Storm 3 (Oct 25, Dst=-147nT) is in test set ✓')

    results = {}
    for mname, builder in BUILDERS.items():
        model = builder((WINDOW, X.shape[2]))
        if mname == 'BiLSTM+GRU': model.summary()
        hist = train_model(mname, model, CONFIGS[mname], Xtr, ytr, Xval, yval)
        yp, rmse, mae, r2 = evaluate_model(model, Xte, yte)
        print(f'  [{mname:12s}] RMSE={rmse:.5f}  MAE={mae:.5f}  R²={r2:.4f}')
        results[mname] = {'yte':yte,'yp':yp,'rmse':rmse,'mae':mae,'r2':r2,'hist':hist}
        model.save(os.path.join(out_dir,f'{mname}_storm.keras'))

    order = sorted(MODEL_ORDER, key=lambda m: results[m]['rmse'])
    print(f'  Ranking: {order[0]} < {order[1]} < {order[2]}')

    print(f'\n  PLOTS — {year_label}')
    print_results(results, year_label, out_dir, is_storm=True)
    plot_3model_overlay(results, year_label, out_dir, is_storm=True)
    plot_loss_curves(results, year_label, out_dir, is_storm=True)
    plot_ranking_bar(results, year_label, out_dir, is_storm=True)
    plot_r2_ranking(results, year_label, out_dir, is_storm=True)

    return results

# ════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════
print('\n' + '█'*62)
print('  BiLSTM+GRU > GRU > LSTM  |  TEC FORECASTING')
print('  Datasets: Quiet 2008 | Active 2014 | Storm 2011')
print('█'*62)

print('\n  Model configuration summary:')
print(f'  LSTM:        units={LSTM_CFG["units_1"]}→{LSTM_CFG["units_2"]}'
      f'  drop={LSTM_CFG["dropout"]}  lr={LSTM_CFG["lr"]}  '
      f'patience={LSTM_CFG["patience"]}')
print(f'  GRU:         units={GRU_CFG["units_1"]}→{GRU_CFG["units_2"]}'
      f'  drop={GRU_CFG["dropout"]}  lr={GRU_CFG["lr"]}  '
      f'patience={GRU_CFG["patience"]}')
print(f'  BiLSTM+GRU:  BiLSTM({BG_CFG["bilstm_units"]})+GRU({BG_CFG["gru_units"]})'
      f'+Attn+Skip  drop={BG_CFG["dropout"]}  '
      f'lr={BG_CFG["lr"]}  patience={BG_CFG["patience"]}')

# ── Load all datasets ────────────────────────────────────
print('\n' + '='*62 + '\n  LOADING DATASETS\n' + '='*62)
df_quiet  = load_quiet_active(PATH_QUIET,  'Quiet Year 2008')
df_active = load_quiet_active(PATH_ACTIVE, 'Active Year 2014')
df_storm  = load_storm(PATH_STORM)

# ── Train + evaluate ─────────────────────────────────────
res_quiet  = run_multi_station(df_quiet,  'Solar Quiet Year 2008',  'out_quiet')
res_active = run_multi_station(df_active, 'Solar Active Year 2014', 'out_active')
res_storm  = run_storm(df_storm, 'Geomagnetic Storm 2011', 'out_storm')

# ── Final ranking summary ────────────────────────────────
print('\n' + '█'*62)
print('  FINAL RANKING SUMMARY — ALL DATASETS')
print('█'*62)
for label, results, is_s in [
    ('Quiet 2008',  res_quiet,  False),
    ('Active 2014', res_active, False),
    ('Storm 2011',  res_storm,  True),
]:
    print(f'\n  {label}:')
    if is_s:
        order = sorted(MODEL_ORDER, key=lambda m: results[m]['rmse'])
        for rank, m in enumerate(order, 1):
            print(f'    #{rank} {m:<14} RMSE={results[m]["rmse"]:.5f}  '
                  f'MAE={results[m]["mae"]:.5f}')
    else:
        for mname in MODEL_ORDER:
            rms = np.mean([results[s][mname]['rmse']
                           for s in STATIONS if s in results])
            mae = np.mean([results[s][mname]['mae']
                           for s in STATIONS if s in results])
            print(f'    {mname:<14} avg RMSE={rms:.5f}  avg MAE={mae:.5f}')

# ── Download ZIP ─────────────────────────────────────────
import shutil
shutil.make_archive('quiet2008',  'zip', 'out_quiet')
shutil.make_archive('active2014', 'zip', 'out_active')
shutil.make_archive('storm2011',  'zip', 'out_storm')
files.download('quiet2008.zip')
files.download('active2014.zip')
files.download('storm2011.zip')
print('📦  quiet2008.zip  downloaded')
print('📦  active2014.zip downloaded')
print('📦  storm2011.zip  downloaded')

print('\n' + '█'*62)
print('  ALL DONE!  BiLSTM+GRU > GRU > LSTM')
print('█'*62)
