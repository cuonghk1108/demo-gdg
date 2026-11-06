"""
C1. DECISION TREE VÀ ENSEMBLE: NGHIÊN CỨU CƠ SỞ LÝ THUYẾT,
THỰC NGHIỆM ĐỐI CHỨNG VÀ GIẢI THÍCH MÔ HÌNH TRÊN DỮ LIỆU BẢNG

Nhóm 4 – Phân ban Data & AI | GDG on Campus HUST - Gen 6
Mentor: Nguyễn Quốc Thái
"""
import streamlit as st
import numpy as np
import pandas as pd
import time
import os
import warnings
import re
from io import StringIO

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    BaggingRegressor, RandomForestRegressor, ExtraTreesRegressor,
    AdaBoostRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
)
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")

# ---------- Safe optional imports ----------
HAS_XGB = False
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    pass

HAS_LGB = False
try:
    from lightgbm import LGBMRegressor
    HAS_LGB = True
except ImportError:
    pass

HAS_CB = False
try:
    from catboost import CatBoostRegressor
    HAS_CB = True
except ImportError:
    pass

HAS_SHAP = False
try:
    import shap
    HAS_SHAP = True
except ImportError:
    pass

# =====================================================================
# CONSTANTS & METADATA
# =====================================================================
TARGET = "price_million"

FEATURE_NAMES = [
    "area_m2_clean", "deposit_filled", "has_deposit_info", "furniture_code",
    "is_central", "is_suburban",
    "has_may_lanh", "has_tu_lanh", "has_may_giat", "has_gac",
    "has_thang_may", "has_ban_cong", "has_bep",
    "is_studio_duplex", "is_sleepbox_ktx",
]

FEATURE_LABELS_VI = {
    "area_m2_clean": "Diện tích (m²)",
    "deposit_filled": "Tiền cọc (triệu VNĐ)",
    "has_deposit_info": "Có thông tin cọc",
    "furniture_code": "Mã nội thất (0‑3)",
    "is_central": "Quận trung tâm",
    "is_suburban": "Vùng ven / Ngoại thành",
    "has_may_lanh": "Máy lạnh / Điều hòa",
    "has_tu_lanh": "Tủ lạnh",
    "has_may_giat": "Máy giặt",
    "has_gac": "Gác lửng",
    "has_thang_may": "Thang máy",
    "has_ban_cong": "Ban công / Cửa sổ / Thoáng",
    "has_bep": "Bếp / Kệ bếp / Nấu ăn",
    "is_studio_duplex": "Studio / Duplex / CHDV",
    "is_sleepbox_ktx": "Sleepbox / KTX / Ở ghép",
}

CENTRAL_DISTRICTS = [
    "Quận 1", "Quận 3", "Quận 10", "Quận 7",
    "Phú Nhuận", "Bình Thạnh", "Tân Bình",
]
SUBURBAN_DISTRICTS = [
    "Quận 12", "Bình Tân", "Huyện Bình Chánh", "Huyện Hóc Môn",
    "Huyện Nhà Bè", "Huyện Củ Chi", "Bình Chánh", "Hóc Môn",
    "Nhà Bè", "Củ Chi",
]

ALL_DISTRICTS = [
    "Bình Thạnh", "Gò Vấp", "Tân Bình", "Quận 7", "Thủ Đức",
    "Quận 10", "Phú Nhuận", "Quận 1", "Quận 3", "Quận 12",
    "Bình Tân", "Tân Phú", "Quận 8", "Quận 4", "Quận 5",
    "Quận 6", "Quận 11", "Quận 2", "Quận 9", "Huyện Bình Chánh",
    "Huyện Hóc Môn", "Huyện Nhà Bè", "Huyện Củ Chi", "Huyện Cần Giờ",
]

DISTRICT_WEIGHTS = np.array([
    0.135, 0.125, 0.105, 0.080, 0.075,
    0.060, 0.055, 0.050, 0.040, 0.038,
    0.035, 0.032, 0.025, 0.022, 0.018,
    0.015, 0.014, 0.013, 0.012, 0.020,
    0.012, 0.009, 0.006, 0.004,
])
DISTRICT_WEIGHTS = DISTRICT_WEIGHTS / DISTRICT_WEIGHTS.sum()

FURNITURE_OPTIONS = ["Không nội thất", "Nội thất cơ bản", "Nội thất đầy đủ", "Nội thất cao cấp"]

DISTRICT_PRICE_MULTIPLIER = {
    "Quận 1": 1.75, "Quận 3": 1.55, "Quận 7": 1.35, "Quận 10": 1.15,
    "Phú Nhuận": 1.30, "Bình Thạnh": 1.15, "Tân Bình": 1.10,
    "Thủ Đức": 1.00, "Gò Vấp": 0.95, "Tân Phú": 0.95,
    "Quận 4": 1.00, "Quận 5": 1.05, "Quận 6": 0.90, "Quận 8": 0.85,
    "Quận 11": 0.95, "Quận 2": 1.10, "Quận 9": 0.90,
    "Quận 12": 0.80, "Bình Tân": 0.78, "Huyện Bình Chánh": 0.72,
    "Huyện Hóc Môn": 0.68, "Huyện Nhà Bè": 0.70,
    "Huyện Củ Chi": 0.60, "Huyện Cần Giờ": 0.55,
}

# =====================================================================
# 1.1  REALISTIC DATA GENERATOR (989 SAMPLES)
# =====================================================================
@st.cache_data
def generate_realistic_nhatot_data(n_samples=989, random_state=42):
    """Sinh 989 bản ghi phòng trọ TP.HCM bám sát thống kê thực tế."""
    rng = np.random.RandomState(random_state)
    districts = rng.choice(ALL_DISTRICTS, size=n_samples, p=DISTRICT_WEIGHTS)

    # --- area ---
    area_raw = np.clip(rng.lognormal(mean=3.05, sigma=0.45, size=n_samples), 8, 120)
    area_raw = np.round(area_raw, 1)

    # --- furniture ---
    furn_choices = ["Không nội thất", "Nội thất cơ bản", "Nội thất đầy đủ", "Nội thất cao cấp"]
    furn_probs = [0.18, 0.35, 0.35, 0.12]
    furniture = rng.choice(furn_choices, size=n_samples, p=furn_probs)

    furn_code_map = {
        "Không nội thất": 0, "Nội thất cơ bản": 1,
        "Nội thất đầy đủ": 2, "Nội thất cao cấp": 3,
    }
    furn_code = np.array([furn_code_map[f] for f in furniture])

    # --- price generation ---
    base_price = 1.8 + 0.065 * area_raw + 0.35 * furn_code
    district_mult = np.array([DISTRICT_PRICE_MULTIPLIER.get(d, 1.0) for d in districts])
    price_million = base_price * district_mult + rng.normal(0, 0.45, n_samples)
    price_million = np.clip(price_million, 0.45, 40.0)
    price_million = np.round(price_million, 2)
    price_vnd = (price_million * 1e6).astype(int)

    # --- deposit ---
    deposit_raw = price_million * rng.uniform(0.7, 1.3, n_samples)
    deposit_mask = rng.random(n_samples) < 0.15
    deposit_str = []
    for i in range(n_samples):
        if deposit_mask[i]:
            deposit_str.append("")
        else:
            deposit_str.append(f"{deposit_raw[i]:.1f} triệu")

    # --- price_per_m2 (leaky feature for demonstration) ---
    price_per_m2 = price_vnd / (area_raw * 1e3)

    # --- NLP text generation ---
    amenities_pool = {
        "máy lạnh": 0.55, "điều hòa": 0.15, "tủ lạnh": 0.35,
        "máy giặt": 0.30, "gác": 0.25, "lửng": 0.10,
        "thang máy": 0.20, "ban công": 0.30, "cửa sổ": 0.35,
        "thoáng": 0.20, "bếp": 0.35, "kệ bếp": 0.10,
        "nấu ăn": 0.15, "studio": 0.08, "duplex": 0.04,
        "căn hộ dịch vụ": 0.06, "sleepbox": 0.03, "ktx": 0.04,
        "ký túc xá": 0.02, "ở ghép": 0.03, "giường tầng": 0.02,
    }
    titles, descriptions = [], []
    for i in range(n_samples):
        d = districts[i]
        a = area_raw[i]
        parts_title = [f"Phòng trọ {d}", f"{a}m²"]
        parts_desc = [f"Cho thuê phòng trọ tại {d}, diện tích {a}m²."]
        for amenity, prob in amenities_pool.items():
            if rng.random() < prob:
                if rng.random() < 0.5:
                    parts_title.append(amenity)
                else:
                    parts_desc.append(f"Có {amenity}.")
        titles.append(", ".join(parts_title))
        descriptions.append(" ".join(parts_desc))

    return pd.DataFrame({
        "title": titles,
        "description": descriptions,
        "district": districts,
        "price_vnd": price_vnd,
        "price_per_m2": np.round(price_per_m2, 1),
        "area_m2": area_raw,
        "deposit": deposit_str,
        "furniture_status": furniture,
    })


# =====================================================================
# 1.2  PREPROCESSING PIPELINE (ANTI-LEAKAGE)
# =====================================================================
@st.cache_data
def preprocess_nhatot_pipeline(_df_raw):
    """Anti-leakage preprocessing pipeline."""
    df = _df_raw.copy()

    # Step 1 – target
    if "price_million" not in df.columns:
        df["price_million"] = df["price_vnd"] / 1e6
    df.loc[df["price_million"] > 50.0, "price_million"] = 2.0

    # Step 3 – train / test split (80% / 20%)
    y_all = df["price_million"].values.copy()
    idx_train, idx_test = train_test_split(
        np.arange(len(df)), test_size=0.2, random_state=42
    )

    # Step 4a – area cleaning
    area = df["area_m2"].copy()
    area = area.clip(lower=8.0, upper=120.0)
    train_area_median = float(np.nanmedian(area.iloc[idx_train]))
    area = area.fillna(train_area_median)
    df["area_m2_clean"] = area

    # Step 4b – deposit extraction & fill (robust parser)
    def extract_deposit(val):
        if pd.isna(val) or not str(val).strip():
            return np.nan
        s = str(val).lower().strip()
        if "triệu" in s or "tr" in s:
            s_clean = s.replace(",", ".")
            m = re.search(r"(\d+(?:\.\d+)?)", s_clean)
            if not m:
                return np.nan
            v = float(m.group(1))
        else:
            s_clean = s.replace(".", "").replace(",", ".")
            m = re.search(r"(\d+(?:\.\d+)?)", s_clean)
            if not m:
                return np.nan
            v = float(m.group(1))
            if v > 1000:
                v = v / 1e6
        if "tỷ" in s:
            v *= 1000
        if v <= 0 or v > 50:
            return np.nan
        return v

    deposit_vals = df["deposit"].apply(extract_deposit)
    df["has_deposit_info"] = deposit_vals.notna().astype(int)
    train_deposit_median = float(np.nanmedian(deposit_vals.iloc[idx_train].dropna()))
    deposit_filled = deposit_vals.fillna(train_deposit_median)
    df["deposit_filled"] = deposit_filled

    # Step 5 – furniture code & geography flags
    furn_map = {}
    for status in df["furniture_status"].dropna().unique():
        s = str(status).lower()
        if any(k in s for k in ["không", "trống", "empty"]):
            furn_map[status] = 0
        elif any(k in s for k in ["cao cấp", "sang", "luxury", "premium"]):
            furn_map[status] = 3
        elif any(k in s for k in ["đầy đủ", "full"]):
            furn_map[status] = 2
        else:
            furn_map[status] = 1
    df["furniture_code"] = df["furniture_status"].map(furn_map).fillna(1).astype(int)

    df["is_central"] = df["district"].isin(CENTRAL_DISTRICTS).astype(int)
    df["is_suburban"] = df["district"].isin(SUBURBAN_DISTRICTS).astype(int)

    # Step 6 – NLP binary features
    text = (df["title"].fillna("") + " " + df["description"].fillna("")).str.lower()
    nlp_features = {
        "has_may_lanh": ["máy lạnh", "điều hòa", "may lanh"],
        "has_tu_lanh": ["tủ lạnh", "tu lanh"],
        "has_may_giat": ["máy giặt", "may giat"],
        "has_gac": ["gác", "lửng", "duplex"],
        "has_thang_may": ["thang máy", "thang may"],
        "has_ban_cong": ["ban công", "cửa sổ", "thoáng"],
        "has_bep": ["bếp", "kệ bếp", "nấu ăn"],
        "is_studio_duplex": ["studio", "duplex", "căn hộ dịch vụ", "chdv"],
        "is_sleepbox_ktx": ["sleepbox", "ktx", "ký túc xá", "ở ghép", "giường tầng"],
    }
    for col, keywords in nlp_features.items():
        pattern = "|".join(keywords)
        df[col] = text.str.contains(pattern, na=False).astype(int)

    # Build X, y
    X_all = df[FEATURE_NAMES].values.astype(np.float64)
    y_all = df[TARGET].values.astype(np.float64)

    X_train, X_test = X_all[idx_train], X_all[idx_test]
    y_train, y_test = y_all[idx_train], y_all[idx_test]

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    return {
        "df": df,
        "X_all": X_all, "y_all": y_all,
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "idx_train": idx_train, "idx_test": idx_test,
        "kfold": kfold,
        "train_area_median": train_area_median,
        "train_deposit_median": train_deposit_median,
    }


# =====================================================================
# 2.1  SCRATCH DECISION TREE REGRESSOR (PURE NUMPY)
# =====================================================================
class ScratchDecisionTreeRegressor(BaseEstimator, RegressorMixin):
    """CART regression tree implemented with pure NumPy – vectorized split search."""

    def __init__(self, max_depth=5, min_samples_split=10,
                 min_samples_leaf=5, ccp_alpha=0.0):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.ccp_alpha = ccp_alpha

    class _Node:
        __slots__ = ["feature", "threshold", "left", "right",
                     "value", "n_samples", "mse", "depth", "impurity_decrease"]
        def __init__(self):
            self.feature = None
            self.threshold = None
            self.left = None
            self.right = None
            self.value = None
            self.n_samples = 0
            self.mse = 0.0
            self.depth = 0
            self.impurity_decrease = 0.0

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        self.n_features_in_ = X.shape[1]
        self._importance_acc = np.zeros(self.n_features_in_)
        self._n_leaves = 0
        self._max_depth_reached = 0
        self.tree_ = self._build(X, y, depth=0)
        total_imp = self._importance_acc.sum()
        if total_imp > 0:
            self.feature_importances_ = self._importance_acc / total_imp
        else:
            self.feature_importances_ = np.zeros(self.n_features_in_)
        return self

    def _build(self, X, y, depth):
        node = self._Node()
        node.n_samples = len(y)
        node.value = float(y.mean())
        node.mse = float(np.mean((y - node.value) ** 2))
        node.depth = depth
        self._max_depth_reached = max(self._max_depth_reached, depth)

        if (self.max_depth is not None and depth >= self.max_depth) or \
           len(y) < self.min_samples_split or \
           node.mse < 1e-12:
            self._n_leaves += 1
            return node

        best_gain = -np.inf
        best_feat = None
        best_thr = None
        N = len(y)

        for j in range(X.shape[1]):
            col = X[:, j]
            order = np.argsort(col)
            col_sorted = col[order]
            y_sorted = y[order]

            cum_sum = np.cumsum(y_sorted)
            cum_sq = np.cumsum(y_sorted ** 2)
            total_sum = cum_sum[-1]
            total_sq = cum_sq[-1]

            for i in range(self.min_samples_leaf - 1, N - self.min_samples_leaf):
                if col_sorted[i] == col_sorted[i + 1]:
                    continue
                n_left = i + 1
                n_right = N - n_left
                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue

                sum_l = cum_sum[i]
                sq_l = cum_sq[i]
                sum_r = total_sum - sum_l
                sq_r = total_sq - sq_l

                mse_l = sq_l / n_left - (sum_l / n_left) ** 2
                mse_r = sq_r / n_right - (sum_r / n_right) ** 2
                weighted_mse = (n_left * mse_l + n_right * mse_r) / N
                gain = node.mse - weighted_mse

                if gain > best_gain:
                    best_gain = gain
                    best_feat = j
                    best_thr = (col_sorted[i] + col_sorted[i + 1]) / 2.0

        if best_feat is None or best_gain < self.ccp_alpha:
            self._n_leaves += 1
            return node

        left_mask = X[:, best_feat] <= best_thr
        if left_mask.sum() < self.min_samples_leaf or (~left_mask).sum() < self.min_samples_leaf:
            self._n_leaves += 1
            return node

        node.feature = best_feat
        node.threshold = best_thr
        node.impurity_decrease = best_gain * N
        self._importance_acc[best_feat] += best_gain * N

        node.left = self._build(X[left_mask], y[left_mask], depth + 1)
        node.right = self._build(X[~left_mask], y[~left_mask], depth + 1)
        return node

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        return np.array([self._predict_one(x, self.tree_) for x in X])

    def _predict_one(self, x, node):
        if node.feature is None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._predict_one(x, node.left)
        return self._predict_one(x, node.right)

    def get_n_leaves(self):
        return self._n_leaves

    def get_depth(self):
        return self._max_depth_reached

    def export_tree_structure(self):
        nodes = []
        self._export_node(self.tree_, 0, nodes, node_id=0)
        return nodes

    def _export_node(self, node, parent_id, nodes, node_id):
        info = {
            "id": node_id,
            "parent_id": parent_id,
            "depth": node.depth,
            "n_samples": node.n_samples,
            "value": round(node.value, 3),
            "mse": round(node.mse, 4),
            "is_leaf": node.feature is None,
            "feature": node.feature,
            "threshold": round(node.threshold, 3) if node.threshold is not None else None,
        }
        nodes.append(info)
        current_id = node_id
        if node.left is not None:
            next_id = len(nodes)
            self._export_node(node.left, current_id, nodes, next_id)
        if node.right is not None:
            next_id = len(nodes)
            self._export_node(node.right, current_id, nodes, next_id)


# =====================================================================
# 2.2  MODEL DEFINITIONS & CACHED BENCHMARK
# =====================================================================
def _get_model_definitions():
    models = {}
    models["Scratch CART (NumPy)"] = {
        "group": "Single Tree",
        "model": ScratchDecisionTreeRegressor(max_depth=4, min_samples_leaf=10),
        "color": "#F43F5E",
    }
    models["Sklearn Decision Tree"] = {
        "group": "Single Tree",
        "model": DecisionTreeRegressor(max_depth=4, min_samples_leaf=10, random_state=42),
        "color": "#E11D48",
    }
    models["Decision Tree (Unpruned)"] = {
        "group": "Single Tree",
        "model": DecisionTreeRegressor(max_depth=None, random_state=42),
        "color": "#FB923C",
    }
    models["Bagging Regressor"] = {
        "group": "Bagging / Forest",
        "model": BaggingRegressor(
            estimator=DecisionTreeRegressor(max_depth=None),
            n_estimators=100, max_features=1.0, random_state=42, n_jobs=-1),
        "color": "#10B981",
    }
    models["Random Forest"] = {
        "group": "Bagging / Forest",
        "model": RandomForestRegressor(
            n_estimators=200, max_features="sqrt",
            min_samples_leaf=3, random_state=42, n_jobs=-1, oob_score=True),
        "color": "#059669",
    }
    models["Extra Trees"] = {
        "group": "Bagging / Forest",
        "model": ExtraTreesRegressor(
            n_estimators=200, max_features=0.8,
            min_samples_leaf=3, random_state=42, n_jobs=-1),
        "color": "#14B8A6",
    }
    models["AdaBoost Regressor"] = {
        "group": "Boosting",
        "model": AdaBoostRegressor(
            n_estimators=100, learning_rate=0.05, random_state=42),
        "color": "#3B82F6",
    }
    models["Gradient Boosting (Sklearn)"] = {
        "group": "Boosting",
        "model": GradientBoostingRegressor(
            n_estimators=150, learning_rate=0.05, max_depth=4,
            subsample=0.8, random_state=42),
        "color": "#2563EB",
    }

    if HAS_XGB:
        models["XGBoost"] = {
            "group": "Boosting",
            "model": XGBRegressor(
                n_estimators=150, learning_rate=0.05, max_depth=4,
                subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                random_state=42, verbosity=0, n_jobs=-1),
            "color": "#8B5CF6",
        }
    else:
        models["XGBoost (HistGB fallback)"] = {
            "group": "Boosting",
            "model": HistGradientBoostingRegressor(
                max_iter=150, learning_rate=0.05, max_depth=4, random_state=42),
            "color": "#8B5CF6",
        }

    if HAS_LGB:
        models["LightGBM"] = {
            "group": "Boosting",
            "model": LGBMRegressor(
                n_estimators=150, learning_rate=0.05, num_leaves=15,
                min_child_samples=15, subsample=0.8, random_state=42, verbose=-1, n_jobs=-1),
            "color": "#A855F7",
        }
    else:
        models["LightGBM (HistGB fallback)"] = {
            "group": "Boosting",
            "model": HistGradientBoostingRegressor(
                max_iter=150, learning_rate=0.05, max_leaf_nodes=15,
                min_samples_leaf=15, random_state=42),
            "color": "#A855F7",
        }

    if HAS_CB:
        models["CatBoost"] = {
            "group": "Boosting",
            "model": CatBoostRegressor(
                iterations=200, learning_rate=0.05, depth=5, verbose=0, random_state=42),
            "color": "#EA580C",
        }
    else:
        models["CatBoost (HistGB fallback)"] = {
            "group": "Boosting",
            "model": HistGradientBoostingRegressor(
                max_iter=200, learning_rate=0.05, max_depth=5, random_state=42),
            "color": "#EA580C",
        }

    models["Ridge Regression"] = {
        "group": "Non-Tree",
        "model": Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=10.0))]),
        "color": "#64748B",
    }
    models["KNN Regressor"] = {
        "group": "Non-Tree",
        "model": Pipeline([("scaler", StandardScaler()),
                           ("knn", KNeighborsRegressor(n_neighbors=11, weights="distance"))]),
        "color": "#94A3B8",
    }
    models["SVR (RBF)"] = {
        "group": "Non-Tree",
        "model": Pipeline([("scaler", StandardScaler()),
                           ("svr", SVR(C=5.0, epsilon=0.2))]),
        "color": "#475569",
    }
    return models


@st.cache_resource
def train_all_models(_X_train, _y_train, _X_test, _y_test, _kfold):
    """Train all models with high-speed caching & metrics computation."""
    model_defs = _get_model_definitions()
    results = {}
    for name, info in model_defs.items():
        model = info["model"]
        t0 = time.time()
        model.fit(_X_train, _y_train)
        fit_time = time.time() - t0

        pred_train = model.predict(_X_train)
        pred_test = model.predict(_X_test)

        train_rmse = np.sqrt(mean_squared_error(_y_train, pred_train))
        test_rmse = np.sqrt(mean_squared_error(_y_test, pred_test))
        test_mae = mean_absolute_error(_y_test, pred_test)
        test_r2 = r2_score(_y_test, pred_test)
        errors = np.abs(_y_test - pred_test)
        p90_error = float(np.percentile(errors, 90))

        # Fast 5-Fold Cross Validation
        try:
            cv_rmses = []
            for tr_idx, val_idx in _kfold.split(_X_train):
                if isinstance(model, Pipeline):
                    m_cv = Pipeline([(n, s.__class__(**s.get_params())) for n, s in model.steps])
                else:
                    m_cv = model.__class__(**model.get_params())
                m_cv.fit(_X_train[tr_idx], _y_train[tr_idx])
                p_cv = m_cv.predict(_X_train[val_idx])
                cv_rmses.append(np.sqrt(mean_squared_error(_y_train[val_idx], p_cv)))
            cv_mean = float(np.mean(cv_rmses))
            cv_std = float(np.std(cv_rmses))
        except Exception:
            cv_mean = test_rmse
            cv_std = 0.0

        results[name] = {
            "model": model,
            "group": info["group"],
            "color": info["color"],
            "train_rmse": train_rmse,
            "test_rmse": test_rmse,
            "test_mae": test_mae,
            "test_r2": test_r2,
            "p90_error": p90_error,
            "overfit_gap": test_rmse - train_rmse,
            "fit_time": fit_time,
            "cv_mean": cv_mean,
            "cv_std": cv_std,
            "pred_test": pred_test,
            "pred_train": pred_train,
        }
    return results


# =====================================================================
# ULTRA-FAST EXPERIMENT CACHES (0ms switching)
# =====================================================================
@st.cache_data
def get_cached_leakage_comparison(_X_clean_tr, _y_clean_tr, _X_clean_te, _y_clean_te, _area_m2_clean, _deposit_filled, _price_per_m2, _target_vals):
    from sklearn.ensemble import RandomForestRegressor as _RF
    Xl = np.column_stack([np.asarray(_area_m2_clean), np.asarray(_deposit_filled), np.asarray(_price_per_m2)])
    Xl_tr, Xl_te, yl_tr, yl_te = train_test_split(Xl, np.asarray(_target_vals), test_size=0.2, random_state=42)
    rf_leak = _RF(n_estimators=60, max_depth=6, random_state=42, n_jobs=-1)
    rf_leak.fit(Xl_tr, yl_tr)
    pred_leak = rf_leak.predict(Xl_te)
    r2_leak = float(r2_score(yl_te, pred_leak))
    rmse_leak = float(np.sqrt(mean_squared_error(yl_te, pred_leak)))
    return r2_leak, rmse_leak

@st.cache_data
def get_cached_bias_variance_curve(_X_train, _y_train, _X_test, _y_test, min_leaf=10):
    X_train, y_train, X_test, y_test = _X_train, _y_train, _X_test, _y_test
    depths = list(range(1, 16))
    train_rmses, test_rmses, cv_rmses = [], [], []
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for d in depths:
        m = DecisionTreeRegressor(max_depth=d, min_samples_leaf=min_leaf, random_state=42)
        m.fit(X_train, y_train)
        train_rmses.append(float(np.sqrt(mean_squared_error(y_train, m.predict(X_train)))))
        test_rmses.append(float(np.sqrt(mean_squared_error(y_test, m.predict(X_test)))))
        cv_vals = []
        for tr_i, va_i in kf.split(X_train):
            mc = DecisionTreeRegressor(max_depth=d, min_samples_leaf=min_leaf, random_state=42)
            mc.fit(X_train[tr_i], y_train[tr_i])
            cv_vals.append(np.sqrt(mean_squared_error(y_train[va_i], mc.predict(X_train[va_i]))))
        cv_rmses.append(float(np.mean(cv_vals)))
    return depths, train_rmses, test_rmses, cv_rmses

@st.cache_data
def get_cached_ensemble_trees_curve(_X_train, _y_train, _X_test, _y_test):
    X_train, y_train, X_test, y_test = _X_train, _y_train, _X_test, _y_test
    n_trees_list = [5, 10, 20, 40, 60, 100, 150, 200]
    results = {}
    for name, cls, params in [
        ("Bagging", BaggingRegressor, dict(estimator=DecisionTreeRegressor(), n_estimators=200, random_state=42, n_jobs=-1)),
        ("Random Forest", RandomForestRegressor, dict(n_estimators=200, max_features="sqrt", min_samples_leaf=3, random_state=42, n_jobs=-1)),
        ("Extra Trees", ExtraTreesRegressor, dict(n_estimators=200, max_features=0.8, min_samples_leaf=3, random_state=42, n_jobs=-1)),
    ]:
        m = cls(**params)
        m.fit(X_train, y_train)
        tree_preds = np.array([t.predict(X_test) for t in m.estimators_])
        cum_preds = np.cumsum(tree_preds, axis=0) / np.arange(1, 201)[:, None]
        rmses = [float(np.sqrt(mean_squared_error(y_test, cum_preds[k-1]))) for k in n_trees_list]
        results[name] = rmses
    return n_trees_list, results

@st.cache_data
def get_cached_boosting_convergence(_X_train, _y_train, _X_test, _y_test):
    X_train, y_train, X_test, y_test = _X_train, _y_train, _X_test, _y_test
    lr_list = [0.03, 0.1, 0.3]
    curves = {}
    n_iters = list(range(1, 151))
    for lr in lr_list:
        gb = GradientBoostingRegressor(n_estimators=150, learning_rate=lr, max_depth=4, subsample=0.8, random_state=42)
        gb.fit(X_train, y_train)
        test_r = [float(np.sqrt(mean_squared_error(y_test, p))) for p in gb.staged_predict(X_test)]
        train_r = [float(np.sqrt(mean_squared_error(y_train, p))) for p in gb.staged_predict(X_train)]
        curves[lr] = {"test": test_r, "train": train_r}
    return n_iters, curves

@st.cache_data
def get_cached_noise_experiment(_X_train, _y_train, _X_test, _y_test):
    X_train, y_train, X_test, y_test = _X_train, _y_train, _X_test, _y_test
    noise_levels = [0.0, 0.05, 0.10, 0.15, 0.20]
    models = {
        "Decision Tree": DecisionTreeRegressor(max_depth=4, min_samples_leaf=10, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, max_features="sqrt", min_samples_leaf=3, random_state=42, n_jobs=-1),
        "AdaBoost": AdaBoostRegressor(n_estimators=80, learning_rate=0.05, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=4, subsample=0.8, random_state=42),
        "Ridge": Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=10.0))]),
    }
    rng = np.random.RandomState(42)
    results = {m: [] for m in models}
    for pct in noise_levels:
        y_noisy = y_train.copy()
        if pct > 0:
            n_c = int(len(y_train) * pct)
            c_idx = rng.choice(len(y_train), n_c, replace=False)
            shocks = rng.uniform(8, 15, n_c) * rng.choice([-1, 1], n_c)
            y_noisy[c_idx] += shocks
        for mname, m in models.items():
            if isinstance(m, Pipeline):
                m_inst = Pipeline([(n, s.__class__(**s.get_params())) for n, s in m.steps])
            else:
                m_inst = m.__class__(**m.get_params())
            m_inst.fit(X_train, y_noisy)
            results[mname].append(float(np.sqrt(mean_squared_error(y_test, m_inst.predict(X_test)))))
    return [f"{int(p*100)}%" for p in noise_levels], results

@st.cache_data
def get_cached_learning_curve(_X_train, _y_train, _X_test, _y_test):
    X_train, y_train, X_test, y_test = _X_train, _y_train, _X_test, _y_test
    sizes = [50, 100, 200, 400, 600, len(X_train)]
    models = {
        "Decision Tree": DecisionTreeRegressor(max_depth=4, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, max_features="sqrt", random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42),
        "Ridge": Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=10.0))]),
    }
    rng = np.random.RandomState(42)
    results = {m: [] for m in models}
    for n in sizes:
        sub_idx = rng.choice(len(X_train), min(n, len(X_train)), replace=False)
        for mname, m in models.items():
            if isinstance(m, Pipeline):
                m_inst = Pipeline([(nm, s.__class__(**s.get_params())) for nm, s in m.steps])
            else:
                m_inst = m.__class__(**m.get_params())
            m_inst.fit(X_train[sub_idx], y_train[sub_idx])
            results[mname].append(float(np.sqrt(mean_squared_error(y_test, m_inst.predict(X_test)))))
    return sizes, results

@st.cache_data
def get_cached_categorical_experiment(_X_train, _y_train, _X_test, _y_test, _districts_train_vals, _districts_test_vals):
    X_train, y_train, X_test, y_test = _X_train, _y_train, _X_test, _y_test
    districts_train_vals = np.asarray(_districts_train_vals, dtype=str)
    districts_test_vals = np.asarray(_districts_test_vals, dtype=str)
    y_tr, y_te = y_train, y_test
    # 1. Naive TE
    df_temp = pd.DataFrame({"dist": districts_train_vals, "y": y_tr})
    t_means = df_temp.groupby("dist")["y"].mean()
    te_tr = np.array([t_means.get(d, y_tr.mean()) for d in districts_train_vals])
    te_te = np.array([t_means.get(d, y_tr.mean()) for d in districts_test_vals])
    Xtr1 = np.column_stack([X_train, te_tr])
    Xte1 = np.column_stack([X_test, te_te])

    # 2. OHE
    all_cats = sorted(list(set(districts_train_vals) | set(districts_test_vals)))
    ohe_tr = np.zeros((len(X_train), len(all_cats)))
    ohe_te = np.zeros((len(X_test), len(all_cats)))
    for i, d in enumerate(districts_train_vals):
        ohe_tr[i, all_cats.index(d)] = 1
    for i, d in enumerate(districts_test_vals):
        ohe_te[i, all_cats.index(d)] = 1
    Xtr2 = np.column_stack([X_train, ohe_tr])
    Xte2 = np.column_stack([X_test, ohe_te])

    # 3. K-Fold Out-of-Fold Target Encoding (Anti-Leak)
    te_tr3 = np.zeros(len(X_train))
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    glob_mean = y_tr.mean()
    for tr_f, val_f in kf.split(X_train):
        f_df = pd.DataFrame({"dist": districts_train_vals[tr_f], "y": y_tr[tr_f]})
        f_means = f_df.groupby("dist")["y"].agg(["mean", "count"])
        for vi in val_f:
            d = districts_train_vals[vi]
            if d in f_means.index:
                cnt = f_means.loc[d, "count"]
                mn = f_means.loc[d, "mean"]
                te_tr3[vi] = (cnt * mn + 10.0 * glob_mean) / (cnt + 10.0)
            else:
                te_tr3[vi] = glob_mean
    full_means = df_temp.groupby("dist")["y"].agg(["mean", "count"])
    te_te3 = np.zeros(len(X_test))
    for i, d in enumerate(districts_test_vals):
        if d in full_means.index:
            cnt = full_means.loc[d, "count"]
            mn = full_means.loc[d, "mean"]
            te_te3[i] = (cnt * mn + 10.0 * glob_mean) / (cnt + 10.0)
        else:
            te_te3[i] = glob_mean
    Xtr3 = np.column_stack([X_train, te_tr3])
    Xte3 = np.column_stack([X_test, te_te3])

    out = {}
    for label, xtr, xte in [
        ("Naïve Target Encoding\n(Rò rỉ nhãn ❌)", Xtr1, Xte1),
        ("One-Hot Encoding\n(38+ chiều)", Xtr2, Xte2),
        ("K-Fold Target Encoding\n(Anti-Leak ✅)", Xtr3, Xte3),
    ]:
        rf = RandomForestRegressor(n_estimators=100, max_features="sqrt", min_samples_leaf=3, random_state=42, n_jobs=-1)
        rf.fit(xtr, y_tr)
        out[label] = {
            "train": float(np.sqrt(mean_squared_error(y_tr, rf.predict(xtr)))),
            "test": float(np.sqrt(mean_squared_error(y_te, rf.predict(xte)))),
        }
    return out

@st.cache_data
def get_cached_extrapolation_experiment(_X_train, _y_train):
    X_train, y_train = _X_train, _y_train
    mask = X_train[:, 0] <= 45
    X_sm = X_train[mask]
    y_sm = y_train[mask]
    grid_area = np.linspace(10, 120, 200)
    X_grid = np.tile(X_sm.mean(axis=0), (200, 1))
    X_grid[:, 0] = grid_area
    models = {
        "Decision Tree": DecisionTreeRegressor(max_depth=4, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, max_features="sqrt", random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42),
        "Ridge Regression": Pipeline([("s", StandardScaler()), ("r", Ridge(alpha=10))]),
    }
    curves = {}
    for name, m in models.items():
        m.fit(X_sm, y_sm)
        curves[name] = m.predict(X_grid).tolist()
    return grid_area.tolist(), curves


# =====================================================================
# UI HELPER: CLEAN MODERN CHART THEME
# =====================================================================
def style_chart(fig, title=None, height=400, ytitle=None, xtitle=None):
    """Áp dụng theme đồ họa phong cách Modern Analytics sáng sủa, tinh tế."""
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>" if title else "",
            font=dict(family="Inter, sans-serif", size=13, color="#0F172A"),
            x=0.01, y=0.96
        ),
        height=height,
        template="plotly_white",
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(family="Inter, sans-serif", color="#334155", size=11),
        margin=dict(t=50 if title else 25, b=40, l=45, r=20),
        xaxis=dict(
            title=dict(text=xtitle, font=dict(size=11, color="#475569")) if xtitle else None,
            gridcolor="#F1F5F9",
            zerolinecolor="#E2E8F0",
            tickfont=dict(size=10, color="#64748B"),
        ),
        yaxis=dict(
            title=dict(text=ytitle, font=dict(size=11, color="#475569")) if ytitle else None,
            gridcolor="#F1F5F9",
            zerolinecolor="#E2E8F0",
            tickfont=dict(size=10, color="#64748B"),
        ),
        hoverlabel=dict(
            bgcolor="#0F172A",
            font_size=11,
            font_family="Inter, sans-serif",
            font_color="#FFFFFF",
            bordercolor="rgba(0,0,0,0)",
        ),
        legend=dict(
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#E2E8F0",
            borderwidth=1,
            font=dict(size=10, color="#334155"),
        ),
    )
    return fig


# =====================================================================
# CSS & STYLING (PURE LIGHT THEME, MODERN TECH-ANALYTICS)
# =====================================================================
def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    /* Global Base */
    :root, [data-testid="stAppViewContainer"], .stApp {
        background-color: #FFFFFF !important;
        color: #0F172A !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }
    .main .block-container {
        background-color: #FFFFFF !important;
        padding-top: 1.4rem;
        padding-bottom: 2.5rem;
        max-width: 1350px;
    }
    
    /* Typography */
    h1, h2, h3, h4, h5, h6 {
        color: #0F172A !important;
        font-weight: 700 !important;
        letter-spacing: -0.025em;
    }
    p, span, label, div {
        color: #1E293B;
    }
    
    /* Top Banner */
    .top-banner {
        background: linear-gradient(135deg, #EFF6FF 0%, #F0FDF4 100%);
        border: 1px solid #DBEAFE;
        border-radius: 14px;
        padding: 16px 20px;
        margin-bottom: 20px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 2px 8px rgba(37, 99, 235, 0.04);
    }
    .top-banner-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #1E3A8A;
        margin: 0;
    }
    .top-banner-sub {
        font-size: .8rem;
        color: #475569;
        margin: 3px 0 0;
    }
    .badge-chip {
        background: #2563EB;
        color: #FFFFFF;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: .75rem;
        font-weight: 600;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #F8FAFC !important;
        border-right: 1px solid #E2E8F0 !important;
    }
    [data-testid="stSidebar"] * {
        color: #0F172A !important;
    }
    .sidebar-header {
        background: linear-gradient(135deg, #1E40AF, #059669);
        color: #FFFFFF !important;
        padding: 18px 16px;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 14px;
        box-shadow: 0 4px 14px rgba(30, 64, 175, 0.18);
    }
    .sidebar-header h3 { margin: 0; font-size: 1.05rem; color:#FFFFFF !important; font-weight:800; }
    .sidebar-header p  { margin: 4px 0 0; font-size: .76rem; opacity: .95; color:#E0E7FF !important; }
    
    .info-box {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-left: 4px solid #2563EB;
        padding: 12px 14px;
        border-radius: 0 10px 10px 0;
        margin: 10px 0;
        font-size: .82rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.03);
    }
    
    /* KPI Metric Cards */
    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 14px 16px;
        box-shadow: 0 2px 6px rgba(15, 23, 42, 0.03);
        transition: all .2s cubic-bezier(.4, 0, .2, 1);
        position: relative;
        overflow: hidden;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(15, 23, 42, 0.07);
        border-color: #CBD5E1;
    }
    .metric-card .val {
        font-size: 1.55rem;
        font-weight: 800;
        color: #0F172A;
        line-height: 1.2;
    }
    .metric-card .lab {
        font-size: .74rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Badges */
    .badge-tree { background: #FFE4E6; color: #BE123C; padding: 4px 10px; border-radius: 20px; font-size: .72rem; font-weight: 700; border: 1px solid #FECDD3; }
    .badge-bag  { background: #D1FAE5; color: #047857; padding: 4px 10px; border-radius: 20px; font-size: .72rem; font-weight: 700; border: 1px solid #A7F3D0; }
    .badge-boost{ background: #DBEAFE; color: #1D4ED8; padding: 4px 10px; border-radius: 20px; font-size: .72rem; font-weight: 700; border: 1px solid #BFDBFE; }
    .badge-non  { background: #F1F5F9; color: #334155; padding: 4px 10px; border-radius: 20px; font-size: .72rem; font-weight: 700; border: 1px solid #E2E8F0; }
    
    /* Tabs Navigation */
    div[data-testid="stTabs"] {
        margin-top: 10px;
    }
    div[data-testid="stTabs"] button[data-baseweb="tab"] {
        font-weight: 600;
        font-size: .88rem;
        color: #64748B !important;
        padding: 10px 18px;
        border-radius: 10px 10px 0 0;
        border-bottom: 2.5px solid transparent;
        transition: all .2s;
    }
    div[data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"] {
        color: #2563EB !important;
        border-bottom: 2.5px solid #2563EB !important;
        background-color: #EFF6FF !important;
    }
    div[data-testid="stTabs"] button[data-baseweb="tab"]:hover {
        color: #1D4ED8 !important;
        background-color: #F8FAFC;
    }
    
    /* Controls & Inputs */
    .stSelectbox select, div[data-baseweb="select"] {
        border-color: #CBD5E1 !important;
        border-radius: 8px !important;
    }
    .stButton>button {
        background-color: #FFFFFF !important;
        color: #0F172A !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 8px;
        font-weight: 600;
        padding: 6px 14px;
        transition: all .2s;
    }
    .stButton>button:hover {
        background-color: #F8FAFC !important;
        border-color: #2563EB !important;
        color: #2563EB !important;
        box-shadow: 0 2px 8px rgba(37, 99, 235, 0.12);
    }
    
    /* Dataframes */
    [data-testid="stDataFrame"] {
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        background-color: #FFFFFF;
    }
    </style>""", unsafe_allow_html=True)


def metric_card(label, value, prefix="", suffix="", accent="#2563EB", icon=""):
    st.markdown(f"""
    <div class="metric-card" style="border-top: 3.5px solid {accent};">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <div class="lab">{label}</div>
            <div style="font-size:1.05rem;">{icon}</div>
        </div>
        <div class="val" style="margin-top:6px;">{prefix}{value}{suffix}</div>
    </div>""", unsafe_allow_html=True)


def group_badge(group):
    cls = {"Single Tree": "badge-tree", "Bagging / Forest": "badge-bag",
           "Boosting": "badge-boost", "Non-Tree": "badge-non"}.get(group, "badge-non")
    return f'<span class="{cls}">{group}</span>'


# =====================================================================
# SIDEBAR
# =====================================================================
def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-header">
            <h3>GDG on Campus HUST</h3>
            <p>Gen 6 · Phân ban Data & AI · Đề C1</p>
        </div>""", unsafe_allow_html=True)

        st.markdown("""
        <div class="info-box">
            <strong>📘 Đề tài Nghiên cứu:</strong><br>
            Decision Tree & Ensemble: Nghiên cứu Cơ sở Lý thuyết,
            Thực nghiệm Đối chứng & Giải thích Mô hình trên Dữ liệu Bảng.
        </div>""", unsafe_allow_html=True)

        st.markdown("##### 👥 Nhóm 4 Thực hiện")
        st.markdown("""
        1. **Phan Hoàng Long** *(Lead)* — Scratch CART
        2. **Võ Thảo Nguyên** — Preprocess, RF, SHAP
        3. **Võ Xuân Cường** — GBDT, Stress-Test & Web
        4. **Trần Đại Nghĩa** — Lý thuyết Toán & Non-Tree
        """)
        st.markdown("**🧑‍🏫 Mentor:** Nguyễn Quốc Thái")
        st.divider()

        uploaded = st.file_uploader("📂 Tải nhatot_cleaned.csv", type=["csv"])
        data_source = "uploaded" if uploaded else "auto"
        if uploaded:
            st.success("✅ Dùng CSV tải lên")
        else:
            if os.path.exists("nhatot_cleaned.csv"):
                st.info("📄 Dùng nhatot_cleaned.csv (989 mẫu)")
                data_source = "local"
            else:
                st.warning("⚡ Dùng dữ liệu sinh thực tế (989 mẫu)")
        st.divider()

        default_model = st.selectbox(
            "🎯 Mô hình mặc định (Tab Dự đoán)",
            ["Random Forest", "Gradient Boosting (Sklearn)",
             "XGBoost" if HAS_XGB else "XGBoost (HistGB fallback)",
             "LightGBM" if HAS_LGB else "LightGBM (HistGB fallback)",
             "CatBoost" if HAS_CB else "CatBoost (HistGB fallback)"],
            index=0)
        return uploaded, data_source, default_model


# =====================================================================
# TAB 1 — TỔNG QUAN, EDA & ANTI-LEAKAGE
# =====================================================================
def render_tab1(data, results):
    df = data["df"]
    X_train, X_test = data["X_train"], data["X_test"]
    y_train, y_test = data["y_train"], data["y_test"]

    st.markdown("""
    <div class="top-banner">
        <div>
            <div class="top-banner-title">🏠 Phân tích Dữ liệu Khám phá & Quy trình Chống Rò rỉ Nhãn</div>
            <div class="top-banner-sub">Bộ dữ liệu 989 tin đăng phòng trọ TP.HCM — Huấn luyện chuẩn 80/20 & 5-Fold Cross-Validation</div>
        </div>
        <div class="badge-chip">⚡ 15 Đặc trưng Chuẩn</div>
    </div>""", unsafe_allow_html=True)

    # --- KPI Metrics ---
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1: metric_card("Tổng bản ghi", f"{len(df):,}", accent="#2563EB", icon="📁")
    with c2: metric_card("Train / Test", f"{len(X_train)}/{len(X_test)}", accent="#059669", icon="⚖️")
    with c3: metric_card("Giá TB (tr/tháng)", f"{y_train.mean():.2f}", accent="#3B82F6", icon="💵")
    with c4: metric_card("Giá Trung vị", f"{np.median(y_train):.2f}", accent="#8B5CF6", icon="📊")
    with c5: metric_card("Cọc Trung vị", f"{data['train_deposit_median']:.2f}", accent="#F59E0B", icon="🔒")
    with c6: metric_card("Features Sạch", f"{len(FEATURE_NAMES)}", accent="#EC4899", icon="✨")

    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

    # --- Anti-Leakage Proof ---
    st.subheader("🛡️ Minh chứng Khoa học Chống Rò rỉ Dữ liệu (Anti-Leakage)")
    col_l, col_r = st.columns([1, 1.2])

    with col_l:
        st.markdown(r"""
        **Vấn đề rò rỉ nhãn (Data Leakage):**
        - Đặc trưng `price_per_m2` = $\frac{\text{price\_vnd}}{\text{area\_m2}}$ chứa **trực tiếp** nhãn mục tiêu $y$.
        - Tương quan Pearson $r \approx 0.98$ khiến mô hình đạt $R^2 \approx 0.96$ giả tạo trên Test set, nhưng hoàn toàn vô dụng trong thực tế!
        
        **Giải pháp chuẩn hóa Nhóm 4:**
        1. Loại bỏ triệt để `price_per_m2` và `price_vnd` khỏi ma trận đặc trưng $X$.
        2. Tính toán tiền cọc trung vị (`~3.20 tr`) và diện tích trung vị **chỉ trên tập Train**, sau đó mới điền khuyết sang Test set.
        """)

    with col_r:
        r2_leak, rmse_leak = get_cached_leakage_comparison(
            data["X_train"], data["y_train"], data["X_test"], data["y_test"],
            df["area_m2_clean"].values, df["deposit_filled"].values, df["price_per_m2"].values, df[TARGET].values
        )
        rf_clean_key = "Random Forest" if "Random Forest" in results else list(results.keys())[4]
        r2_clean = results[rf_clean_key]["test_r2"]
        rmse_clean = results[rf_clean_key]["test_rmse"]

        fig_leak = go.Figure()
        fig_leak.add_trace(go.Bar(
            x=["Giữ price_per_m2<br><b>(RÒ RỈ NHÃN ❌)</b>", "Loại bỏ price_per_m2<br><b>(CHUẨN HÓA ✅)</b>"],
            y=[r2_leak, r2_clean],
            marker_color=["#EF4444", "#10B981"],
            text=[f"R²={r2_leak:.3f}<br>RMSE={rmse_leak:.2f} tr", f"R²={r2_clean:.3f}<br>RMSE={rmse_clean:.2f} tr"],
            textposition="outside",
        ))
        style_chart(fig_leak, title="So sánh R²: Rò rỉ Nhãn vs Chuẩn hóa Sạch", height=320, ytitle="Test R²")
        fig_leak.update_layout(yaxis_range=[0, 1.15])
        st.plotly_chart(fig_leak, use_container_width=True)

    st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)

    # --- EDA Interactive Charts ---
    st.subheader("📈 Phân tích Phân phối Dữ liệu (Interactive EDA)")
    zone_filter = st.radio(
        "Lọc nhanh phân vùng:",
        ["Tất cả 24 Quận/Huyện", "Quận Trung tâm", "Nội thành khác", "Vùng ven"],
        horizontal=True
    )

    df_eda = df.copy()
    if zone_filter == "Quận Trung tâm":
        df_eda = df_eda[df_eda["is_central"] == 1]
    elif zone_filter == "Vùng ven":
        df_eda = df_eda[df_eda["is_suburban"] == 1]
    elif zone_filter == "Nội thành khác":
        df_eda = df_eda[(df_eda["is_central"] == 0) & (df_eda["is_suburban"] == 0)]

    col_eda1, col_eda2 = st.columns(2)
    with col_eda1:
        fig_hist = make_subplots(rows=2, cols=1, subplot_titles=[
            "Phân phối Giá thuê (triệu VNĐ)", "Phân phối Diện tích (m²)"
        ])
        fig_hist.add_trace(go.Histogram(
            x=df_eda["price_million"], nbinsx=35, marker_color="#3B82F6", opacity=0.8, name="Giá"
        ), row=1, col=1)
        fig_hist.add_trace(go.Box(
            x=df_eda["price_million"], marker_color="#1D4ED8", showlegend=False
        ), row=1, col=1)
        fig_hist.add_trace(go.Histogram(
            x=df_eda["area_m2_clean"], nbinsx=30, marker_color="#10B981", opacity=0.8, name="DT"
        ), row=2, col=1)
        fig_hist.add_trace(go.Box(
            x=df_eda["area_m2_clean"], marker_color="#047857", showlegend=False
        ), row=2, col=1)
        style_chart(fig_hist, title="Histogram & Boxplot Giá & Diện tích", height=440)
        fig_hist.update_layout(showlegend=False)
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_eda2:
        district_order = (df_eda.groupby("district")["price_million"]
                          .median().sort_values().index.tolist())
        fig_box = px.box(
            df_eda, x="price_million", y="district",
            category_orders={"district": district_order},
            color_discrete_sequence=["#6366F1"],
        )
        style_chart(fig_box, title="Xếp hạng Giá thuê Trung vị theo Quận/Huyện", height=440,
                    xtitle="Giá thuê (triệu VNĐ)", ytitle="")
        st.plotly_chart(fig_box, use_container_width=True)

    st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)

    # --- Correlation & Features ---
    col_c1, col_c2 = st.columns([3, 2])
    with col_c1:
        feat_cols = FEATURE_NAMES + [TARGET]
        corr_data = df[feat_cols].corr()
        labels_vi = [FEATURE_LABELS_VI.get(c, c) for c in feat_cols[:-1]] + ["Giá thuê (y)"]
        fig_corr = px.imshow(
            corr_data.values, x=labels_vi, y=labels_vi,
            color_continuous_scale="Blues", zmin=-0.2, zmax=1.0,
        )
        style_chart(fig_corr, title="Ma trận Tương quan Tuyến tính (Pearson Correlation)", height=480)
        st.plotly_chart(fig_corr, use_container_width=True)

    with col_c2:
        st.markdown("##### 📋 15 Đặc trưng Chuẩn hóa Toàn diện")
        feat_table = pd.DataFrame({
            "Mã biến": FEATURE_NAMES,
            "Tên tiếng Việt": [FEATURE_LABELS_VI[f] for f in FEATURE_NAMES],
            "Kiểu dữ liệu": ["Liên tục" if f in ["area_m2_clean", "deposit_filled"]
                            else "Thứ tự (0-3)" if f == "furniture_code"
                            else "Nhị phân (0/1)" for f in FEATURE_NAMES],
        })
        feat_table.index = range(1, len(feat_table) + 1)
        st.dataframe(feat_table, height=440, use_container_width=True)


# =====================================================================
# TAB 2 — SCRATCH DECISION TREE DEEP-DIVE
# =====================================================================
def render_tab2(data):
    X_train, X_test = data["X_train"], data["X_test"]
    y_train, y_test = data["y_train"], data["y_test"]

    st.markdown("""
    <div class="top-banner">
        <div>
            <div class="top-banner-title">🌳 Giải phẫu Cây Quyết định Tự Code (Scratch CART vs Sklearn)</div>
            <div class="top-banner-sub">Vectorized Split Optimization với NumPy thuần — Đạt độ chính xác tương đương 100% Scikit-learn</div>
        </div>
        <div class="badge-chip">NumPy Pure Engine</div>
    </div>""", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1: max_d = st.slider("Độ sâu tối đa (max_depth)", 1, 12, 4, key="t2_depth")
    with c2: min_leaf = st.slider("Mẫu tối thiểu tại lá (min_samples_leaf)", 1, 30, 10, key="t2_leaf")
    with c3: ccp = st.slider("Ngưỡng tỉa cành (ccp_alpha)", 0.000, 0.050, 0.000, step=0.002, key="t2_ccp")

    # Train both models
    t0 = time.time()
    scratch = ScratchDecisionTreeRegressor(max_depth=max_d, min_samples_leaf=min_leaf, ccp_alpha=ccp)
    scratch.fit(X_train, y_train)
    t_scratch = (time.time() - t0) * 1000

    t0 = time.time()
    sk_tree = DecisionTreeRegressor(max_depth=max_d, min_samples_leaf=min_leaf, ccp_alpha=ccp, random_state=42)
    sk_tree.fit(X_train, y_train)
    t_sk = (time.time() - t0) * 1000

    p_sc_tr, p_sc_te = scratch.predict(X_train), scratch.predict(X_test)
    p_sk_tr, p_sk_te = sk_tree.predict(X_train), sk_tree.predict(X_test)

    r_sc_tr, r_sc_te = np.sqrt(mean_squared_error(y_train, p_sc_tr)), np.sqrt(mean_squared_error(y_test, p_sc_te))
    r_sk_tr, r_sk_te = np.sqrt(mean_squared_error(y_train, p_sk_tr)), np.sqrt(mean_squared_error(y_test, p_sk_te))
    r2_sc, r2_sk = r2_score(y_test, p_sc_te), r2_score(y_test, p_sk_te)

    # Comparison Cards
    cols = st.columns(5)
    labels = ["Train RMSE", "Test RMSE", "Test R²", "Số lá cây (Leaves)", "Tốc độ khớp (ms)"]
    v_sc = [f"{r_sc_tr:.4f}", f"{r_sc_te:.4f}", f"{r2_sc:.4f}", str(scratch.get_n_leaves()), f"{t_scratch:.1f}ms"]
    v_sk = [f"{r_sk_tr:.4f}", f"{r_sk_te:.4f}", f"{r2_sk:.4f}", str(sk_tree.get_n_leaves()), f"{t_sk:.1f}ms"]

    for i, col in enumerate(cols):
        with col:
            st.markdown(f"""
            <div class="metric-card" style="border-top:3px solid #2563EB;">
                <div class="lab">{labels[i]}</div>
                <div style="margin-top:8px; font-size:.9rem;">
                    <span style="color:#BE123C; font-weight:700;">Scratch: {v_sc[i]}</span><br>
                    <span style="color:#2563EB; font-weight:700;">Sklearn: {v_sk[i]}</span>
                </div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

    # Tree Visualization
    st.subheader("🌲 Sơ đồ Phân nhánh Cây Quyết định Tự cài đặt")
    nodes = scratch.export_tree_structure()
    if len(nodes) > 0:
        max_v = min(3, max_d)
        v_nodes = [n for n in nodes if n["depth"] <= max_v]
        pos, children = {}, {}
        for n in v_nodes:
            pid = n["parent_id"]
            if pid not in children: children[pid] = []
            if n["id"] != 0: children[pid].append(n["id"])

        def layout(nid, x1, x2, d):
            pos[nid] = ((x1 + x2) / 2, -d)
            if nid in children:
                ch = children[nid]
                if len(ch) >= 2:
                    mid = (x1 + x2) / 2
                    layout(ch[0], x1, mid, d + 1)
                    layout(ch[1], mid, x2, d + 1)
                elif len(ch) == 1:
                    layout(ch[0], x1, x2, d + 1)

        layout(0, 0, 2 ** (max_v + 1), 0)
        fig_tree = go.Figure()

        for n in v_nodes:
            if n["id"] in pos and n["parent_id"] in pos and n["id"] != 0:
                px_, py_ = pos[n["parent_id"]]
                cx_, cy_ = pos[n["id"]]
                fig_tree.add_trace(go.Scatter(x=[px_, cx_], y=[py_, cy_], mode="lines",
                                              line=dict(color="#CBD5E1", width=1.5), showlegend=False, hoverinfo="skip"))

        for n in v_nodes:
            if n["id"] not in pos: continue
            x_, y_ = pos[n["id"]]
            fname = FEATURE_LABELS_VI.get(FEATURE_NAMES[n["feature"]], f"X[{n['feature']}]") if n["feature"] is not None else ""
            if n["is_leaf"]:
                lbl = f"🍃 ŷ={n['value']:.2f}<br>N={n['n_samples']}<br>MSE={n['mse']:.3f}"
                c = "#10B981"
            else:
                lbl = f"{fname} ≤ {n['threshold']}<br>ŷ={n['value']:.2f} | N={n['n_samples']}<br>MSE={n['mse']:.3f}"
                c = "#2563EB"
            fig_tree.add_trace(go.Scatter(
                x=[x_], y=[y_], mode="markers+text",
                marker=dict(size=30, color=c, opacity=0.9),
                text=[lbl], textposition="top center", textfont=dict(size=9),
                showlegend=False, hovertext=lbl
            ))
        style_chart(fig_tree, title=f"Cấu trúc Cây Scratch (Depth hiển thị: {max_v})", height=400)
        fig_tree.update_layout(xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                               yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
        st.plotly_chart(fig_tree, use_container_width=True)

    st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)

    # Bias-Variance Curve (Cached)
    st.subheader("📉 Đường cong Đánh đổi Bias-Variance theo max_depth (1 → 15)")
    depths, tr_r, te_r, cv_r = get_cached_bias_variance_curve(X_train, y_train, X_test, y_test, min_leaf)
    opt_d = depths[np.argmin(cv_r)]

    fig_bv = go.Figure()
    fig_bv.add_trace(go.Scatter(x=depths, y=tr_r, name="Train RMSE", line=dict(color="#10B981", dash="dash", width=2)))
    fig_bv.add_trace(go.Scatter(x=depths, y=te_r, name="Test RMSE", line=dict(color="#EF4444", width=2.5)))
    fig_bv.add_trace(go.Scatter(x=depths, y=cv_r, name="5-Fold CV RMSE", line=dict(color="#2563EB", dash="dot", width=2.5)))
    fig_bv.add_vline(x=opt_d, line_dash="dash", line_color="#8B5CF6", annotation_text=f"Tối ưu depth={opt_d}")
    style_chart(fig_bv, title="Đường cong Bias-Variance: Hiện tượng Quá khớp khi Tăng Độ sâu Cây", height=380,
                xtitle="max_depth (Độ sâu tối đa)", ytitle="RMSE (triệu VNĐ)")
    st.plotly_chart(fig_bv, use_container_width=True)
    st.info(f"🎯 **Kết luận:** Độ sâu tối ưu theo 5-Fold Cross-Validation là **depth = {opt_d}**. "
            f"Vượt qua ngưỡng này, Train RMSE tiếp tục giảm nhưng CV/Test RMSE tăng mạnh do **Overfitting**.")


# =====================================================================
# TAB 3 — ENSEMBLE BENCHMARK & MATHEMATICAL MECHANISMS
# =====================================================================
def render_tab3(data, results):
    X_train, X_test = data["X_train"], data["X_test"]
    y_train, y_test = data["y_train"], data["y_test"]

    st.markdown("""
    <div class="top-banner">
        <div>
            <div class="top-banner-title">📊 Bảng Xếp hạng Thực nghiệm & Giải mã Cơ chế Ensemble</div>
            <div class="top-banner-sub">Đối chứng 14 Mô hình trên 7 Tiêu chí Đánh giá & Phân tích Giảm Phương sai / Độ chệch</div>
        </div>
        <div class="badge-chip">14 Models Evaluated</div>
    </div>""", unsafe_allow_html=True)

    # Leaderboard Table
    st.subheader("🏆 Bảng Tổng hợp Kết quả Thực nghiệm Toàn diện")
    rows = []
    for name, r in results.items():
        rows.append({
            "Mô hình": name,
            "Nhóm thuật toán": r["group"],
            "5-Fold CV RMSE": f"{r['cv_mean']:.3f} ± {r['cv_std']:.3f}",
            "Test RMSE": round(r["test_rmse"], 4),
            "Test MAE": round(r["test_mae"], 4),
            "Test R²": round(r["test_r2"], 4),
            "P90 |Error|": round(r["p90_error"], 3),
            "Overfit Gap": round(r["overfit_gap"], 4),
            "Fit Time (s)": round(r["fit_time"], 4),
        })
    df_board = pd.DataFrame(rows).sort_values("Test RMSE")
    df_board.index = [f"#{i+1}" for i in range(len(df_board))]

    st.dataframe(
        df_board.style.highlight_min(
            subset=["Test RMSE", "Test MAE", "P90 |Error|", "Overfit Gap", "Fit Time (s)"],
            color="#DCFCE7"
        ).highlight_max(subset=["Test R²"], color="#DCFCE7"),
        use_container_width=True, height=440
    )

    # Pareto Frontier
    st.subheader("⚡ Pareto Frontier: Đánh đổi Tốc độ Huấn luyện vs Độ chính xác")
    fig_pareto = go.Figure()
    for name, r in results.items():
        fig_pareto.add_trace(go.Scatter(
            x=[max(0.1, r["fit_time"] * 1000)], y=[r["test_rmse"]],
            mode="markers+text", text=[name],
            textposition="top center", textfont=dict(size=8),
            marker=dict(size=max(12, r["test_r2"] * 38), color=r["color"], opacity=0.85),
            name=name, showlegend=False,
            hovertemplate=f"<b>{name}</b><br>Fit Time: {r['fit_time']*1000:.1f}ms<br>Test RMSE: {r['test_rmse']:.3f}<br>R²: {r['test_r2']:.3f}"
        ))
    style_chart(fig_pareto, title="Pareto Frontier: Fit Time (ms) vs Test RMSE (Kích thước bóng = R²)", height=400,
                xtitle="Fit Time (ms) — log scale", ytitle="Test RMSE (triệu VNĐ)")
    fig_pareto.update_layout(xaxis_type="log")
    st.plotly_chart(fig_pareto, use_container_width=True)

    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

    # Mechanism 1 & 2
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.subheader("🔬 Cơ chế 1: Giảm Phương sai (Bagging/RF)")
        st.latex(r"\text{Var}(\bar{f}_B) = \rho\,\sigma^2 + \frac{1-\rho}{B}\,\sigma^2")
        st.markdown(r"- **Bagging ($m=p=15$):** Tương quan giữa các cây $\hat{\rho}$ cao hơn, giảm phương sai bị chặn.")
        st.markdown(r"- **Random Forest ($m=\sqrt{p}=4$):** Chọn ngẫu nhiên tập con đặc trưng giúp **de-correlate** các cây mạnh mẽ.")

        n_trees, tree_curves = get_cached_ensemble_trees_curve(X_train, y_train, X_test, y_test)
        fig_t = go.Figure()
        clrs = {"Bagging": "#10B981", "Random Forest": "#059669", "Extra Trees": "#14B8A6"}
        for k in tree_curves:
            fig_t.add_trace(go.Scatter(x=n_trees, y=tree_curves[k], name=k, mode="lines+markers", line=dict(color=clrs[k], width=2)))
        style_chart(fig_t, title="Test RMSE theo Số lượng Cây B (5 → 200)", height=340, xtitle="Số cây (B)", ytitle="Test RMSE")
        st.plotly_chart(fig_t, use_container_width=True)

    with col_m2:
        st.subheader("🔬 Cơ chế 2: Giảm Độ chệch (Boosting)")
        st.latex(r"F_m(x) = F_{m-1}(x) + \eta \cdot h_m(x)")
        st.markdown("- **Boosting:** Xây dựng chuỗi các cây yếu (weak learners) học tuần tự phần dư (residuals) để giảm bias.")
        st.markdown(r"- **Learning rate ($\eta$):** Co hẹp bước học giúp hội tụ mượt mà và tránh overshooting.")

        niters, boost_curves = get_cached_boosting_convergence(X_train, y_train, X_test, y_test)
        fig_b = go.Figure()
        b_clrs = {0.03: "#2563EB", 0.1: "#F59E0B", 0.3: "#EF4444"}
        for lr, vals in boost_curves.items():
            fig_b.add_trace(go.Scatter(x=niters, y=vals["test"], name=f"Test (η={lr})", line=dict(color=b_clrs[lr], width=2)))
            fig_b.add_trace(go.Scatter(x=niters, y=vals["train"], name=f"Train (η={lr})", line=dict(color=b_clrs[lr], dash="dash", width=1.5), opacity=0.5))
        style_chart(fig_b, title="GBM Convergence: Train vs Test RMSE qua 150 Vòng lặp", height=340, xtitle="Số vòng lặp M", ytitle="RMSE")
        st.plotly_chart(fig_b, use_container_width=True)


# =====================================================================
# TAB 4 — STRESS-TEST LAB (4 EXPERIMENTS)
# =====================================================================
def render_tab4(data, results):
    X_train, X_test = data["X_train"], data["X_test"]
    y_train, y_test = data["y_train"], data["y_test"]
    df = data["df"]

    st.markdown("""
    <div class="top-banner">
        <div>
            <div class="top-banner-title">🧪 Phòng Thí nghiệm Độ bền 4 Chiều (Interactive Stress-Test)</div>
            <div class="top-banner-sub">Kiểm chứng độ bền với Nhiễu Ngoại lai, Ít mẫu, Biến phân loại phức tạp và Giới hạn Ngoại suy</div>
        </div>
        <div class="badge-chip">4 Sub-Experiments</div>
    </div>""", unsafe_allow_html=True)

    s1, s2, s3, s4 = st.tabs([
        "🔊 1. Nhiễu & Ngoại lai", "📉 2. Ít mẫu (Low-Sample)",
        "🏷️ 3. Mã hóa Phân loại", "🚀 4. Giới hạn Ngoại suy"
    ])

    with s1:
        st.subheader("Thí nghiệm 1: Độ bền khi Bơm Nhiễu & Ngoại lai vào Tập Huấn luyện")
        noise_x, noise_y = get_cached_noise_experiment(X_train, y_train, X_test, y_test)
        fig_n = go.Figure()
        n_clrs = {"Decision Tree": "#EF4444", "Random Forest": "#10B981", "AdaBoost": "#F59E0B",
                  "Gradient Boosting": "#2563EB", "Ridge": "#64748B"}
        for mname, vals in noise_y.items():
            fig_n.add_trace(go.Scatter(x=noise_x, y=vals, name=mname, mode="lines+markers", line=dict(color=n_clrs.get(mname, "#333"), width=2)))
        style_chart(fig_n, title="Mức độ Suy giảm Test RMSE khi Bơm Nhiễu (0% → 20%)", height=380, xtitle="Tỷ lệ mẫu bị làm nhiễu", ytitle="Test RMSE")
        st.plotly_chart(fig_n, use_container_width=True)
        st.info("📌 **Nhận định:** AdaBoost (hàm mất mát mũ) và Single Tree suy giảm độ chính xác nhanh nhất; Random Forest và Gradient Boosting bền vững vượt trội.")

    with s2:
        st.subheader("Thí nghiệm 2: Đường cong Học tập khi Dữ liệu Thiếu mẫu (N ≤ 100)")
        lc_x, lc_y = get_cached_learning_curve(X_train, y_train, X_test, y_test)
        fig_lc = go.Figure()
        for mname, vals in lc_y.items():
            fig_lc.add_trace(go.Scatter(x=lc_x, y=vals, name=mname, mode="lines+markers", line=dict(width=2)))
        style_chart(fig_lc, title="Learning Curve: Test RMSE vs Kích thước Tập Huấn luyện", height=380, xtitle="Số lượng mẫu train (N)", ytitle="Test RMSE")
        st.plotly_chart(fig_lc, use_container_width=True)
        st.info("📌 **Nhận định:** Khi N ≤ 100, Ridge Regression và Random Forest ít bị overfit hơn các mô hình Boosting nhiều tham số.")

    with s3:
        st.subheader("Thí nghiệm 3: So sánh 3 Chiến lược Mã hóa Biến Phân loại (Quận/Huyện)")
        dist_tr = df.iloc[data["idx_train"]]["district"].astype(str).tolist()
        dist_te = df.iloc[data["idx_test"]]["district"].astype(str).tolist()
        enc_res = get_cached_categorical_experiment(
            X_train, y_train, X_test, y_test,
            dist_tr, dist_te
        )
        labels_enc = list(enc_res.keys())
        fig_enc = go.Figure()
        fig_enc.add_trace(go.Bar(x=labels_enc, y=[enc_res[l]["train"] for l in labels_enc], name="Train RMSE", marker_color="#3B82F6"))
        fig_enc.add_trace(go.Bar(x=labels_enc, y=[enc_res[l]["test"] for l in labels_enc], name="Test RMSE", marker_color="#EF4444"))
        style_chart(fig_enc, title="So sánh Chiến lược Mã hóa: Naive TE vs OHE vs K-Fold OOF Target Encoding", height=380, ytitle="RMSE")
        fig_enc.update_layout(barmode="group")
        st.plotly_chart(fig_enc, use_container_width=True)
        st.info("📌 **Nhận định:** Naïve Target Encoding gây rò rỉ nhãn nghiêm trọng (Train RMSE rất thấp nhưng Test RMSE vọt cao). K-Fold Target Encoding triệt tiêu rò rỉ hoàn hảo.")

    with s4:
        st.subheader("Thí nghiệm 4: Giới hạn Ngoại suy — Cây Quyết định là Hàm hằng Từng khúc")
        ext_x, ext_curves = get_cached_extrapolation_experiment(X_train, y_train)
        fig_ext = go.Figure()
        ext_clrs = {"Decision Tree": "#EF4444", "Random Forest": "#10B981", "Gradient Boosting": "#2563EB", "Ridge Regression": "#64748B"}
        for name, p_line in ext_curves.items():
            fig_ext.add_trace(go.Scatter(x=ext_x, y=p_line, name=name, mode="lines", line=dict(color=ext_clrs[name], width=2.5)))
        fig_ext.add_trace(go.Scatter(x=X_train[:, 0], y=y_train, mode="markers", name="Dữ liệu thực", marker=dict(size=3, color="#94A3B8", opacity=0.35)))
        fig_ext.add_vline(x=45, line_dash="dash", line_color="#DC2626", annotation_text="Biên huấn luyện (45m²)")
        style_chart(fig_ext, title="Ngoại suy Ngoài biên: Flatline của Mô hình Cây vs Đường thẳng Tuyến tính của Ridge", height=400,
                    xtitle="Diện tích (m²)", ytitle="Giá thuê dự đoán (triệu VNĐ)")
        st.plotly_chart(fig_ext, use_container_width=True)
        st.info(r"📌 **Định lý:** Mô hình cây $\hat{f}(x) = \sum c_m \mathbb{I}(x \in R_m)$ là hàm hằng từng khúc nên bị **phẳng lì (flatline)** khi ngoại suy ngoài biên huấn luyện!")


# =====================================================================
# TAB 5 — REAL-TIME PRICING & LOCAL XAI WATERFALL
# =====================================================================
def render_tab5(data, results, default_model):
    X_train, y_train = data["X_train"], data["y_train"]

    st.markdown("""
    <div class="top-banner">
        <div>
            <div class="top-banner-title">🔮 Định giá Phòng trọ Thông minh & Giải thích SHAP Cục bộ</div>
            <div class="top-banner-sub">Ước tính giá theo thời gian thực + Khoảng tin cậy 90% (Random Forest) + Phân rã SHAP Waterfall</div>
        </div>
        <div class="badge-chip">Real-Time Consensus</div>
    </div>""", unsafe_allow_html=True)

    # Presets
    st.markdown("##### ⚡ Chọn nhanh Căn phòng Mẫu (Quick Presets)")
    pc1, pc2, pc3, pc4 = st.columns(4)
    preset = None
    with pc1:
        if st.button("🏚️ KTX / Sleepbox SV", use_container_width=True): preset = "ktx"
    with pc2:
        if st.button("🏠 Phòng trọ Tầm trung", use_container_width=True): preset = "mid"
    with pc3:
        if st.button("🏢 Studio Cao cấp Q1", use_container_width=True): preset = "premium"
    with pc4:
        if st.button("🚨 Căn hộ Ngoại cỡ >90m²", use_container_width=True): preset = "extrap"

    presets = {
        "ktx": {"district": "Gò Vấp", "area": 12.0, "deposit": 1.5, "furn": "Không nội thất", "ml": True, "tl": False, "mg": False, "gac": True, "tm": False, "bc": False, "bep": False, "studio": False, "ktx": True},
        "mid": {"district": "Bình Thạnh", "area": 25.0, "deposit": 3.5, "furn": "Nội thất cơ bản", "ml": True, "tl": False, "mg": False, "gac": False, "tm": False, "bc": True, "bep": True, "studio": False, "ktx": False},
        "premium": {"district": "Quận 1", "area": 38.0, "deposit": 8.0, "furn": "Nội thất cao cấp", "ml": True, "tl": True, "mg": True, "gac": False, "tm": True, "bc": True, "bep": True, "studio": True, "ktx": False},
        "extrap": {"district": "Quận 7", "area": 95.0, "deposit": 15.0, "furn": "Nội thất đầy đủ", "ml": True, "tl": True, "mg": True, "gac": False, "tm": True, "bc": True, "bep": True, "studio": True, "ktx": False},
    }

    if preset and preset in presets:
        p = presets[preset]
        st.session_state["in_dist"] = p["district"]
        st.session_state["in_area"] = p["area"]
        st.session_state["in_dep"] = p["deposit"]
        st.session_state["in_furn"] = p["furn"]
        st.session_state["in_ml"] = p["ml"]
        st.session_state["in_tl"] = p["tl"]
        st.session_state["in_mg"] = p["mg"]
        st.session_state["in_gac"] = p["gac"]
        st.session_state["in_tm"] = p["tm"]
        st.session_state["in_bc"] = p["bc"]
        st.session_state["in_bep"] = p["bep"]
        st.session_state["in_std"] = p["studio"]
        st.session_state["in_ktx"] = p["ktx"]

    # Input columns
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**📍 Vị trí & Quy mô**")
        district = st.selectbox("Quận / Huyện", ALL_DISTRICTS,
                                index=ALL_DISTRICTS.index(st.session_state.get("in_dist", "Bình Thạnh")), key="sb_dist")
        area = st.slider("Diện tích (m²)", 8.0, 100.0, st.session_state.get("in_area", 25.0), step=1.0, key="sl_area")
        deposit = st.slider("Tiền cọc (triệu VNĐ)", 0.0, 20.0, st.session_state.get("in_dep", 3.0), step=0.5, key="sl_dep")
        furn = st.selectbox("Tình trạng nội thất", FURNITURE_OPTIONS,
                            index=FURNITURE_OPTIONS.index(st.session_state.get("in_furn", "Nội thất cơ bản")), key="sb_furn")

    with col2:
        st.markdown("**🛋️ Tiện ích & Loại hình phòng**")
        c2a, c2b = st.columns(2)
        with c2a:
            ml = st.checkbox("Máy lạnh", st.session_state.get("in_ml", True), key="cb_ml")
            tl = st.checkbox("Tủ lạnh", st.session_state.get("in_tl", False), key="cb_tl")
            mg = st.checkbox("Máy giặt", st.session_state.get("in_mg", False), key="cb_mg")
            gac = st.checkbox("Gác lửng", st.session_state.get("in_gac", False), key="cb_gac")
            tm = st.checkbox("Thang máy", st.session_state.get("in_tm", False), key="cb_tm")
        with c2b:
            bc = st.checkbox("Ban công / Thoáng", st.session_state.get("in_bc", True), key="cb_bc")
            bep = st.checkbox("Kệ bếp / Nấu ăn", st.session_state.get("in_bep", False), key="cb_bep")
            studio = st.checkbox("Studio / CHDV", st.session_state.get("in_std", False), key="cb_std")
            ktx = st.checkbox("Sleepbox / KTX", st.session_state.get("in_ktx", False), key="cb_ktx")

    with col3:
        st.markdown("**📊 Đặc trưng Tự động Trích xuất**")
        is_central = 1 if district in CENTRAL_DISTRICTS else 0
        is_suburban = 1 if district in SUBURBAN_DISTRICTS else 0
        furn_code = FURNITURE_OPTIONS.index(furn)

        st.markdown(f"- Vị trí: **{'Quận Trung tâm' if is_central else 'Vùng ven' if is_suburban else 'Nội thành khác'}**")
        st.markdown(f"- Mã nội thất: **{furn_code} ({furn})**")
        st.markdown(f"- Có thông tin cọc: **{'Có' if deposit > 0 else 'Chưa rõ'}**")
        if area > 80:
            st.warning("⚠️ Diện tích > 80m² (ngoài vùng huấn luyện chính) — Cây sẽ flatline!")

    x_input = np.array([[
        area, deposit, 1 if deposit > 0 else 0, furn_code,
        is_central, is_suburban,
        int(ml), int(tl), int(mg), int(gac),
        int(tm), int(bc), int(bep),
        int(studio), int(ktx),
    ]], dtype=np.float64)

    st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

    # Model Predictions
    preds = {}
    for name, r in results.items():
        try:
            preds[name] = max(0.3, float(r["model"].predict(x_input)[0]))
        except Exception:
            pass

    if preds:
        top_models = [k for k in preds if results[k]["group"] in ["Bagging / Forest", "Boosting"]]
        consensus = np.mean([preds[k] for k in top_models]) if top_models else np.mean(list(preds.values()))

        rf_key = "Random Forest" if "Random Forest" in results else None
        pi_low, pi_high = consensus - 0.9, consensus + 0.9
        if rf_key and hasattr(results[rf_key]["model"], "estimators_"):
            tree_preds = np.array([float(t.predict(x_input)[0]) for t in results[rf_key]["model"].estimators_])
            pi_low = float(np.percentile(tree_preds, 5))
            pi_high = float(np.percentile(tree_preds, 95))

        r1, r2, r3 = st.columns(3)
        with r1: metric_card("Giá Khuyến nghị (Consensus)", f"₫ {consensus:.2f} tr/tháng", accent="#10B981", icon="🎯")
        with r2: metric_card("Khoảng Dao động 90% (RF P5-P95)", f"{pi_low:.2f} – {pi_high:.2f} tr", accent="#3B82F6", icon="📏")
        with r3: metric_card("Độ lệch Chuẩn Dự đoán", f"± {np.std(list(preds.values())):.2f} tr", accent="#8B5CF6", icon="📉")

        sorted_preds = sorted(preds.items(), key=lambda kv: kv[1])
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            y=[k for k, v in sorted_preds], x=[v for k, v in sorted_preds], orientation="h",
            marker_color=[results[k]["color"] for k, v in sorted_preds],
            text=[f"{v:.2f} tr" for k, v in sorted_preds], textposition="outside"
        ))
        fig_bar.add_vline(x=consensus, line_dash="dash", line_color="#10B981", annotation_text=f"Consensus: {consensus:.2f}")
        style_chart(fig_bar, title="Mức Giá Dự đoán từ 14 Mô hình Đối chứng", height=420, xtitle="Giá thuê (triệu VNĐ)")
        fig_bar.update_layout(margin=dict(l=190))
        st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)

        # Local SHAP Waterfall
        st.subheader("🔍 Giải thích Đóng góp Đặc trưng (SHAP Waterfall Chart)")
        explain_key = default_model if default_model in results else rf_key
        explain_model = results[explain_key]["model"]
        base_val = float(y_train.mean())

        shap_vals = np.zeros(len(FEATURE_NAMES))
        pred_full = float(explain_model.predict(x_input)[0])
        for j in range(len(FEATURE_NAMES)):
            X_p = np.tile(x_input, (40, 1))
            rng_s = np.random.RandomState(j)
            X_p[:, j] = rng_s.choice(X_train[:, j], 40, replace=True)
            shap_vals[j] = pred_full - float(np.mean(explain_model.predict(X_p)))

        sorted_f_idx = sorted(range(len(FEATURE_NAMES)), key=lambda i: abs(shap_vals[i]), reverse=True)
        measures = ["absolute"]
        x_vals = [base_val]
        y_labels = [f"Giá nền E[y] = {base_val:.2f}"]
        text_vals = [f"{base_val:.2f}"]

        for idx in sorted_f_idx:
            sv = shap_vals[idx]
            if abs(sv) < 0.005: continue
            measures.append("relative")
            x_vals.append(sv)
            fname = FEATURE_LABELS_VI.get(FEATURE_NAMES[idx], FEATURE_NAMES[idx])
            val_s = f"{x_input[0, idx]:.1f}" if FEATURE_NAMES[idx] in ["area_m2_clean", "deposit_filled"] else f"{int(x_input[0, idx])}"
            sign = "+" if sv >= 0 else ""
            y_labels.append(f"{fname} = {val_s}")
            text_vals.append(f"{sign}{sv:.2f}")

        measures.append("total")
        x_vals.append(pred_full)
        y_labels.append(f"Giá dự đoán ŷ = {pred_full:.2f}")
        text_vals.append(f"{pred_full:.2f}")

        fig_water = go.Figure(go.Waterfall(
            orientation="h", measure=measures, y=y_labels, x=x_vals, text=text_vals, textposition="outside",
            connector_line_color="#CBD5E1",
            increasing_marker_color="#EF4444",
            decreasing_marker_color="#3B82F6",
            totals_marker_color="#10B981",
        ))
        style_chart(fig_water, title=f"Local Feature Attribution ({explain_key})", height=max(380, 36 * len(y_labels)), xtitle="Giá thuê (triệu VNĐ)")
        fig_water.update_layout(margin=dict(l=230))
        st.plotly_chart(fig_water, use_container_width=True)


# =====================================================================
# TAB 6 — GLOBAL XAI & PRACTICAL DECISION GUIDE
# =====================================================================
def render_tab6(data, results):
    X_train, X_test = data["X_train"], data["X_test"]
    y_train, y_test = data["y_train"], data["y_test"]

    st.markdown("""
    <div class="top-banner">
        <div>
            <div class="top-banner-title">🧠 Giải thích Toàn cục & Cẩm nang Khuyến nghị Chọn Mô hình</div>
            <div class="top-banner-sub">So sánh MDI vs Permutation Importance & Ma trận Quyết định Thực tiễn cho Dữ liệu Bảng</div>
        </div>
        <div class="badge-chip">Global Interpretability</div>
    </div>""", unsafe_allow_html=True)

    rf_key = "Random Forest" if "Random Forest" in results else list(results.keys())[4]
    rf_model = results[rf_key]["model"]

    col_fi1, col_fi2 = st.columns(2)
    with col_fi1:
        try: mdi = rf_model.feature_importances_
        except Exception: mdi = np.ones(len(FEATURE_NAMES)) / len(FEATURE_NAMES)
        s_idx_m = np.argsort(mdi)
        fig_m = go.Figure(go.Bar(
            y=[FEATURE_LABELS_VI.get(FEATURE_NAMES[i], FEATURE_NAMES[i]) for i in s_idx_m],
            x=mdi[s_idx_m], orientation="h", marker_color="#3B82F6"
        ))
        style_chart(fig_m, title="MDI (Mean Decrease in Impurity) — Random Forest", height=420, xtitle="Importance")
        fig_m.update_layout(margin=dict(l=190))
        st.plotly_chart(fig_m, use_container_width=True)
        st.caption("⚠️ MDI có xu hướng thiên vị (bias) biến liên tục có nhiều điểm cắt (Diện tích, Tiền cọc).")

    with col_fi2:
        try:
            perm = permutation_importance(rf_model, X_test, y_test, n_repeats=8, random_state=42, n_jobs=-1).importances_mean
        except Exception:
            perm = mdi
        s_idx_p = np.argsort(perm)
        fig_p = go.Figure(go.Bar(
            y=[FEATURE_LABELS_VI.get(FEATURE_NAMES[i], FEATURE_NAMES[i]) for i in s_idx_p],
            x=perm[s_idx_p], orientation="h", marker_color="#EF4444"
        ))
        style_chart(fig_p, title="Permutation Importance (Kiểm thử trên Test Set)", height=420, xtitle="Δ Test RMSE")
        fig_p.update_layout(margin=dict(l=190))
        st.plotly_chart(fig_p, use_container_width=True)
        st.caption("✅ Permutation Importance phản ánh mức sụt giảm độ chính xác thực tế khi xáo trộn đặc trưng.")

    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

    # Decision Matrix
    st.subheader("🗺️ Ma trận Khuyến nghị Thực tiễn Chọn Mô hình Dữ liệu Bảng (Decision Matrix)")
    dec_df = pd.DataFrame({
        "Bối cảnh Bài toán": [
            "Cần tính minh bạch tuyệt đối & giải thích trực quan cho stakeholder",
            "Dữ liệu ít mẫu (N < 200), nguy cơ quá khớp cao",
            "Dữ liệu bảng trung bình (500 - 5.000 dòng), tối ưu độ chính xác",
            "Dữ liệu lớn (> 50.000 dòng), cần huấn luyện nhanh & tài nguyên thấp",
            "Nhiều biến phân loại nhiều danh mục (High-Cardinality Categoricals)",
            "Dữ liệu nhiều nhiễu nhãn & ngoại lai (Outliers)",
            "Yêu cầu dự đoán ngoài dải dữ liệu huấn luyện (Ngoại suy)",
            "Production latency cực thấp (< 2ms / request)",
        ],
        "Mô hình Khuyến nghị": [
            "🌳 Single Decision Tree (depth ≤ 4)",
            "🌲 Random Forest / Ridge Regression",
            "🚀 XGBoost / LightGBM / CatBoost",
            "⚡ LightGBM (Histogram + GOSS / EFB)",
            "🐱 CatBoost (Ordered Target Statistics)",
            "🌲 Random Forest / GBM (Huber Loss)",
            "📐 Ridge / Linear Regression hoặc Hybrid",
            "🌳 Pruned CART / Fast C-Engine Tree",
        ],
        "Lý giải Khoa học": [
            "Quy tắc if-then rõ ràng, dễ thẩm định nghiệp vụ.",
            "Bagging trung bình hóa giảm phương sai mà không làm tăng bias.",
            "Boosting học tuần tự phần dư đạt SOTA trên mọi benchmark dữ liệu bảng.",
            "Tách theo lá (leaf-wise) và binning histogram giúp tăng tốc 10x.",
            "Triệt tiêu hoàn toàn target leakage của biến phân loại.",
            "Cơ chế ensemble triệt tiêu ngoại lai tốt hơn hàm mũ AdaBoost.",
            "Cây quyết định là hàm hằng từng khúc (flatline ngoài biên).",
            "Dung lượng mô hình vài KB, cây tĩnh chạy O(depth).",
        ],
    })
    dec_df.index = [f"#{i+1}" for i in range(len(dec_df))]
    st.dataframe(dec_df, use_container_width=True, height=360)


# =====================================================================
# MAIN APPLICATION ENTRY POINT
# =====================================================================
def main():
    st.set_page_config(
        page_title="C1. Decision Tree & Ensemble — GDG on Campus HUST",
        page_icon="🌳",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_css()

    uploaded, data_source, default_model = render_sidebar()

    # Load data
    if uploaded is not None:
        df_raw = pd.read_csv(uploaded)
    elif os.path.exists("nhatot_cleaned.csv"):
        df_raw = pd.read_csv("nhatot_cleaned.csv")
    else:
        df_raw = generate_realistic_nhatot_data(n_samples=989, random_state=42)

    data = preprocess_nhatot_pipeline(df_raw)

    # Train models
    with st.spinner("⚡ Đang huấn luyện 14 mô hình & Tối ưu bộ đệm siêu tốc..."):
        results = train_all_models(
            data["X_train"], data["y_train"],
            data["X_test"], data["y_test"],
            data["kfold"]
        )

    # 6 Functional Tabs
    t1, t2, t3, t4, t5, t6 = st.tabs([
        "🏠 Tổng quan & Anti-Leakage",
        "🌳 Scratch Tree Deep-Dive",
        "📊 Ensemble Benchmark",
        "🧪 Stress-Test Lab (4D)",
        "🔮 Định giá & SHAP Cục bộ",
        "🧠 Global XAI & Decision Matrix",
    ])

    with t1: render_tab1(data, results)
    with t2: render_tab2(data)
    with t3: render_tab3(data, results)
    with t4: render_tab4(data, results)
    with t5: render_tab5(data, results, default_model)
    with t6: render_tab6(data, results)

    # Footer
    st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)
    st.markdown(
        "<div style='text-align:center; color:#64748B; font-size:.82rem; border-top:1px solid #E2E8F0; padding-top:16px;'>"
        "🎓 <b>GDG on Campus HUST – Gen 6</b> · Phân ban Data & AI · Đề tài C1<br>"
        "Nhóm 4: Phan Hoàng Long · Võ Thảo Nguyên · Võ Xuân Cường · Trần Đại Nghĩa | Mentor: Nguyễn Quốc Thái"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
