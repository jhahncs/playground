"""
지방소멸위험지수(ext_idx) 선행 영향요인 탐색을 위한 기계학습 파이프라인
====================================================================
- 데이터: panel_edit_foreigner_nidx_univ_2015_2025_261020.csv (2015~2025 읍면동 패널)
- 분석 목적:
  t기(현재 연도)의 외국인·다문화·대학 유학생·시군구 경제/복지 지표가
  차년도(t+1) 지방소멸위험지수(ext_idx_{t+1}) 및 지수 변화량(Δext_idx = ext_idx_{t+1} - ext_idx_t)에
  미치는 선행 영향력을 트리 기반 앙상블(Random Forest, XGBoost, LightGBM)과 SHAP으로 규명.
- 데이터 누수(Leakage) 방지:
  1) 지방소멸지수 산식(20~39세 여성 / 65세 이상)에 직접 관여하는 주민등록 연령별 인구 변수 제외
  2) 지역 규모 효과를 통제하기 위해 인구 대비 비율(천 명당/퍼센트) 파생변수 생성 및 활용
  3) 동일 읍면동(umd_id)이 Train/Test에 동시에 분할되어 발생하는 패널 누수를 막기 위해
     GroupKFold(그룹: umd_id) 및 시점 기반 분할(Temporal Split) 지원
"""

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance

# 선택적 라이브러리 로드 (XGBoost, LightGBM, SHAP)
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("[알림] xgboost가 설치되어 있지 않습니다. (pip install xgboost)")

try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    print("[알림] lightgbm이 설치되어 있지 않습니다. (pip install lightgbm)")

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("[알림] shap이 설치되어 있지 않습니다. (pip install shap)")

warnings.filterwarnings("ignore")

# 한글 폰트 설정 (Windows: 맑은 고딕)
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


# =====================================================================
# 1. 변수 한글 라벨 매핑 (시각화 및 결과 해석용)
# =====================================================================
FEATURE_KOR_MAP = {
    # 원본 외국인주민 (읍면동)
    "fr_worker": "외국인근로자수(읍면동)",
    "fr_marriage": "결혼이민자수(읍면동)",
    "fr_student": "외국인유학생수_거주기준(읍면동)",
    "fr_diaspora": "외국국적동포수(읍면동)",
    "fr_other": "기타외국인수(읍면동)",
    "nat_acq": "한국국적취득자수(읍면동)",
    "nat_marriage": "혼인귀화자수(읍면동)",
    "nat_other": "기타귀화자수(읍면동)",
    "frchild_born": "외국인주민자녀_국내출생(읍면동)",
    # 원본 다문화가구 (읍면동)
    "mc_kor_spouse": "다문화_한국인배우자수(읍면동)",
    "mc_marriage": "다문화_결혼이민자수(읍면동)",
    "mc_natural": "다문화_귀화자수(읍면동)",
    "mc_child_nat": "다문화자녀_귀화외국국적(읍면동)",
    "mc_child_born": "다문화자녀_국내출생(읍면동)",
    "mc_cohab_kor": "다문화동거_내국인(읍면동)",
    "mc_cohab_frg": "다문화동거_외국인(읍면동)",
    # 인구 대비 비율 (인구 1,000명당)
    "fr_worker_per1k": "외국인근로자 비율(천명당)",
    "fr_marriage_per1k": "결혼이민자 비율(천명당)",
    "fr_student_per1k": "외국인유학생_거주 비율(천명당)",
    "fr_diaspora_per1k": "외국국적동포 비율(천명당)",
    "fr_other_per1k": "기타외국인 비율(천명당)",
    "fr_total_per1k": "전체 외국인주민 비율(천명당)",
    "nat_acq_per1k": "한국국적취득자 비율(천명당)",
    "frchild_born_per1k": "외국인주민자녀 비율(천명당)",
    "mc_total_per1k": "다문화가구원 비율(천명당)",
    "mc_child_born_per1k": "다문화 국내출생자녀 비율(천명당)",
    # 시군구 단위 외국인/귀화자 연령대 요약 비율
    "sgg_nat_acq_youth_ratio": "시군구 귀화자 중 청년(20-39세) 비중",
    "sgg_fr_marriage_youth_ratio": "시군구 결혼이민자 중 청년(20-39세) 비중",
    "sgg_fr_student_youth_ratio": "시군구 유학생 중 청년(20-39세) 비중",
    "sgg_fr_marriage_f_ratio": "시군구 결혼이민자 중 여성 비중",
    "sgg_fr_student_f_ratio": "시군구 유학생 중 여성 비중",
    # 시군구 재정·복지·경제 지표
    "sgg_bud_in": "시군구 재정자립도(%)",
    "sgg_recipient_basic_per1k": "시군구 기초생활수급자(읍면동인구대비)",
    "sgg_recipient_basic": "시군구 기초생활수급자수",
    "sgg_recipient_facility": "시군구 시설수급자수",
    "sgg_ac_ra": "시군구 경제활동참가율(%)",
    "sgg_em_ra": "시군구 고용률(%)",
    "sgg_unem_ra": "시군구 실업률(%)",
    # 대학 유학생 (교육부 대학 본교 기준)
    "n_univ": "유학생유치 대학수(읍면동)",
    "intl_total": "대학 외국인유학생 총수",
    "intl_total_per1k": "대학 외국인유학생 비율(천명당)",
    "intl_lang_ratio": "대학유학생 중 어학연수생 비중",
    "intl_ug_ratio": "대학유학생 중 학부생 비중",
    "intl_grad_ratio": "대학유학생 중 대학원생(석박사) 비중",
    "intl_eng_sci_ratio": "대학유학생 중 이공계(공학·자연) 비중",
    "intl_fund_self_ratio": "대학유학생 중 자비유학생 비중",
    "intl_fund_schol_ratio": "대학유학생 중 장학(정부·대학초청) 비중",
    # 전년 대비 증감(동태적 변화 변수)
    "d_fr_total_per1k": "외국인주민 비율 전년비 증감",
    "d_sgg_em_ra": "시군구 고용률 전년비 증감",
    "d_sgg_bud_in": "시군구 재정자립도 전년비 증감",
    "d_intl_total_per1k": "대학유학생 비율 전년비 증감",
    # 기준점 통제용 (선택적)
    "ext_idx_t": "당해연도(t) 지방소멸위험지수",
}


# =====================================================================
# 2. 데이터 로드 및 시차(Lag) 타겟·파생변수 생성
# =====================================================================
def load_and_preprocess_data(
    csv_path: str,
    target_mode: str = "delta",  # "delta": Δext_idx (t -> t+1 변화량), "next_level": ext_idx_{t+1} (차년도 수준)
    include_baseline_idx: bool = False,
    use_ratio_features_only: bool = True,
) -> tuple[pd.DataFrame, list[str], str]:
    """
    패널 데이터를 불러와 시차 종속변수(t+1)를 생성하고,
    인구 구조 변수를 제외한 외국인·다문화·대학·경제 파생변수를 구축합니다.
    """
    print(f"[1/4] 데이터 로드 중: {csv_path}")
    df = pd.read_csv(csv_path)

    # 패널 정렬 (읍면동 ID, 연도 순)
    df = df.sort_values(["umd_id", "year"]).reset_index(drop=True)

    # -----------------------------------------------------------------
    # (A) 차년도(t+1) 타겟 생성 (동일 umd_id 내에서 연도가 연속(year + 1)인 경우만 매칭)
    # -----------------------------------------------------------------
    df["next_year"] = df.groupby("umd_id")["year"].shift(-1)
    df["ext_idx_next"] = df.groupby("umd_id")["ext_idx"].shift(-1)

    # 연도 연속성 검증 (예: 2024 -> 2025)
    valid_consecutive = df["next_year"] == (df["year"] + 1)
    df.loc[~valid_consecutive, "ext_idx_next"] = np.nan

    # 차년도 지수 변화량: Δext_idx = ext_idx_{t+1} - ext_idx_t
    # (양수면 지방소멸 위험 완화/개선, 음수면 지방소멸 위험 심화)
    df["delta_ext_idx"] = df["ext_idx_next"] - df["ext_idx"]
    df["ext_idx_t"] = df["ext_idx"]

    # 2015~2024년 데이터 중 차년도(2016~2025년) 타겟이 존재하는 행 필터링
    # 참고: 코드북 기준 외국인주민 현황은 2015~2024년 수록, 2025년은 ext_idx만 존재하므로
    # t = 2015~2024년의 설명변수로 t+1 = 2016~2025년의 ext_idx를 예측하는 구조가 데이터 손실 없이 완벽히 맞물림!
    df_model = df[df["year"].between(2015, 2024) & df["ext_idx_next"].notna()].copy()

    # -----------------------------------------------------------------
    # (B) 인구 대비 비율(천 명당) 및 구조적 파생변수 생성 (규모 왜곡 및 누수 방지)
    # -----------------------------------------------------------------
    pop_denom = df_model["pop_total"].replace(0, np.nan) / 1000.0  # 인구 1,000명 단위

    # 1) 읍면동 외국인주민 비율 (인구 1,000명당)
    df_model["fr_worker_per1k"] = df_model["fr_worker"] / pop_denom
    df_model["fr_marriage_per1k"] = df_model["fr_marriage"] / pop_denom
    df_model["fr_student_per1k"] = df_model["fr_student"] / pop_denom
    df_model["fr_diaspora_per1k"] = df_model["fr_diaspora"] / pop_denom
    df_model["fr_other_per1k"] = df_model["fr_other"] / pop_denom

    fr_cols = ["fr_worker", "fr_marriage", "fr_student", "fr_diaspora", "fr_other"]
    df_model["fr_total_per1k"] = df_model[fr_cols].sum(axis=1, min_count=1) / pop_denom
    df_model["nat_acq_per1k"] = df_model["nat_acq"] / pop_denom
    df_model["frchild_born_per1k"] = df_model["frchild_born"] / pop_denom

    # 2) 읍면동 다문화가구 비율 (인구 1,000명당)
    # 2015년은 다문화 세부 변수가 결측이므로 2016~2024년에 주로 작동
    mc_cols = [
        "mc_kor_spouse", "mc_marriage", "mc_natural",
        "mc_child_nat", "mc_child_born", "mc_cohab_kor", "mc_cohab_frg"
    ]
    df_model["mc_total_per1k"] = df_model[mc_cols].sum(axis=1, min_count=1) / pop_denom
    df_model["mc_child_born_per1k"] = df_model["mc_child_born"] / pop_denom

    # 3) 시군구 외국인/귀화자 중 청년층(20~39세) 및 여성 비중 (인구 유입의 질적 특성)
    eps = 1e-6
    sgg_nat_total = df_model["sgg_nat_acq_m"].fillna(0) + df_model["sgg_nat_acq_f"].fillna(0)
    sgg_nat_youth = (
        df_model["sgg_nat_acq_m_20_29"].fillna(0) + df_model["sgg_nat_acq_m_30_39"].fillna(0) +
        df_model["sgg_nat_acq_f_20_29"].fillna(0) + df_model["sgg_nat_acq_f_30_39"].fillna(0)
    )
    df_model["sgg_nat_acq_youth_ratio"] = np.where(sgg_nat_total > 0, sgg_nat_youth / (sgg_nat_total + eps), np.nan)

    sgg_mar_total = df_model["sgg_fr_marriage_m"].fillna(0) + df_model["sgg_fr_marriage_f"].fillna(0)
    sgg_mar_youth = (
        df_model["sgg_fr_marriage_m_20_29"].fillna(0) + df_model["sgg_fr_marriage_m_30_39"].fillna(0) +
        df_model["sgg_fr_marriage_f_20_29"].fillna(0) + df_model["sgg_fr_marriage_f_30_39"].fillna(0)
    )
    df_model["sgg_fr_marriage_youth_ratio"] = np.where(sgg_mar_total > 0, sgg_mar_youth / (sgg_mar_total + eps), np.nan)
    df_model["sgg_fr_marriage_f_ratio"] = np.where(sgg_mar_total > 0, df_model["sgg_fr_marriage_f"] / (sgg_mar_total + eps), np.nan)

    sgg_stu_total = df_model["sgg_fr_student_m"].fillna(0) + df_model["sgg_fr_student_f"].fillna(0)
    sgg_stu_youth = (
        df_model["sgg_fr_student_m_20_29"].fillna(0) + df_model["sgg_fr_student_m_30_39"].fillna(0) +
        df_model["sgg_fr_student_f_20_29"].fillna(0) + df_model["sgg_fr_student_f_30_39"].fillna(0)
    )
    df_model["sgg_fr_student_youth_ratio"] = np.where(sgg_stu_total > 0, sgg_stu_youth / (sgg_stu_total + eps), np.nan)
    df_model["sgg_fr_student_f_ratio"] = np.where(sgg_stu_total > 0, df_model["sgg_fr_student_f"] / (sgg_stu_total + eps), np.nan)

    # 2015년은 시군구 연령별 외국인 데이터가 없으므로 청년 비중을 NaN 처리
    for col in ["sgg_nat_acq_youth_ratio", "sgg_fr_marriage_youth_ratio", "sgg_fr_student_youth_ratio"]:
        df_model.loc[df_model["year"] == 2015, col] = np.nan

    # 4) 대학 외국인 유학생 특성 (교육부 기준, 읍면동 내 본교 소재 대학)
    df_model["intl_total_per1k"] = df_model["intl_total"] / pop_denom
    intl_tot = df_model["intl_total"].replace(0, np.nan)
    df_model["intl_lang_ratio"] = (df_model["intl_lang"] / intl_tot).fillna(0.0)
    df_model["intl_ug_ratio"] = (df_model["intl_ug"] / intl_tot).fillna(0.0)
    df_model["intl_grad_ratio"] = ((df_model["intl_ms"] + df_model["intl_phd"]) / intl_tot).fillna(0.0)

    eng_sci = (
        df_model["intl_ug_eng"] + df_model["intl_ug_sci"] +
        df_model["intl_ms_eng"] + df_model["intl_ms_sci"] +
        df_model["intl_phd_eng"] + df_model["intl_phd_sci"]
    )
    df_model["intl_eng_sci_ratio"] = (eng_sci / intl_tot).fillna(0.0)

    # 자비 vs 장학(우리정부·대학·자국정부) 유학생 비중 (2022년은 원자료 결측이므로 NaN 유지)
    df_model["intl_fund_self_ratio"] = np.where(
        df_model["intl_fund_self"].isna(),
        np.nan,
        (df_model["intl_fund_self"] / intl_tot).fillna(0.0),
    )
    schol_sum = df_model["intl_fund_kgov"] + df_model["intl_fund_univ"] + df_model["intl_fund_hgov"]
    df_model["intl_fund_schol_ratio"] = np.where(
        df_model["intl_fund_self"].isna(),
        np.nan,
        (schol_sum / intl_tot).fillna(0.0),
    )

    # 5) 전년 대비 변화량(Momentum) 변수 (t-1 대비 t의 변화)
    df_model["d_fr_total_per1k"] = df_model.groupby("umd_id")["fr_total_per1k"].diff()
    df_model["d_sgg_em_ra"] = df_model.groupby("umd_id")["sgg_em_ra"].diff()
    df_model["d_sgg_bud_in"] = df_model.groupby("umd_id")["sgg_bud_in"].diff()
    df_model["d_intl_total_per1k"] = df_model.groupby("umd_id")["intl_total_per1k"].diff()

    # -----------------------------------------------------------------
    # (C) 최종 설명변수(Feature) 목록 확정 (주민등록 인구/연령 변수 철저 배제)
    # -----------------------------------------------------------------
    ratio_and_macro_features = [
        # 읍면동 외국인·귀화·다문화 비율 (천 명당)
        "fr_worker_per1k",
        "fr_marriage_per1k",
        "fr_student_per1k",
        "fr_diaspora_per1k",
        "fr_other_per1k",
        "fr_total_per1k",
        "nat_acq_per1k",
        "frchild_born_per1k",
        "mc_total_per1k",
        "mc_child_born_per1k",
        # 시군구 외국인 연령/성별 구성비
        "sgg_nat_acq_youth_ratio",
        "sgg_fr_marriage_youth_ratio",
        "sgg_fr_student_youth_ratio",
        "sgg_fr_marriage_f_ratio",
        "sgg_fr_student_f_ratio",
        # 시군구 재정·복지·고용 지표
        "sgg_bud_in",
        "sgg_recipient_basic",
        "sgg_recipient_facility",
        "sgg_ac_ra",
        "sgg_em_ra",
        "sgg_unem_ra",
        # 대학 및 유학생 지표
        "n_univ",
        "intl_total_per1k",
        "intl_lang_ratio",
        "intl_ug_ratio",
        "intl_grad_ratio",
        "intl_eng_sci_ratio",
        "intl_fund_self_ratio",
        "intl_fund_schol_ratio",
        # 전년 대비 증감(동태적 변화)
        "d_fr_total_per1k",
        "d_sgg_em_ra",
        "d_sgg_bud_in",
        "d_intl_total_per1k",
    ]

    raw_count_features = [
        "fr_worker", "fr_marriage", "fr_student", "fr_diaspora", "fr_other",
        "nat_acq", "frchild_born",
        "mc_kor_spouse", "mc_marriage", "mc_natural", "mc_child_nat", "mc_child_born",
        "intl_total",
    ]

    feature_cols = list(ratio_and_macro_features)
    if not use_ratio_features_only:
        feature_cols.extend(raw_count_features)

    if include_baseline_idx:
        feature_cols.append("ext_idx_t")

    target_col = "delta_ext_idx" if target_mode == "delta" else "ext_idx_next"

    # 타겟 무한대/결측 제거
    df_model = df_model.replace([np.inf, -np.inf], np.nan)
    df_model = df_model.dropna(subset=[target_col]).reset_index(drop=True)

    print(f" -> 전처리 완료: 총 {len(df_model):,}개 관측치 (읍면동 수: {df_model['umd_id'].nunique():,}개, 연도: {df_model['year'].min()}~{df_model['year'].max()})")
    print(f" -> 타겟 변수: {target_col} ({'차년도 지수 변화량 Δext_idx' if target_mode == 'delta' else '차년도 지방소멸지수 ext_idx_{t+1}'})")
    print(f" -> 투입 설명변수 수: {len(feature_cols)}개 (주민등록 연령·성별 인구 변수 전부 제외 완료)")

    return df_model, feature_cols, target_col


# =====================================================================
# 3. 모델 정의 및 교차검증 / 학습 함수
# =====================================================================
def get_models(random_state: int = 42) -> dict:
    """Random Forest, XGBoost, LightGBM 모델 객체를 생성합니다."""
    models = {
        "RandomForest": RandomForestRegressor(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=10,
            max_features="sqrt",
            n_jobs=-1,
            random_state=random_state,
        )
    }

    if HAS_XGB:
        models["XGBoost"] = XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=1.0,
            reg_lambda=2.0,
            random_state=random_state,
            n_jobs=-1,
            tree_method="hist",
        )

    if HAS_LGBM:
        models["LightGBM"] = LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=7,
            min_child_samples=30,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=1.0,
            reg_lambda=2.0,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        )

    return models


def impute_for_sklearn(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    XGBoost/LightGBM은 결측치(NaN)를 자체 처리하지만,
    RandomForest 및 Permutation Importance를 위해 Train 중앙값으로 결측치를 대체한 복사본을 반환합니다.
    """
    medians = X_train.median()
    return X_train.fillna(medians), X_test.fillna(medians)


def train_and_evaluate(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    split_mode: str = "temporal",  # "temporal": 연도 기준 분할(권장), "group": 읍면동 Group 분할
    test_start_year: int = 2023,
) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    패널 구조에 맞게 Train/Test를 분할하고 앙상블 모델들을 학습·평가합니다.
    - temporal 분할: 2016~2022년 데이터로 학습 -> 2023~2024년(타겟 기준 2024~2025년) 예측
    - group 분할: 읍면동(umd_id) 기준으로 80% 학습 / 20% 검증 (동일 읍면동 중복 방지)
    """
    print(f"\n[2/4] 모델 학습 및 검증 시작 (분할 방식: {split_mode})")

    # 다문화가구 및 시군구 외국인 연령 데이터가 모두 존재하는 2016년 이후 구간을 기본 학습 대상으로 사용
    df_use = df[df["year"] >= 2016].copy().reset_index(drop=True)

    # 상하위 0.5% 극단적 이상치(Outlier) 윈저라이징 (소규모 인구 읍면동의 지수 급등락 노이즈 완화)
    q_low, q_high = df_use[target_col].quantile([0.005, 0.995])
    df_use[target_col] = df_use[target_col].clip(lower=q_low, upper=q_high)

    if split_mode == "temporal":
        train_mask = df_use["year"] < test_start_year
        test_mask = df_use["year"] >= test_start_year
        print(f" -> 시점 분할(Temporal Split): Train({df_use.loc[train_mask, 'year'].min()}~{df_use.loc[train_mask, 'year'].max()}년, {train_mask.sum():,}행) / Test({df_use.loc[test_mask, 'year'].min()}~{df_use.loc[test_mask, 'year'].max()}년, {test_mask.sum():,}행)")
    else:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, test_idx = next(gss.split(df_use, groups=df_use["umd_id"]))
        train_mask = df_use.index.isin(train_idx)
        test_mask = df_use.index.isin(test_idx)
        print(f" -> 지역 그룹 분할(Group Split): Train({train_mask.sum():,}행) / Test({test_mask.sum():,}행)")

    X_train = df_use.loc[train_mask, feature_cols].copy()
    y_train = df_use.loc[train_mask, target_col].copy()
    X_test = df_use.loc[test_mask, feature_cols].copy()
    y_test = df_use.loc[test_mask, target_col].copy()

    X_train_imp, X_test_imp = impute_for_sklearn(X_train, X_test)

    models = get_models()
    results = {}

    for name, model in models.items():
        # RandomForest는 결측치 대체본 사용, XGB/LGBM은 원본(또는 대체본) 사용
        if name == "RandomForest":
            model.fit(X_train_imp, y_train)
            preds = model.predict(X_test_imp)
        else:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        results[name] = {
            "model": model,
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2,
            "preds": preds,
        }
        print(f"   * [{name:12s}] RMSE: {rmse:.4f} | MAE: {mae:.4f} | R²: {r2:.4f}")

    return results, X_train, X_test, y_train, y_test


# =====================================================================
# 4. 변수 중요도(Feature Importance) 및 SHAP 심층 분석
# =====================================================================
def analyze_and_plot_importances(
    results: dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    target_col: str,
    output_dir: str = "outputs",
) -> pd.DataFrame:
    """
    1) 모델별 자체 변수 중요도(Gain / Impurity) 및 Permutation Importance를 통합 비교하고
    2) 최적 부스팅 모델(LightGBM 또는 XGBoost)에 대해 SHAP Summary Plot & Bar Plot & Dependence Plot을 저장합니다.
    """
    print(f"\n[3/4] 변수 중요도 및 SHAP 분석 중... (저장 경로: {output_dir}/)")
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    feature_names = X_test.columns.tolist()
    kor_names = [FEATURE_KOR_MAP.get(c, c) for c in feature_names]

    # 1) 모델별 기본 Feature Importance 취합 (정규화하여 비교)
    imp_df = pd.DataFrame({"feature": feature_names, "feature_kor": kor_names})

    for name, res in results.items():
        model = res["model"]
        if hasattr(model, "feature_importances_"):
            raw_imp = np.array(model.feature_importances_, dtype=float)
            norm_imp = raw_imp / (raw_imp.sum() + 1e-12)
            imp_df[f"{name}_imp"] = norm_imp

    # 2) 최고 성능 모델 선택 (R2 기준, SHAP 호환을 위해 LightGBM/XGBoost 우선)
    best_name = max(results.keys(), key=lambda k: results[k]["R2"])
    shap_model_name = (
        "LightGBM" if "LightGBM" in results
        else ("XGBoost" if "XGBoost" in results else best_name)
    )
    best_model = results[shap_model_name]["model"]
    print(f" -> SHAP 기준 모델: {shap_model_name} (Test R² = {results[shap_model_name]['R2']:.4f})")

    # 3) SHAP 값 산출 및 시각화
    if HAS_SHAP:
        # 연산 속도를 위해 테스트셋에서 최대 2,000개 샘플 추출
        sample_size = min(2000, len(X_test))
        X_shap = X_test.sample(n=sample_size, random_state=42).copy()
        X_shap_kor = X_shap.rename(columns=FEATURE_KOR_MAP)

        explainer = shap.TreeExplainer(best_model)
        shap_values = explainer.shap_values(X_shap)

        # 변수별 평균 절대 SHAP 값(|SHAP|) 계산 및 방향성(상관계수 부호) 산출
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        shap_direction = []
        for idx, col in enumerate(feature_names):
            valid = X_shap[col].notna()
            if valid.sum() > 10 and X_shap.loc[valid, col].std() > 0:
                corr = np.corrcoef(X_shap.loc[valid, col], shap_values[valid, idx])[0, 1]
                shap_direction.append("양(+) [지수 개선/완화]" if corr > 0 else "음(-) [지수 하락/심화]")
            else:
                shap_direction.append("중립/비선형")

        shap_map = dict(zip(feature_names, mean_abs_shap))
        dir_map = dict(zip(feature_names, shap_direction))
        imp_df["SHAP_mean_abs"] = imp_df["feature"].map(shap_map)
        imp_df["SHAP_direction"] = imp_df["feature"].map(dir_map)
        imp_df = imp_df.sort_values("SHAP_mean_abs", ascending=False).reset_index(drop=True)

        # (Plot 1) SHAP Beeswarm Summary Plot (방향성 + 중요도)
        plt.figure(figsize=(11, 9))
        shap.summary_plot(
            shap_values,
            X_shap_kor,
            max_display=20,
            show=False,
        )
        plt.title(f"지방소멸 영향요인 SHAP Summary Plot ({shap_model_name} -> {target_col})", fontsize=14, pad=15)
        plt.tight_layout()
        beeswarm_file = out_path / f"shap_summary_beeswarm_{target_col}.png"
        plt.savefig(beeswarm_file, dpi=300, bbox_inches="tight")
        plt.close()

        # (Plot 2) SHAP Bar Plot (상위 20개 절대 중요도)
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values,
            X_shap_kor,
            plot_type="bar",
            max_display=20,
            show=False,
        )
        plt.title(f"지방소멸 핵심 영향변수 Top 20 (|SHAP| 평균, {shap_model_name})", fontsize=14, pad=15)
        plt.tight_layout()
        bar_file = out_path / f"shap_importance_bar_{target_col}.png"
        plt.savefig(bar_file, dpi=300, bbox_inches="tight")
        plt.close()

        # (Plot 3) 상위 4대 핵심 변수의 SHAP Dependence Plot (비선형 임계점 확인)
        top4_features = imp_df["feature"].head(4).tolist()
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        for i, feat in enumerate(top4_features):
            feat_kor = FEATURE_KOR_MAP.get(feat, feat)
            shap.dependence_plot(
                feat_kor,
                shap_values,
                X_shap_kor,
                ax=axes[i],
                show=False,
                interaction_index="auto",
            )
            axes[i].set_title(f"[Top {i+1}] {feat_kor} 효과", fontsize=12)
        plt.suptitle(f"상위 4대 핵심 변수의 비선형 한계효과 (SHAP Dependence, {target_col})", fontsize=15, y=1.02)
        plt.tight_layout()
        dep_file = out_path / f"shap_dependence_top4_{target_col}.png"
        plt.savefig(dep_file, dpi=300, bbox_inches="tight")
        plt.close()

        print(f" -> SHAP 시각화 저장 완료:\n    1) {beeswarm_file}\n    2) {bar_file}\n    3) {dep_file}")
    else:
        # SHAP 미설치 시 앙상블 평균 중요도로 정렬
        imp_cols = [c for c in imp_df.columns if c.endswith("_imp")]
        imp_df["Mean_Ensemble_Imp"] = imp_df[imp_cols].mean(axis=1)
        imp_df = imp_df.sort_values("Mean_Ensemble_Imp", ascending=False).reset_index(drop=True)

    # (Plot 4) 3대 앙상블 모델(RF, XGB, LGBM) 변수 중요도 비교 차트
    imp_cols = [c for c in imp_df.columns if c.endswith("_imp")]
    if imp_cols:
        top15 = imp_df.head(15).copy()
        top15_melt = top15.melt(
            id_vars=["feature_kor"],
            value_vars=imp_cols,
            var_name="Model",
            value_name="Normalized_Importance",
        )
        top15_melt["Model"] = top15_melt["Model"].str.replace("_imp", "")

        plt.figure(figsize=(11, 8))
        sns.barplot(
            data=top15_melt,
            x="Normalized_Importance",
            y="feature_kor",
            hue="Model",
            palette="viridis",
        )
        plt.title(f"트리 앙상블 모델별(RF / XGB / LGBM) 상위 15개 변수 중요도 비교 ({target_col})", fontsize=13)
        plt.xlabel("정규화 변수 중요도 (Normalized Importance)")
        plt.ylabel("설명변수")
        plt.legend(title="모델", loc="lower right")
        plt.tight_layout()
        comp_file = out_path / f"ensemble_importance_comparison_{target_col}.png"
        plt.savefig(comp_file, dpi=300, bbox_inches="tight")
        plt.close()
        print(f" -> 앙상블 중요도 비교 차트 저장 완료: {comp_file}")

    # 중요도 요약 CSV 저장
    csv_out = out_path / f"feature_importance_summary_{target_col}.csv"
    imp_df.to_csv(csv_out, index=False, encoding="utf-8-sig")
    print(f" -> 변수 중요도 요약표 CSV 저장 완료: {csv_out}")

    return imp_df


# =====================================================================
# 5. 메인 실행부 (Experiment 1: 지수 변화량 예측 / Experiment 2: 차년도 지수 수준 예측)
# =====================================================================
def main():
    csv_path = "panel_edit_foreigner_nidx_univ_2015_2025_261020.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {csv_path}")

    print("=" * 75)
    print(" [분석 1] 차년도 지방소멸위험지수 변화량(Δext_idx = ext_idx_{t+1} - ext_idx_t) 예측")
    print("  - 목적: 어떤 지역 특성(외국인·다문화·대학·경제)이 지방소멸 속도를 늦추거나 완화하는가?")
    print("=" * 75)
    df_delta, feat_delta, target_delta = load_and_preprocess_data(
        csv_path=csv_path,
        target_mode="delta",
        include_baseline_idx=True,  # 초기 지수 수준(ext_idx_t)을 통제한 상태에서의 순수 한계효과 확인
        use_ratio_features_only=True,
    )
    res_delta, X_tr_d, X_te_d, y_tr_d, y_te_d = train_and_evaluate(
        df_delta, feat_delta, target_delta, split_mode="temporal", test_start_year=2023
    )
    imp_delta = analyze_and_plot_importances(
        res_delta, X_te_d, y_te_d, target_col=target_delta, output_dir="outputs/delta_model"
    )

    print("\n[분석 1 결과 요약: Δext_idx 상위 10개 영향 변수]")
    display_cols = [c for c in ["feature_kor", "SHAP_mean_abs", "SHAP_direction", "LightGBM_imp", "XGBoost_imp", "RandomForest_imp"] if c in imp_delta.columns]
    print(imp_delta[display_cols].head(10).to_string(index=False))

    print("\n" + "=" * 75)
    print(" [분석 2] 차년도 지방소멸위험지수 수준(ext_idx_{t+1}) 직접 예측 (초기 지수 제외)")
    print("  - 목적: 인구 변수를 전혀 쓰지 않고 외국인·다문화·대학·경제 지표만으로")
    print("          차년도 지방소멸위험 수준을 얼마나 설명할 수 있으며 핵심 변수는 무엇인가?")
    print("=" * 75)
    df_lvl, feat_lvl, target_lvl = load_and_preprocess_data(
        csv_path=csv_path,
        target_mode="next_level",
        include_baseline_idx=False,  # 당해연도 지수(ext_idx_t) 없이 순수 외생 설명변수로만 예측
        use_ratio_features_only=True,
    )
    res_lvl, X_tr_l, X_te_l, y_tr_l, y_te_l = train_and_evaluate(
        df_lvl, feat_lvl, target_lvl, split_mode="group"  # 읍면동 Group 분할로 지역 간 격차 결정요인 탐색
    )
    imp_lvl = analyze_and_plot_importances(
        res_lvl, X_te_l, y_te_l, target_col=target_lvl, output_dir="outputs/level_model"
    )

    print("\n[분석 2 결과 요약: ext_idx_{t+1} 상위 10개 영향 변수]")
    display_cols_lvl = [c for c in ["feature_kor", "SHAP_mean_abs", "SHAP_direction", "LightGBM_imp", "XGBoost_imp", "RandomForest_imp"] if c in imp_lvl.columns]
    print(imp_lvl[display_cols_lvl].head(10).to_string(index=False))
    print("\n[완료] 모든 분석 결과표(.csv)와 SHAP 그래프(.png)가 'outputs/' 폴더에 저장되었습니다.")


if __name__ == "__main__":
    main()
