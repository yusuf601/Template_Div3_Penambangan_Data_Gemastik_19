"""Sparse TF-IDF pipeline supporting linear estimators, SGD, Ridge, and LightGBM."""

from typing import Any, Optional
import warnings

from lightgbm import LGBMClassifier, LGBMRegressor, early_stopping
import numpy as np
import pandas as pd
import scipy.sparse
import scipy.special
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge, RidgeClassifier, SGDClassifier, SGDRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import LinearSVC

from utils.core.errors import ConfigError
from utils.text.base import TextPipelineBase
from utils.text.config import TextConfig


class TfidfPipeline(TextPipelineBase):
    """TF-IDF sparse feature pipeline with linear or LightGBM estimators."""

    _use_tabular_stats: bool = True

    def __init__(self, cfg: TextConfig) -> None:
        self.model_name = f"tfidf_{cfg.estimator}"
        super().__init__(cfg)

    def build_model(self, fold: int) -> dict[str, Any]:
        """Return fold identifier and deterministic seed."""
        return {"fold": fold, "seed": self.cfg.seed + max(int(fold), 0)}

    def fit_predict_fold(
        self,
        model: Any,
        tr: pd.DataFrame,
        va: pd.DataFrame,
        te: pd.DataFrame,
        fold: int,
    ) -> tuple[Any, np.ndarray, np.ndarray]:
        """Vectorize text as sparse CSR matrices, fit estimator, and predict on va and te."""
        feat_names: list[str] = []
        blocks_tr: list[scipy.sparse.spmatrix] = []
        blocks_va: list[scipy.sparse.spmatrix] = []
        blocks_te: list[scipy.sparse.spmatrix] = []

        # 1. Leak-free pair TF-IDF cosine similarity features
        pair_dense_tr: list[np.ndarray] = []
        pair_dense_va: list[np.ndarray] = []
        pair_dense_te: list[np.ndarray] = []
        pair_dense_names: list[str] = []

        if len(self.cfg.text_cols) >= 2 and self.cfg.add_pair_features:
            col_a = f"{self.cfg.text_cols[0]}__clean"
            col_b = f"{self.cfg.text_cols[1]}__clean"
            concat_text = pd.concat([tr[col_a], tr[col_b]], ignore_index=True)

            min_df_pair = self.cfg.min_df if len(tr) >= self.cfg.min_df else 1
            shared_word_vec = TfidfVectorizer(
                ngram_range=self.cfg.word_ngram_range,
                max_features=self.cfg.max_features_word,
                min_df=min_df_pair,
                sublinear_tf=self.cfg.sublinear_tf,
            )
            try:
                shared_word_vec.fit(concat_text)
            except ValueError:
                shared_word_vec = TfidfVectorizer(ngram_range=(1, 1), min_df=1).fit(concat_text)

            w_a_tr = shared_word_vec.transform(tr[col_a])
            w_b_tr = shared_word_vec.transform(tr[col_b])
            cos_w_tr = np.asarray(w_a_tr.multiply(w_b_tr).sum(axis=1)).ravel()

            cos_w_va = np.zeros(len(va), dtype=float)
            if len(va) > 0:
                w_a_va = shared_word_vec.transform(va[col_a])
                w_b_va = shared_word_vec.transform(va[col_b])
                cos_w_va = np.asarray(w_a_va.multiply(w_b_va).sum(axis=1)).ravel()

            w_a_te = shared_word_vec.transform(te[col_a])
            w_b_te = shared_word_vec.transform(te[col_b])
            cos_w_te = np.asarray(w_a_te.multiply(w_b_te).sum(axis=1)).ravel()

            pair_dense_tr.append(cos_w_tr)
            pair_dense_va.append(cos_w_va)
            pair_dense_te.append(cos_w_te)
            pair_dense_names.append("pair__tfidf_word_cos")

            if self.cfg.use_char_ngrams:
                shared_char_vec = TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=self.cfg.char_ngram_range,
                    max_features=self.cfg.max_features_char,
                    min_df=min_df_pair,
                    sublinear_tf=self.cfg.sublinear_tf,
                )
                try:
                    shared_char_vec.fit(concat_text)
                except ValueError:
                    shared_char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), min_df=1).fit(concat_text)

                c_a_tr = shared_char_vec.transform(tr[col_a])
                c_b_tr = shared_char_vec.transform(tr[col_b])
                cos_c_tr = np.asarray(c_a_tr.multiply(c_b_tr).sum(axis=1)).ravel()

                cos_c_va = np.zeros(len(va), dtype=float)
                if len(va) > 0:
                    c_a_va = shared_char_vec.transform(va[col_a])
                    c_b_va = shared_char_vec.transform(va[col_b])
                    cos_c_va = np.asarray(c_a_va.multiply(c_b_va).sum(axis=1)).ravel()

                c_a_te = shared_char_vec.transform(te[col_a])
                c_b_te = shared_char_vec.transform(te[col_b])
                cos_c_te = np.asarray(c_a_te.multiply(c_b_te).sum(axis=1)).ravel()

                pair_dense_tr.append(cos_c_tr)
                pair_dense_va.append(cos_c_va)
                pair_dense_te.append(cos_c_te)
                pair_dense_names.append("pair__tfidf_char_cos")

        # 2. Word and character TF-IDF vectorization per text column
        for col in self.cfg.text_cols:
            col_clean = f"{col}__clean"
            min_df_col = self.cfg.min_df if len(tr) >= self.cfg.min_df else 1

            word_vec = TfidfVectorizer(
                ngram_range=self.cfg.word_ngram_range,
                max_features=self.cfg.max_features_word,
                min_df=min_df_col,
                sublinear_tf=self.cfg.sublinear_tf,
            )
            try:
                w_tr = word_vec.fit_transform(tr[col_clean])
            except ValueError:
                word_vec = TfidfVectorizer(ngram_range=(1, 1), min_df=1)
                w_tr = word_vec.fit_transform(tr[col_clean])

            w_va = word_vec.transform(va[col_clean]) if len(va) > 0 else scipy.sparse.csr_matrix((0, w_tr.shape[1]))
            w_te = word_vec.transform(te[col_clean])

            blocks_tr.append(w_tr)
            blocks_va.append(w_va)
            blocks_te.append(w_te)

            w_names = [f"{col}__w_{fn}" for fn in word_vec.get_feature_names_out()]
            feat_names.extend(w_names)

            if self.cfg.use_char_ngrams:
                char_vec = TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=self.cfg.char_ngram_range,
                    max_features=self.cfg.max_features_char,
                    min_df=min_df_col,
                    sublinear_tf=self.cfg.sublinear_tf,
                )
                try:
                    c_tr = char_vec.fit_transform(tr[col_clean])
                except ValueError:
                    char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), min_df=1)
                    c_tr = char_vec.fit_transform(tr[col_clean])

                c_va = char_vec.transform(va[col_clean]) if len(va) > 0 else scipy.sparse.csr_matrix((0, c_tr.shape[1]))
                c_te = char_vec.transform(te[col_clean])

                blocks_tr.append(c_tr)
                blocks_va.append(c_va)
                blocks_te.append(c_te)

                c_names = [f"{col}__c_{fn}" for fn in char_vec.get_feature_names_out()]
                feat_names.extend(c_names)

        # 3. Categorical one-hot encoding
        if self.cfg.cat_cols:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
            cat_tr = ohe.fit_transform(tr[self.cfg.cat_cols])
            cat_va = ohe.transform(va[self.cfg.cat_cols]) if len(va) > 0 else scipy.sparse.csr_matrix((0, cat_tr.shape[1]))
            cat_te = ohe.transform(te[self.cfg.cat_cols])

            blocks_tr.append(cat_tr)
            blocks_va.append(cat_va)
            blocks_te.append(cat_te)

            feat_names.extend(list(ohe.get_feature_names_out(self.cfg.cat_cols)))

        # 4. Dense scaled numeric, tabular stats, and pair cosine features
        num_stat_cols = list(dict.fromkeys(self.cfg.num_cols + self._stat_cols))
        dense_tr_cols: list[np.ndarray] = []
        dense_va_cols: list[np.ndarray] = []
        dense_te_cols: list[np.ndarray] = []
        dense_col_names: list[str] = []

        if num_stat_cols:
            dense_tr_cols.append(tr[num_stat_cols].to_numpy(dtype=float))
            dense_va_cols.append(
                va[num_stat_cols].to_numpy(dtype=float) if len(va) > 0 else np.empty((0, len(num_stat_cols)))
            )
            dense_te_cols.append(te[num_stat_cols].to_numpy(dtype=float))
            dense_col_names.extend(num_stat_cols)

        if pair_dense_names:
            arr_pair_tr = np.column_stack(pair_dense_tr)
            arr_pair_va = np.column_stack(pair_dense_va) if len(va) > 0 else np.empty((0, len(pair_dense_names)))
            arr_pair_te = np.column_stack(pair_dense_te)

            dense_tr_cols.append(arr_pair_tr)
            dense_va_cols.append(arr_pair_va)
            dense_te_cols.append(arr_pair_te)
            dense_col_names.extend(pair_dense_names)

        mat_tr: Optional[np.ndarray] = None
        mat_va: Optional[np.ndarray] = None
        mat_te: Optional[np.ndarray] = None

        if dense_col_names:
            mat_tr = np.column_stack(dense_tr_cols)
            mat_va = np.column_stack(dense_va_cols) if len(va) > 0 else np.empty((0, len(dense_col_names)))
            mat_te = np.column_stack(dense_te_cols)

            blocks_tr.append(scipy.sparse.csr_matrix(mat_tr))
            blocks_va.append(scipy.sparse.csr_matrix(mat_va))
            blocks_te.append(scipy.sparse.csr_matrix(mat_te))
            feat_names.extend(dense_col_names)

        X_tr_sparse = scipy.sparse.hstack(blocks_tr, format="csr")
        X_va_sparse = (
            scipy.sparse.hstack(blocks_va, format="csr")
            if len(va) > 0
            else scipy.sparse.csr_matrix((0, X_tr_sparse.shape[1]))
        )
        X_te_sparse = scipy.sparse.hstack(blocks_te, format="csr")

        # 5. Estimator fitting and predicting
        fold_seed = self.cfg.seed + max(int(fold), 0)
        task = self.cfg.task
        estimator_type = self.cfg.estimator
        y_tr = tr[self.cfg.target].to_numpy()
        y_va = va[self.cfg.target].to_numpy() if len(va) > 0 else np.empty(0)

        if estimator_type == "lgbm":
            n_svd = min(self.cfg.svd_components, max(1, X_tr_sparse.shape[1] - 1))
            if X_tr_sparse.shape[1] <= self.cfg.svd_components:
                X_tr_lgb = X_tr_sparse.toarray()
                X_va_lgb = X_va_sparse.toarray() if len(va) > 0 else np.empty((0, X_tr_sparse.shape[1]))
                X_te_lgb = X_te_sparse.toarray()
                lgb_feat_names = feat_names
            else:
                svd = TruncatedSVD(n_components=n_svd, random_state=fold_seed)
                svd_tr = svd.fit_transform(X_tr_sparse)
                svd_va = svd.transform(X_va_sparse) if len(va) > 0 else np.empty((0, n_svd))
                svd_te = svd.transform(X_te_sparse)
                lgb_feat_names = [f"svd_{i}" for i in range(n_svd)]

                if dense_col_names and mat_tr is not None and mat_te is not None:
                    X_tr_lgb = np.column_stack([svd_tr, mat_tr])
                    X_va_lgb = (
                        np.column_stack([svd_va, mat_va])
                        if len(va) > 0
                        else np.empty((0, n_svd + len(dense_col_names)))
                    )
                    X_te_lgb = np.column_stack([svd_te, mat_te])
                    lgb_feat_names.extend(dense_col_names)
                else:
                    X_tr_lgb = svd_tr
                    X_va_lgb = svd_va
                    X_te_lgb = svd_te

            clf_cls = LGBMRegressor if task == "regression" else LGBMClassifier

            if fold >= 0:
                lgb_est = clf_cls(
                    n_estimators=500,
                    random_state=fold_seed,
                    n_jobs=-1,
                    verbosity=-1,
                )
                lgb_est.fit(
                    X_tr_lgb,
                    y_tr,
                    eval_set=[(X_va_lgb, y_va)],
                    callbacks=[early_stopping(30, verbose=False)],
                )
                best_iter = getattr(lgb_est, "best_iteration_", 500) or 500
                self.fold_info[fold]["best_iteration"] = int(best_iter)
            else:
                refit_n = max(1, self.refit_iterations() or 100)
                lgb_est = clf_cls(
                    n_estimators=refit_n,
                    random_state=fold_seed,
                    n_jobs=-1,
                    verbosity=-1,
                )
                lgb_est.fit(X_tr_lgb, y_tr)

            fitted_model = lgb_est

            if task == "regression":
                va_pred = lgb_est.predict(X_va_lgb) if len(va) > 0 else np.empty(0)
                te_pred = lgb_est.predict(X_te_lgb)
            elif task == "binary":
                proba_va = lgb_est.predict_proba(X_va_lgb) if len(va) > 0 else np.empty((0, 2))
                proba_te = lgb_est.predict_proba(X_te_lgb)
                m_classes = list(lgb_est.classes_)
                pos_idx = m_classes.index(1) if 1 in m_classes else -1
                va_pred = proba_va[:, pos_idx] if len(va) > 0 else np.empty(0)
                te_pred = proba_te[:, pos_idx]
            else:
                proba_va = (
                    lgb_est.predict_proba(X_va_lgb)
                    if len(va) > 0
                    else np.empty((0, len(self.classes_) if self.classes_ else 0))
                )
                proba_te = lgb_est.predict_proba(X_te_lgb)
                va_pred, te_pred = self._align_multiclass_proba(
                    proba_va, proba_te, list(lgb_est.classes_), len(va), len(te)
                )

            if fold >= 0 and hasattr(lgb_est, "feature_importances_"):
                imp = np.asarray(lgb_est.feature_importances_, dtype=float).ravel()
                self._record_importances(fold, imp, lgb_feat_names)

        else:
            # Linear / SGD / Ridge estimators
            if estimator_type == "logistic":
                fitted_model = LogisticRegression(
                    C=self.cfg.c_reg, solver="lbfgs", max_iter=1000, random_state=fold_seed
                )
                fitted_model.fit(X_tr_sparse, y_tr)

            elif estimator_type == "linear_svc":
                counts = pd.Series(y_tr).value_counts()
                min_class_count = int(counts.min()) if len(counts) > 0 else 0
                if min_class_count < 2:
                    warnings.warn(
                        "min_class_count < 2 in tr; falling back from LinearSVC to LogisticRegression",
                        UserWarning,
                        stacklevel=2,
                    )
                    fitted_model = LogisticRegression(
                        C=self.cfg.c_reg, solver="lbfgs", max_iter=1000, random_state=fold_seed
                    )
                    fitted_model.fit(X_tr_sparse, y_tr)
                else:
                    base_svc = LinearSVC(C=self.cfg.c_reg, random_state=fold_seed)
                    fitted_model = CalibratedClassifierCV(estimator=base_svc, cv=min(3, min_class_count))
                    fitted_model.fit(X_tr_sparse, y_tr)

            elif estimator_type == "sgd":
                if task in ("binary", "multiclass"):
                    fitted_model = SGDClassifier(
                        loss="log_loss",
                        alpha=1.0 / (self.cfg.c_reg * max(len(tr), 1)),
                        random_state=fold_seed,
                    )
                else:
                    fitted_model = SGDRegressor(
                        alpha=1.0 / (self.cfg.c_reg * max(len(tr), 1)),
                        random_state=fold_seed,
                    )
                fitted_model.fit(X_tr_sparse, y_tr)

            elif estimator_type == "ridge":
                if task in ("binary", "multiclass"):
                    fitted_model = RidgeClassifier(alpha=1.0 / self.cfg.c_reg, random_state=fold_seed)
                else:
                    fitted_model = Ridge(alpha=1.0 / self.cfg.c_reg, random_state=fold_seed)
                fitted_model.fit(X_tr_sparse, y_tr)

            else:
                raise ConfigError(
                    f"Estimator {estimator_type!r} tidak didukung.",
                    "Pilih estimator dari ('logistic', 'linear_svc', 'sgd', 'ridge', 'lgbm').",
                )

            # Generate predictions
            if task == "regression":
                va_pred = fitted_model.predict(X_va_sparse) if len(va) > 0 else np.empty(0)
                te_pred = fitted_model.predict(X_te_sparse)

            elif task == "binary":
                if hasattr(fitted_model, "predict_proba"):
                    proba_va = fitted_model.predict_proba(X_va_sparse) if len(va) > 0 else np.empty((0, 2))
                    proba_te = fitted_model.predict_proba(X_te_sparse)
                    m_classes = list(fitted_model.classes_)
                    pos_idx = m_classes.index(1) if 1 in m_classes else -1
                    va_pred = proba_va[:, pos_idx] if len(va) > 0 else np.empty(0)
                    te_pred = proba_te[:, pos_idx]
                elif isinstance(fitted_model, RidgeClassifier):
                    df_va = fitted_model.decision_function(X_va_sparse) if len(va) > 0 else np.empty(0)
                    df_te = fitted_model.decision_function(X_te_sparse)
                    va_pred = 1.0 / (1.0 + np.exp(-df_va)) if len(va) > 0 else np.empty(0)
                    te_pred = 1.0 / (1.0 + np.exp(-df_te))
                else:
                    va_pred = fitted_model.predict(X_va_sparse) if len(va) > 0 else np.empty(0)
                    te_pred = fitted_model.predict(X_te_sparse)

            else:
                # Multiclass
                if hasattr(fitted_model, "predict_proba"):
                    proba_va = (
                        fitted_model.predict_proba(X_va_sparse)
                        if len(va) > 0
                        else np.empty((0, len(fitted_model.classes_)))
                    )
                    proba_te = fitted_model.predict_proba(X_te_sparse)
                    m_cls = list(fitted_model.classes_)
                elif isinstance(fitted_model, RidgeClassifier):
                    df_va = (
                        fitted_model.decision_function(X_va_sparse)
                        if len(va) > 0
                        else np.empty((0, len(fitted_model.classes_)))
                    )
                    df_te = fitted_model.decision_function(X_te_sparse)
                    proba_va = scipy.special.softmax(df_va, axis=1) if len(va) > 0 else np.empty((0, len(fitted_model.classes_)))
                    proba_te = scipy.special.softmax(df_te, axis=1)
                    m_cls = list(fitted_model.classes_)
                else:
                    proba_va = fitted_model.predict(X_va_sparse) if len(va) > 0 else np.empty(0)
                    proba_te = fitted_model.predict(X_te_sparse)
                    m_cls = list(getattr(fitted_model, "classes_", []))

                va_pred, te_pred = self._align_multiclass_proba(proba_va, proba_te, m_cls, len(va), len(te))

            # Record feature importances
            if fold >= 0:
                imp = None
                if hasattr(fitted_model, "coef_"):
                    imp = np.mean(np.abs(np.atleast_2d(fitted_model.coef_)), axis=0).ravel()
                elif hasattr(fitted_model, "calibrated_classifiers_"):
                    coefs = []
                    for c in fitted_model.calibrated_classifiers_:
                        sub_est = getattr(c, "estimator", getattr(c, "base_estimator", None))
                        if hasattr(sub_est, "coef_"):
                            coefs.append(np.mean(np.abs(np.atleast_2d(sub_est.coef_)), axis=0).ravel())
                    if coefs:
                        imp = np.mean(coefs, axis=0)

                if imp is not None:
                    self._record_importances(fold, imp, feat_names)

        return fitted_model, va_pred, te_pred

    def _align_multiclass_proba(
        self,
        proba_va: np.ndarray,
        proba_te: np.ndarray,
        model_classes: list[Any],
        n_va: int,
        n_te: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Align model class probabilities to the full set of pipeline classes_."""
        classes = self.classes_ or model_classes
        k = len(classes)
        aligned_va = np.zeros((n_va, k), dtype=float)
        aligned_te = np.zeros((n_te, k), dtype=float)

        for j, cls_val in enumerate(classes):
            if cls_val in model_classes:
                idx = model_classes.index(cls_val)
                if n_va > 0:
                    aligned_va[:, j] = proba_va[:, idx]
                if n_te > 0:
                    aligned_te[:, j] = proba_te[:, idx]

        if n_va > 0:
            s_va = aligned_va.sum(axis=1, keepdims=True)
            s_va[s_va == 0] = 1.0
            aligned_va = aligned_va / s_va

        if n_te > 0:
            s_te = aligned_te.sum(axis=1, keepdims=True)
            s_te[s_te == 0] = 1.0
            aligned_te = aligned_te / s_te

        return aligned_va, aligned_te

    def _record_importances(self, fold: int, imp: np.ndarray, names: list[str]) -> None:
        """Store top_k_importances into fold_info."""
        if len(imp) == len(names) and len(imp) > 0:
            top_k = min(self.cfg.top_k_importances, len(imp))
            top_indices = np.argsort(-imp)[:top_k]
            imp_dict = {names[i]: float(imp[i]) for i in top_indices}
            self.fold_info[fold]["importances"] = imp_dict


__all__ = [
    "TfidfPipeline",
]
