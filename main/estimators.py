import warnings

import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    Lasso,
    LassoCV,
    LinearRegression,
    LogisticRegression,
    LogisticRegressionCV,
    RidgeCV,
)
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier, MLPRegressor

eps_trim = 1e-2

# Standard Adjustment estimators based sci-kit learn nuisance estimators
# Feedforward Neural Network (Multilayer perceptron)
MLPR_params = {
    "solver": "adam",
    "max_iter": 750,
    "learning_rate_init": 1e-2,
    "hidden_layer_sizes": (100,),
}
MLPC_params = {
    "solver": "adam",
    "max_iter": 750,
    "learning_rate_init": 1e-2,
    "hidden_layer_sizes": (50,),
}

# Random Forest
RFR_params = {"n_estimators": 150, "max_features": "sqrt", "max_depth": 3}
RFC_params = {
    "n_estimators": 200,
    "max_features": "log2",
    "max_depth": 10,
    "bootstrap": True,
    "criterion": "log_loss",
}

# Gradient Boosted Forest
GBFC_params = {
    "n_iter_no_change": None,
    "n_estimators": 100,
    "min_samples_split": 2,
    "max_features": "log2",
    "max_depth": 3,
}
GBFR_params = {
    "n_iter_no_change": None,
    "n_estimators": 75,
    "min_samples_split": 2,
    "max_features": "log2",
    "max_depth": 3,
}

# Hist gradient boosting
HGBFC_params = {
    "max_iter": 1000,
    "max_depth": 4,
    "learning_rate": 0.01,
    "early_stopping": True,
}
HGBFR_params = {
    "max_iter": 1000,
    "max_depth": 4,
    "learning_rate": 0.01,
    "early_stopping": True,
}

# Logistic
Logis_params = {"penalty": None, "max_iter": 1000, "tol": 1e-3}
LogisNet__params = {
    "penalty": "elasticnet",
    "C": 7 * 1e-3,
    "l1_ratio": 0.1,
    "solver": "saga",
    "max_iter": 1000,
}
LogisCV_params = {
    "cv": 3,
    "penalty": "elasticnet",
    "l1_ratios": [0.1, 0.5, 0.9],
    "solver": "saga",
}

# Lasso
Lasso_params = {"alpha": 1e-4, "max_iter": 1000}
LassoCV_params = {"cv": 3, "n_alphas": 50}
RidgeCV_params = {"alphas": np.logspace(-4, 0, 5), "cv": 5}


# The classes below are temporary fixes to avoid warnings (https://github.com/numpy/numpy/issues/28687)
class MyLogisticRegression(LogisticRegression):
    def fit(self, X, y, sample_weight=None):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return super().fit(X, y, sample_weight=sample_weight)

    def predict_proba(self, X):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return super().predict_proba(X)


class MyLinearRegression(LinearRegression):
    def fit(self, X, y, sample_weight=None):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return super().fit(X, y, sample_weight=sample_weight)

    def predict(self, X):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return super().predict(X)


def create_classifier(name: str):
    match name:
        case "MLP":
            return MLPClassifier(**MLPC_params)
        case "RF":
            return RandomForestClassifier(**RFC_params)
        case "Logistic":
            return MyLogisticRegression(**Logis_params)
        case "Logistic-net":
            return LogisticRegression(**LogisNet__params)
        case "Logistic-CV":
            return LogisticRegressionCV(**LogisCV_params)
        case "GBF":
            return GradientBoostingClassifier(**GBFC_params)
        case "histGBF":
            return HistGradientBoostingClassifier(**HGBFC_params)
        case "RidgeCV":
            return LogisticRegressionCV()
        case _:
            print("unknown classifier", name)


def create_regressor(name: str):
    match name:
        case "MLP":
            return MLPRegressor(**MLPR_params)
        case "RF":
            return RandomForestRegressor(**RFR_params)
        case "Lasso":
            return Lasso(**Lasso_params)
        case "LassoCV":
            return LassoCV(**LassoCV_params)
        case "GBF":
            return GradientBoostingRegressor(**HGBFR_params)
        case "histGBF":
            return HistGradientBoostingRegressor(**HGBFC_params)
        case "RidgeCV":
            return RidgeCV(**RidgeCV_params)
        case "OLS":
            return MyLinearRegression()
        case _:
            print("unknown regressor:", name)


############ Single index regression ############
class SingleIndexNetwork(nn.Module):
    def __init__(self, shared_index, hidden_dim=100):
        super().__init__()
        self.index_layer = shared_index
        self.hidden = nn.Linear(1, hidden_dim)
        self.activation = nn.ReLU()
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.index_layer(x)
        x = self.hidden(x)
        x = self.activation(x)
        x = self.output(x)

        return x


# Single index regressor as a sklearn regression class


class SingleIndexRegressor(RegressorMixin, BaseEstimator):
    def __init__(
        self, hidden_dim=100, lr=1e-3, n_iter=1200, joint_regression=False
    ):
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.n_iter = n_iter
        self.joint_regression = joint_regression

    def fit(self, X, y, initial=None):
        if self.joint_regression:
            return self._fit_joint(X, y, initial)  # fit Y given T,W jointly
        else:
            # fit Y given W on each strata T==0 and T==1.
            return self._fit_strat(X, y, initial)

    def _fit_strat(self, X, y, initial=None):
        self.index = nn.Linear(X.shape[1] - 1, 1, bias=False)
        if initial is not None:
            self.index.weight.data = torch.Tensor(initial.reshape(1, -1))
        self.g0 = SingleIndexNetwork(
            shared_index=self.index, hidden_dim=self.hidden_dim
        )
        self.g1 = SingleIndexNetwork(
            shared_index=self.index, hidden_dim=self.hidden_dim
        )

        T = X[:, 0]  # We save treatment in the first column
        W0 = torch.Tensor(X[T == 0][:, 1:])
        Y0 = torch.Tensor(y[T == 0]).unsqueeze(1)
        W1 = torch.Tensor(X[T == 1][:, 1:])
        Y1 = torch.Tensor(y[T == 1]).unsqueeze(1)

        opt0 = torch.optim.Adam(self.g0.parameters(), self.lr)
        opt1 = torch.optim.Adam(self.g1.parameters(), self.lr)
        crit = nn.MSELoss()

        for _ in range(self.n_iter):
            opt0.zero_grad()
            out0 = self.g0(W0)
            loss = crit(out0, Y0)
            loss.backward()
            opt0.step()

            opt1.zero_grad()
            out1 = self.g1(W1)
            loss = crit(out1, Y1)
            loss.backward()
            opt1.step()
        return self

    def _fit_joint(self, X, y, initial=None):
        self.index = nn.Linear(X.shape[1], 1, bias=False)
        if initial is not None:
            self.index.weight.data = torch.Tensor(
                np.insert(initial, 0, 0).reshape(1, -1)
            )
        self.g = SingleIndexNetwork(
            shared_index=self.index, hidden_dim=self.hidden_dim
        )

        tX = torch.Tensor(X)
        tY = torch.Tensor(y).unsqueeze(1)
        opt = torch.optim.Adam(self.g.parameters(), self.lr)
        crit = nn.MSELoss()
        for _ in range(self.n_iter):
            opt.zero_grad()
            out = self.g(tX)
            loss = crit(out, tY)
            loss.backward()
            opt.step()
        return self

    def predict_conditional(self, t, W):
        if self.joint_regression:
            tX = torch.Tensor(np.hstack([t * np.ones((len(W), 1)), W])).float()
            return self.g(tX).detach().numpy().flatten()
        tW = torch.Tensor(W).float()
        gt = self.g1 if t == 1 else self.g0
        return gt(tW).detach().numpy().flatten()

    def predict(self, X):
        if self.joint_regression:
            tX = torch.Tensor(X).float()
            return self.g(tX).detach().numpy().flatten()
        return (1 - X[:, 0]) * self.predict_conditional(0, X[:, 1:]) + X[
            :, 0
        ] * self.predict_conditional(1, X[:, 1:])

    def partial_predict(self, X):
        if self.joint_regression:
            tX = torch.Tensor(np.hstack([np.zeros((len(X), 1)), X])).float()
        else:
            tX = torch.Tensor(X).float()
        return self.index(tX).detach().numpy().flatten()

    def get_index(self):
        idx = self.index.weight.detach().numpy().flatten()
        return idx[1:] if self.joint_regression else idx


##### ##### ##### ##### Adjusted means ##### ##### ##### #####
def stack_TW(t, w):
    return np.hstack([t.reshape(-1, 1), w])


def unadjusted(T, Y, t=1):
    return Y[T == t].mean()


def IM_oracle(T, W, Z, Y, f, m, trim=eps_trim):
    v_reg = f(Z)
    IM_reg = v_reg.mean()
    Var_reg = v_reg.var()

    v_aipw = v_reg + T * (Y - v_reg) / m(W)
    IM_aipw = v_aipw.mean()
    Var_aipw = v_aipw.var()

    clf = create_classifier("Logistic").fit(Z.reshape(-1, 1), T)
    prop = np.clip(clf.predict_proba(Z.reshape(-1, 1)), trim, 1 - trim)
    v_oapw = v_reg + T * (Y - v_reg) / prop[:, 1]
    IM_oapw = v_oapw.mean()
    Var_oapw = v_oapw.var()

    return (IM_reg, IM_aipw, IM_oapw), (Var_reg, Var_aipw, Var_oapw)


def IM_est(
    T,
    W,
    Y,
    classifier="Logistic",
    regressor="OLS",
    trim=eps_trim,
    t=1,
    joint_regression=True,
):
    clf1 = create_classifier(classifier).fit(W, T)
    prop = np.clip(clf1.predict_proba(W), trim, 1 - trim)

    v_ipw = Y * (T == t) / prop[:, t]
    IM_ipw = v_ipw.mean()
    Var_ipw = v_ipw.var()

    if joint_regression:
        regrt = create_regressor(regressor).fit(stack_TW(T, W), Y)
        Qt = regrt.predict(stack_TW(t * np.ones_like(T), W))
    else:
        regrt = create_regressor(regressor).fit(W[T == t], Y[T == t])
        Qt = regrt.predict(W)

    IM_reg = Qt.mean()
    Var_reg = Qt.var()

    v_aipw = Qt + (T == t) * (Y - Qt) / prop[:, t]
    IM_aipw = v_aipw.mean()
    Var_aipw = v_aipw.var()

    # Pruned propensity
    Qdata = np.array([Qt]).T
    clf_pru = create_classifier(classifier).fit(Qdata, T)
    propQ = np.clip(clf_pru.predict_proba(Qdata), trim, 1 - trim)
    v_pru = Qt + ((T == t) * (Y - Qt) / propQ[:, t])
    IM_pru = v_pru.mean()
    Var_pru = v_pru.var()

    return (IM_ipw, IM_reg, IM_aipw, IM_pru), (
        Var_ipw,
        Var_reg,
        Var_aipw,
        Var_pru,
    )


def IM_est_cf(
    T,
    W,
    Y,
    classifier="Logistic",
    regressor="OLS",
    trim=eps_trim,
    t=1,
    n_folds=3,
    joint_regression=True,
):
    (im_ipw, im_reg, im_aipw, im_obpw) = ([] for _ in range(4))
    (var_ipw, var_reg, var_aipw, var_obpw) = ([] for _ in range(4))

    kf = KFold(n_splits=n_folds)

    for idx_train, idx_test in kf.split(W):
        Wa, Ta, Ya = W[idx_train], T[idx_train], Y[idx_train]
        Wb, Tb, Yb = W[idx_test], T[idx_test], Y[idx_test]

        clf1 = create_classifier(classifier).fit(Wa, Ta)
        prop = np.clip(clf1.predict_proba(Wb), trim, 1 - trim)
        v_ipw = Yb * (Tb == t) / prop[:, t]
        im_ipw.append(v_ipw.mean())
        var_ipw.append(v_ipw.var())

        if joint_regression:
            regrt = create_regressor(regressor).fit(stack_TW(Ta, Wa), Ya)
            Qa = regrt.predict(stack_TW(t * np.ones_like(Ta), Wa))
            Qb = regrt.predict(stack_TW(t * np.ones_like(Tb), Wb))
        else:
            regr = create_regressor(regressor).fit(Wa[Ta == t], Ya[Ta == t])
            Qa = regr.predict(Wa)
            Qb = regr.predict(Wb)

        im_reg.append(Qb.mean())
        var_reg.append(Qb.var())

        v_aipw = Qb + (Tb == t) * (Yb - Qb) / prop[:, t]
        im_aipw.append(v_aipw.mean())
        var_aipw.append(v_aipw.var())

        clf_pru = create_classifier(classifier).fit(Qa.reshape(-1, 1), Ta)
        propQ = np.clip(
            clf_pru.predict_proba(Qb.reshape(-1, 1)), trim, 1 - trim
        )
        v_obpw = Qb + (Tb == t) * (Yb - Qb) / propQ[:, t]
        im_obpw.append(v_obpw.mean())
        var_obpw.append(v_obpw.var())

    x = np.mean([im_ipw, im_reg, im_aipw, im_obpw], axis=1)
    y = np.mean([var_ipw, var_reg, var_aipw, var_obpw], axis=1)
    return x, y


def SI_IM(
    T,
    W,
    Y,
    clf="Logistic",
    hidden_dim=100,
    lr=1e-3,
    n_iter=1200,
    trim=eps_trim,
    t=1,
    initial_idx=None,
    joint_regression=True,
):
    sireg = SingleIndexRegressor(
        hidden_dim=hidden_dim,
        lr=lr,
        n_iter=n_iter,
        joint_regression=joint_regression,
    )

    sireg.fit(stack_TW(T, W), Y, initial_idx)
    Qt = sireg.predict_conditional(1, W)
    Z = sireg.partial_predict(W).reshape(-1, 1)

    im_reg = Qt.mean()
    var_reg = Qt.var()

    clfW = create_classifier(clf).fit(W, T)
    propW = np.clip(clfW.predict_proba(W), trim, 1 - trim)
    v_aipw_W = Qt + (T == t) * (Y - Qt) / propW[:, t]
    im_aipw_W = v_aipw_W.mean()
    var_aipw_W = v_aipw_W.var()

    clfZ = create_classifier(clf).fit(Z, T)
    propZ = np.clip(clfZ.predict_proba(Z), trim, 1 - trim)
    v_aipw_Z = Qt + (T == t) * (Y - Qt) / propZ[:, t]
    im_oapw = v_aipw_Z.mean()
    var_oapw = v_aipw_Z.var()

    clfQ = create_classifier(clf).fit(Qt.reshape(-1, 1), T)
    propQ = np.clip(clfQ.predict_proba(Qt.reshape(-1, 1)), trim, 1 - trim)
    v_aipw_Q = Qt + (T == t) * (Y - Qt) / propQ[:, t]
    im_obpw = v_aipw_Q.mean()
    var_obpw = v_aipw_Q.var()

    return (im_reg, im_aipw_W, im_oapw, im_obpw), (
        var_reg,
        var_aipw_W,
        var_oapw,
        var_obpw,
    )


def SI_IM_cf(
    T,
    W,
    Y,
    clf="Logistic",
    hidden_dim=100,
    lr=1e-3,
    n_iter=1200,
    trim=eps_trim,
    t=1,
    n_folds=3,
    initial_idx=None,
    joint_regression=True,
):
    (im_reg, im_aipw, im_oapw, im_obpw) = ([] for _ in range(4))
    (var_reg, var_aipw, var_oapw, var_obpw) = ([] for _ in range(4))

    kf = KFold(n_splits=n_folds)

    for idx_train, idx_test in kf.split(W):
        Wa, Ta, Ya = W[idx_train], T[idx_train], Y[idx_train]
        Wb, Tb, Yb = W[idx_test], T[idx_test], Y[idx_test]

        sireg = SingleIndexRegressor(
            hidden_dim=hidden_dim,
            lr=lr,
            n_iter=n_iter,
            joint_regression=joint_regression,
        )

        sireg.fit(stack_TW(Ta, Wa), Ya, initial=initial_idx)
        Qa = sireg.predict_conditional(t, Wa)
        Qb = sireg.predict_conditional(t, Wb)
        Za = sireg.partial_predict(Wa).reshape(-1, 1)
        Zb = sireg.partial_predict(Wb).reshape(-1, 1)

        im_reg.append(Qb.mean())
        var_reg.append(Qb.var())

        clfW = create_classifier(clf).fit(Wa, Ta)
        propW = np.clip(clfW.predict_proba(Wb), trim, 1 - trim)
        v_aipw = Qb + (Tb == t) * (Yb - Qb) / propW[:, t]
        im_aipw.append(v_aipw.mean())
        var_aipw.append(v_aipw.var())

        clfZ = create_classifier(clf).fit(Za, Ta)
        propZ = np.clip(clfZ.predict_proba(Zb), trim, 1 - trim)
        v_aipw_Z = Qb + (Tb == t) * (Yb - Qb) / propZ[:, t]
        im_oapw.append(v_aipw_Z.mean())
        var_oapw.append(v_aipw_Z.var())

        clfQ = create_classifier(clf).fit(Qa.reshape(-1, 1), Ta)
        propQ = np.clip(clfQ.predict_proba(Qb.reshape(-1, 1)), trim, 1 - trim)
        v_aipw_Q = Qb + (Tb == t) * (Yb - Qb) / propQ[:, t]
        im_obpw.append(v_aipw_Q.mean())
        var_obpw.append(v_aipw_Q.var())

    x = np.mean([im_reg, im_aipw, im_oapw, im_obpw], axis=1)
    y = np.mean([var_reg, var_aipw, var_oapw, var_obpw], axis=1)
    return x, y


############ Single index classification ############
class IndexNetworkBinary(nn.Module):
    def __init__(self, shared_index, hidden_dim=100):
        super().__init__()
        self.index_layer = shared_index
        self.hidden = nn.Linear(shared_index.out_features, hidden_dim)
        self.activation = nn.ReLU()
        self.output = nn.Linear(hidden_dim, 1)
        self.expit = nn.Sigmoid()

    def forward(self, x):
        x = self.index_layer(x)
        x = self.hidden(x)
        x = self.activation(x)
        x = self.output(x)
        x = self.expit(x)
        return x


class IndexClassifier(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        hidden_dim=100,
        index_dim=1,
        lr=0.001,
        n_iter=3000,
        l2pen=0.001,
        joint_training=True,
        initial_idx=None,
    ):
        self.hidden_dim = hidden_dim
        self.index_dim = index_dim
        self.lr = lr
        self.n_iter = n_iter
        self.l2pen = l2pen
        self.joint_training = joint_training
        self.classes_ = np.array([0, 1])  # needed for sklearn cv-scoring
        self.initial_idx = initial_idx

    def fit(self, X, y, initial=None):
        if initial is None:
            initial = self.initial_idx
        if self.joint_training:
            return self._fit_joint(X, y, initial)  # fit Y given T,W jointly
        else:
            # fit Y given W on each strata T==0 and T==1.
            return self._fit_strat(X, y, initial)

    def _fit_strat(self, X, y, initial=None):
        self.index = nn.Linear(X.shape[1] - 1, self.index_dim, bias=False)
        if initial is not None:
            self.index.weight.data = torch.Tensor(initial.reshape(1, -1))
        self.g0 = IndexNetworkBinary(
            shared_index=self.index, hidden_dim=self.hidden_dim
        )
        self.g1 = IndexNetworkBinary(
            shared_index=self.index, hidden_dim=self.hidden_dim
        )

        T = X[:, 0]  # We save treatment in the first column
        W0 = torch.Tensor(X[T == 0][:, 1:])
        Y0 = torch.Tensor(y[T == 0]).unsqueeze(1)
        W1 = torch.Tensor(X[T == 1][:, 1:])
        Y1 = torch.Tensor(y[T == 1]).unsqueeze(1)

        opt0 = torch.optim.Adam(
            self.g0.parameters(), self.lr, weight_decay=self.l2pen
        )
        opt1 = torch.optim.Adam(
            self.g1.parameters(), self.lr, weight_decay=self.l2pen
        )
        crit = nn.BCELoss()

        for _ in range(self.n_iter):
            opt0.zero_grad()
            out0 = self.g0(W0)
            loss = crit(out0, Y0)
            loss.backward()
            opt0.step()

            opt1.zero_grad()
            out1 = self.g1(W1)
            loss = crit(out1, Y1)
            loss.backward()
            opt1.step()
        return self

    def _fit_joint(self, X, y, initial=None):
        self.index = nn.Linear(X.shape[1], self.index_dim, bias=False)
        if initial is not None:
            self.index.weight.data = torch.Tensor(initial.reshape(1, -1))
        self.g = IndexNetworkBinary(
            shared_index=self.index, hidden_dim=self.hidden_dim
        )

        tX = torch.Tensor(X)
        tY = torch.Tensor(y).unsqueeze(1)
        opt = torch.optim.Adam(
            self.g.parameters(), self.lr, weight_decay=self.l2pen
        )
        crit = nn.BCELoss()
        for _ in range(self.n_iter):
            opt.zero_grad()
            out = self.g(tX)
            loss = crit(out, tY)
            loss.backward()
            opt.step()
        return self

    def predict_conditional(self, t, W):
        if self.joint_training:
            tX = torch.Tensor(np.hstack([t * np.ones((len(W), 1)), W])).float()
            return self.g(tX).detach().numpy().flatten()
        tW = torch.Tensor(W).float()
        gt = self.g1 if t == 1 else self.g0
        return gt(tW).detach().numpy().flatten()

    def predict_proba(self, X):
        if self.joint_training:
            tX = torch.Tensor(X).float()
            p = self.g(tX).detach().numpy().flatten()
        else:
            p = (1 - X[:, 0]) * self.predict_conditional(0, X[:, 1:]) + X[
                :, 0
            ] * self.predict_conditional(1, X[:, 1:])
        return np.vstack([1 - p, p]).T

    def predict(self, X):
        return np.round(self.predict_proba(X)[:, 1])

    def partial_predict(self, X):
        if self.joint_training:
            tX = torch.Tensor(np.hstack([np.zeros((len(X), 1)), X])).float()
        else:
            tX = torch.Tensor(X).float()
        return self.index(tX).detach().numpy()

    def get_index(self):
        idx = self.index.weight.detach().numpy().flatten()
        return idx[1:] if self.joint_training else idx


def SI_ATE_binY(
    T,
    W,
    Y,
    clf="MLP",
    hidden_dim=100,
    index_dim=1,
    lr=1e-3,
    n_iter=3000,
    l2pen=0,
    initial=None,
    trim=eps_trim,
    joint_training=True,
):
    clfY = IndexClassifier(
        hidden_dim=hidden_dim,
        index_dim=index_dim,
        lr=lr,
        n_iter=n_iter,
        l2pen=l2pen,
        joint_training=joint_training,
    )
    clfY.fit(stack_TW(T, W), Y, initial)
    Q0 = clfY.predict_conditional(0, W)
    Q1 = clfY.predict_conditional(1, W)
    Z = clfY.partial_predict(W)

    im_reg0 = Q0.mean()
    im_reg1 = Q1.mean()
    Var_reg = (Q1 - Q0).var()

    clfW = create_classifier(clf).fit(W, T)
    propW = np.clip(clfW.predict_proba(W), trim, 1 - trim)
    v_aipw0 = Q0 + (T == 0) * (Y - Q0) / propW[:, 0]
    v_aipw1 = Q1 + (T == 1) * (Y - Q1) / propW[:, 1]

    im_aipw0 = v_aipw0.mean()
    im_aipw1 = v_aipw1.mean()
    Var_aipw = (v_aipw1 - v_aipw0).var()

    clfZ = create_classifier(clf).fit(Z, T)
    propZ = np.clip(clfZ.predict_proba(Z), trim, 1 - trim)
    v_oapw0 = Q0 + (T == 0) * (Y - Q0) / propZ[:, 0]
    v_oapw1 = Q1 + (T == 1) * (Y - Q1) / propZ[:, 1]

    im_oapw0 = v_oapw0.mean()
    im_oapw1 = v_oapw1.mean()
    Var_oapw = (v_oapw1 - v_oapw0).var()

    Q = np.vstack([Q0, Q1]).T
    clfQ = create_classifier(clf).fit(Q, T)
    propQ = np.clip(clfQ.predict_proba(Q), trim, 1 - trim)
    v_obpw0 = Q0 + (T == 0) * (Y - Q0) / propQ[:, 0]
    v_obpw1 = Q1 + (T == 1) * (Y - Q1) / propQ[:, 1]

    im_obpw0 = v_obpw0.mean()
    im_obpw1 = v_obpw1.mean()
    Var_obpw = (v_obpw1 - v_obpw0).var()

    return (
        (im_reg0, im_aipw0, im_oapw0, im_obpw0),
        (im_reg1, im_aipw1, im_oapw1, im_obpw1),
        (Var_reg, Var_aipw, Var_oapw, Var_obpw),
    )


def ATE_est_binY(
    T,
    W,
    Y,
    classifier="Logistic",
    regressor="Logistic",
    fullQ=False,
    trim=eps_trim,
    joint_training=True,
):
    clf1 = create_classifier(classifier).fit(W, T)
    propW = np.clip(clf1.predict_proba(W), trim, 1 - trim)

    v_ipw0 = (T == 0) * Y / propW[:, 0]
    v_ipw1 = (T == 1) * Y / propW[:, 1]
    im_ipw0 = v_ipw0.mean()
    im_ipw1 = v_ipw1.mean()
    Var_ipw = (v_ipw1 - v_ipw0).var()

    if joint_training:
        regr = create_classifier(regressor).fit(stack_TW(T, W), Y)
        Q0 = regr.predict_proba(stack_TW(np.zeros_like(T), W))[:, 1]
        Q1 = regr.predict_proba(stack_TW(np.ones_like(T), W))[:, 1]
    else:
        regr0 = create_classifier(regressor).fit(W[T == 0], Y[T == 0])
        regr1 = create_classifier(regressor).fit(W[T == 1], Y[T == 1])
        Q0 = regr0.predict_proba(W)[:, 1]
        Q1 = regr1.predict_proba(W)[:, 1]

    Q = np.array([Q0, Q1]).T if fullQ else np.array([Q0]).T
    clfQ = create_classifier(classifier).fit(Q, T)
    propQ = np.clip(clfQ.predict_proba(Q), trim, 1 - trim)

    im_reg0 = Q0.mean()
    im_reg1 = Q1.mean()
    Var_reg = (Q1 - Q0).var()

    v_aipw0 = Q0 + (T == 0) * (Y - Q0) / propW[:, 0]
    v_aipw1 = Q1 + (T == 1) * (Y - Q1) / propW[:, 1]

    im_aipw0 = v_aipw0.mean()
    im_aipw1 = v_aipw1.mean()
    Var_aipw = (v_aipw1 - v_aipw0).var()

    v_obpw0 = Q0 + (T == 0) * (Y - Q0) / propQ[:, 0]
    v_obpw1 = Q1 + (T == 1) * (Y - Q1) / propQ[:, 1]

    im_obpw0 = v_obpw0.mean()
    im_obpw1 = v_obpw1.mean()
    Var_obpw = (v_obpw1 - v_obpw0).var()

    return (
        (im_ipw0, im_reg0, im_aipw0, im_obpw0),
        (im_ipw1, im_reg1, im_aipw1, im_obpw1),
        (Var_ipw, Var_reg, Var_aipw, Var_obpw),
    )
