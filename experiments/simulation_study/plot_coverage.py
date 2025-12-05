import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import norm

# plot settings
plt.rcParams["text.usetex"] = True
plt.rcParams.update({"font.size": 13})
plt.rc(
    "text.latex", preamble=r"\usepackage{amsfonts,amssymb,amsthm,amsmath,times}"
)
plt.rc("font", **{"family": "serif", "serif": ["Computer Modern"], "size": 13})
plt.rc("axes", **{"labelsize": 14, "titlesize": 15})


# Load and melt data
df = pd.read_pickle(
    "experiments/simulation_study/results/coverage_results_28-11-25.pkl"
)
id_vars = ["rep", "n", "d", "link", "clf", "joint", "crossfit", "IM"]
all_estimators = [
    "Unadjusted",
    "Regr",
    "IPW",
    "AIPW",
    "OBPW",
    "SI_R",
    "SI_AIPW",
    "SI_OAPW",
    "SI_OBPW",
]
df_est = df.drop(
    columns=["var_" + e for e in all_estimators]
    + ["bs_" + e for e in all_estimators]
)
df_est_melt = df_est.melt(
    id_vars=id_vars, value_name="estimate", var_name="estimator"
)

df_var = df.drop(columns=all_estimators + ["bs_" + e for e in all_estimators])
df_var_melt = df_var.melt(
    id_vars=id_vars, value_name="variance", var_name="estimator"
)
df_var_melt["estimator"] = df_var_melt["estimator"].apply(lambda x: x[4:])

df_bs = df.drop(columns=all_estimators + ["var_" + e for e in all_estimators])
df_bs_melt = df_bs.melt(
    id_vars=id_vars, value_name="BS se", var_name="estimator"
)
df_bs_melt["estimator"] = df_bs_melt["estimator"].apply(lambda x: x[3:])

df = pd.merge(df_est_melt, df_var_melt, on=id_vars + ["estimator"])
df = pd.merge(df, df_bs_melt, on=id_vars + ["estimator"])

# Melt asymptotic and bootstrap standard errors
df["asymp se"] = df["variance"] ** 0.5 / np.sqrt(df["n"])
df.drop(columns=["variance"], inplace=True)

asymp_df = df.drop(columns=["BS se"])
asymp_df = asymp_df.rename(columns={"asymp se": "se"})
asymp_df["se_type"] = "asymp"

bs_df = df.drop(columns=["asymp se"])
bs_df = bs_df.rename(columns={"BS se": "se"})
bs_df["se_type"] = "BS"

df = pd.concat([asymp_df, bs_df], axis=0)

df["lower"] = df["estimate"] - norm.ppf(0.975) * df["se"]
df["upper"] = df["estimate"] + norm.ppf(0.975) * df["se"]
df["cover"] = ((df["lower"] <= df["IM"]) & (df["IM"] <= df["upper"])).astype(
    int
)
df["errorsq"] = (df["estimate"] - df["IM"]) ** 2

# Change estimator names
df.insert(
    7,
    "regression",
    df["estimator"].apply(lambda x: "NN" if x[0:2] == "SI" else "Linear"),
)
df["estimator"] = df["estimator"].apply(
    lambda x: x[3:] if x[0:2] == "SI" else x
)
df["estimator"] = df["estimator"].replace(
    {
        "Unadjusted": "Unadjusted",
        "Regr": "Regression",
        "IPW": "IPW",
        "AIPW": "AIPW",
        "OBPW": "DOPE-BCL",
        "R": "Regression",
        "OAPW": "DOPE-IDX",
    }
)
df["link"] = df["link"].replace(
    {
        "lin": r"$\mathrm{linear}$",
        "cbrt": r"$\mathrm{cube \ root}$",
        "sqr": r"$\mathrm{square}$",
        "sin": r"$\mathrm{sin}$",
    }
)

df.drop(columns=["n", "d", "link", "clf", "joint"], inplace=True)
IM_true = df["IM"].mean()
df.drop(columns=["IM"], inplace=True)
estimators_order = ["Regression", "AIPW", "DOPE-BCL", "DOPE-IDX"]
df = df[
    (df["regression"] == "NN") & ~(df["estimator"].isin(["Unadjusted", "IPW"]))
]

df["se_type"] = df["se_type"].replace(
    {
        "asymp": r"$\mathrm{asymptotic}$",
        "BS": r"$\mathrm{bootstrap}$",
    }
)
df = df.rename(columns={"se_type": "standard error"})

g = sns.FacetGrid(
    df,
    row="crossfit",
    col="standard error",
    margin_titles=True,
    despine=True,
    hue="estimator",
    hue_order=estimators_order,
    aspect=1.6,
)

for (i, j, k), facet_df in g.facet_data():
    ax = g.axes[i, j]
    cov = facet_df["cover"].mean()
    length = 2 * facet_df["se"].median() * norm.ppf(0.975)
    ax.errorbar(
        facet_df["rep"] + k * 105,
        facet_df["estimate"],
        yerr=facet_df["se"] * norm.ppf(0.975),
        fmt="none",
        color="C{}".format(k),
    )
    ax.text(35 + k * 105, 2.95, r"{}\%".format(int(cov * 100)))
    ax.text(35 + k * 105, 0.9, r"{:.2f}".format(length))
    ax.axhline(y=IM_true, color="grey", linestyle="--", alpha=0.5)
    ax.get_xaxis().set_visible(False)
g.set(ylim=(0.8, 3.1))

g.set_titles(
    row_template=r"{row_var}: {row_name}",
    col_template=r"{col_var}: {col_name}",
)
g.add_legend(
    handles=[
        mpl.patches.Patch(color="C{}".format(k), label=estimator)
        for k, estimator in enumerate(estimators_order)
    ],
    loc="upper center",
    ncol=4,
    bbox_to_anchor=(0.3, 0.025),
    title="",
)
g.tight_layout()
g.fig.set_size_inches(12, 5)
g.fig.subplots_adjust(hspace=0.15, wspace=0.15)
g.savefig("plots/confidence_intervals.pdf")
