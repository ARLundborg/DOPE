import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2, norm

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
    "experiments/simulation_study/results/simulation_study_results_26-11-25.pkl"
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
df_est = df.drop(columns=["var_" + e for e in all_estimators])
df_est_melt = df_est.melt(
    id_vars=id_vars, value_name="estimate", var_name="estimator"
)
df_var = df.drop(columns=all_estimators)
df_var_melt = df_var.melt(
    id_vars=id_vars, value_name="variance", var_name="estimator"
)
df_var_melt["estimator"] = df_var_melt["estimator"].apply(lambda x: x[4:])
df = pd.merge(df_est_melt, df_var_melt, on=id_vars + ["estimator"])

# Add columns for coverage, width, and p-value
df["error"] = df["estimate"] - df["IM"]
df["error (scaled)"] = np.sqrt(df["n"]) * np.abs(df["error"])
df["errorsq"] = df["error"] ** 2
df["sd"] = np.sqrt(df["variance"])
alpha = 0.05
df["cover"] = norm.ppf(1 - 0.5 * alpha) * df["sd"] > np.sqrt(df["n"]) * np.abs(
    df["error"]
)  # nominal coverage is 1-alpha
df["width"] = 2 * norm.ppf(1 - 0.5 * alpha) * df["sd"] / np.sqrt(df["n"])
df["zvalue"] = np.sqrt(df["n"]) * df["error"] / df["sd"]
df["pvalue"] = 1 - chi2.cdf(df["zvalue"] ** 2, df=1)
df["log(n)"] = df["n"].apply(lambda x: np.emath.logn(3, x // 300))
df["scaled RMSE"] = np.sqrt(df["n"]) * np.sqrt(df["errorsq"])

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
df = df[df["d"] == 12].copy().drop(columns=["d"])

estimators_order = ["Regression", "AIPW", "DOPE-BCL", "DOPE-IDX"]
links_order = [
    r"$\mathrm{linear}$",
    r"$\mathrm{cube \ root}$",
    r"$\mathrm{square}$",
    r"$\mathrm{sin}$",
]


def line_plot_rmse(
    df,
    joint,
    links,
    omitted_estimators=("Unadjusted", "IPW"),
    ymaxs=(None, None),
):
    plt_df = df[
        (df["joint"] == joint)
        & (df["link"].isin(links))
        & (~df["estimator"].isin(omitted_estimators))
        & (df["crossfit"] == False)
    ].copy()
    plt_df["regression"] = plt_df["regression"].replace(
        {"Linear": r"$\mathrm{linear}$", "NN": r"$\mathrm{neural\ network}$"}
    )
    g = sns.relplot(
        data=plt_df,
        x="log(n)",
        y="scaled RMSE",
        hue="estimator",
        row="link",
        col="regression",
        kind="line",
        facet_kws={"sharey": "row", "sharex": True, "margin_titles": True},
        hue_order=estimators_order,
        row_order=[link for link in links_order if link in links],
        err_style="bars",
        errorbar=("se", norm.ppf(0.975)),
        linestyle="dashed",
        marker="o",
        legend="brief",
        alpha=0.75,
        linewidth=2,
        markersize=6,
        aspect=1,
    )
    g.set_xlabels(r"sample size $n$")
    g.fig.text(
        s=r"$\sqrt{n} \times \mathrm{RMSE}$", x=0.0125, y=0.425, rotation=90
    )
    g.set_ylabels("")
    g.set(xticks=np.arange(3))
    g.set_xticklabels(labels=[300 * (3**l) for l in range(3)])
    sns.move_legend(
        g,
        "upper center",
        ncol=4,
        bbox_to_anchor=(0.45, -0.025),
        title="",
        handles=[
            mpl.patches.Patch(color="C{}".format(k), label=estimator)
            for k, estimator in enumerate(estimators_order)
        ],
    )
    g.set_titles(
        row_template=r"{row_var}: {row_name}",
        col_template=r"{col_var}: {col_name}",
    )
    g.axes[0][0].set_ylim((0, ymaxs[0]))
    g.axes[0][1].set_ylim((0, ymaxs[0]))
    g.axes[1][0].set_ylim((0, ymaxs[1]))
    g.axes[1][1].set_ylim((0, ymaxs[1]))
    g.fig.set_size_inches(8, 5)
    g.fig.subplots_adjust(hspace=0.15, wspace=0.15)
    return g


g = line_plot_rmse(
    df,
    joint=False,
    links=[r"$\mathrm{linear}$", r"$\mathrm{cube \ root}$"],
    ymaxs=(5.5, 8),
)
g.savefig("plots/rmse_stratified_linear_cbrt.pdf")
g = line_plot_rmse(
    df,
    joint=True,
    links=[r"$\mathrm{linear}$", r"$\mathrm{cube \ root}$"],
    ymaxs=(5.5, 8),
)
g.savefig("plots/rmse_joint_linear_cbrt.pdf")
g = line_plot_rmse(
    df,
    joint=False,
    links=[r"$\mathrm{square}$", r"$\mathrm{sin}$"],
    ymaxs=(6, 20),
)
g.savefig("plots/rmse_stratified_square_sin.pdf")
g = line_plot_rmse(
    df,
    joint=True,
    links=[r"$\mathrm{square}$", r"$\mathrm{sin}$"],
    ymaxs=(6, 15),
)
g.savefig("plots/rmse_joint_square_sin.pdf")


def line_plot_crossfit(
    df,
    joint,
    links,
    omitted_estimators=("Unadjusted", "IPW"),
):
    tmp_df = df[
        (df["joint"] == joint)
        & (df["link"].isin(links))
        & (~df["estimator"].isin(omitted_estimators))
    ].copy()
    estimators = tmp_df["estimator"].unique()
    tmp_df["regression"] = tmp_df["regression"].replace(
        {"Linear": r"$\mathrm{linear}$", "NN": r"$\mathrm{neural\ network}$"}
    )
    tmp_df_grp = tmp_df.groupby(
        ["log(n)", "link", "regression", "estimator", "crossfit"]
    )
    if len(tmp_df_grp.count()["scaled RMSE"].unique()) > 1:
        raise ValueError("Not all groups have the same number of repetitions")
    reps = tmp_df_grp.count()["scaled RMSE"].unique()[0]
    tmp_df = tmp_df_grp["scaled RMSE"].mean().reset_index().copy()
    tmp_df["variance"] = tmp_df_grp["scaled RMSE"].var().reset_index(drop=True)
    tmp_df["variance"] = tmp_df["variance"]

    tmp_df = tmp_df.pivot(
        columns=["crossfit"],
        index=["log(n)", "link", "regression", "estimator"],
        values=["scaled RMSE", "variance"],
    )

    plt_df = (
        (tmp_df[("scaled RMSE", True)] / tmp_df[("scaled RMSE", False)])
        .reset_index()
        .copy()
    )
    plt_df.columns = ["log(n)", "link", "regression", "estimator", "ratio"]
    plt_df["se"] = (
        tmp_df[("variance", True)] / tmp_df[("scaled RMSE", False)] ** 2
        + tmp_df[("variance", False)]
        * tmp_df[("scaled RMSE", True)] ** 2
        / tmp_df[("scaled RMSE", False)] ** 4
    ).reset_index(
        drop=True
    ) ** 0.5 / reps**0.5  # Asymptotic variance from delta method
    plt_df["CI"] = norm.ppf(0.975) * plt_df["se"]

    g = sns.relplot(
        data=plt_df,
        x="log(n)",
        y="ratio",
        hue="estimator",
        hue_order=estimators_order,
        row="link",
        row_order=[link for link in links_order if link in links],
        col="regression",
        kind="line",
        facet_kws={"sharey": "row", "sharex": True, "margin_titles": True},
        linestyle="dashed",
        marker="o",
        legend="brief",
        alpha=0.75,
        linewidth=2,
        markersize=6,
        aspect=1,
    )
    facet_data = list(g.facet_data())
    for i, ax in enumerate(g.axes.flatten()):
        facet_df = facet_data[i][1]
        ax.axhline(1, color="grey", linestyle="--", linewidth=1, alpha=0.5)
        for estimator in estimators:
            facet_df_est = facet_df[facet_df["estimator"] == estimator]
            ax.errorbar(
                x="log(n)",
                y="ratio",
                yerr="CI",
                data=facet_df_est,
                alpha=0.75,
                fmt="none",
                ecolor="C{}".format(estimators_order.index(estimator)),
            )
    g.set_xlabels(r"sample size $n$")
    g.fig.text(
        s=r"Ratio of RMSE with and without cross-fitting",
        x=0.0125,
        y=0.2,
        rotation=90,
    )
    g.set_ylabels("")
    g.set(xticks=np.arange(3))
    g.set_xticklabels(labels=[300 * (3**l) for l in range(3)])
    sns.move_legend(
        g,
        "upper center",
        ncol=4,
        bbox_to_anchor=(0.45, -0.025),
        title="",
        handles=[
            mpl.patches.Patch(color="C{}".format(k), label=estimator)
            for k, estimator in enumerate(estimators_order)
        ],
    )
    g.set_titles(
        row_template=r"{row_var}: {row_name}",
        col_template=r"{col_var}: {col_name}",
    )
    g.fig.set_size_inches(8, 5)
    g.fig.subplots_adjust(hspace=0.15, wspace=0.15)
    return g


g = line_plot_crossfit(
    df, joint=False, links=[r"$\mathrm{linear}$", r"$\mathrm{cube \ root}$"]
)
g.savefig("plots/crossfit_stratified_linear_cbrt.pdf")
g = line_plot_crossfit(
    df, joint=True, links=[r"$\mathrm{linear}$", r"$\mathrm{cube \ root}$"]
)
g.savefig("plots/crossfit_joint_linear_cbrt.pdf")
g = line_plot_crossfit(
    df, joint=False, links=[r"$\mathrm{square}$", r"$\mathrm{sin}$"]
)
g.savefig("plots/crossfit_stratified_square_sin.pdf")
g = line_plot_crossfit(
    df, joint=True, links=[r"$\mathrm{square}$", r"$\mathrm{sin}$"]
)
g.savefig("plots/crossfit_joint_square_sin.pdf")
