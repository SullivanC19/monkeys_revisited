from constants import DIR_SYN_RESULTS, DIR_PLOTS
from utilities import load_all_parquets

from .plot_synthetic_estimate_mse import bootstrap_mse, plot_mse_single_k

GROUP_COLS = ["Distribution", "Alpha", "Beta", "Scale", "Subproblems", "k"]


def run():
    input_dir = DIR_SYN_RESULTS / "schaeffer"
    print(f"Loading results from: {input_dir.resolve()}")
    df = load_all_parquets(input_dir)

    print("Bootstrapping MSE and plotting each synthetic setting...")
    for setting, group in df.groupby(GROUP_COLS):
        distribution, alpha, beta, scale, n_subtasks, k = setting
        stats = bootstrap_mse(group, B=10000)
        title = (
            f"{distribution.title()}: alpha={alpha}, beta={beta}, scale={scale}\n"
            f"m={n_subtasks}, k={k}"
        )
        filename = (
            f"schaeffer/estimate_mse-{distribution}-a={alpha}-b={beta}"
            f"-scale={scale}-m={n_subtasks}-k={k}"
        )
        plot_mse_single_k(stats, DIR_PLOTS, title=title, filename=filename)

    print(f"Saved figures -> {(DIR_PLOTS / 'syn' / 'schaeffer').resolve()}")


if __name__ == "__main__":
    run()
