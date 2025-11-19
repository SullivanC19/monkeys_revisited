from runners.simulate_pass_at_k_estimation import run as simulate_pass_at_k_estimation
from runners.simulate_synthetic_pass_at_k_estimation import run as simulate_synthetic_pass_at_k_estimation
from plotters.plot_estimate_mse import run as plot_estimate_mse
from plotters.plot_estimate_pass_at_k import run as plot_estimate_pass_at_k
from plotters.plot_heatmap import run as plot_heatmap
from plotters.plot_attempts_histogram import run as plot_attempts_histogram
from plotters.plot_synthetic_estimate_mse import run as plot_synthetic_estimate_mse
from plotters.plot_synthetic_beta_sampling import run as plot_synthetic_beta_sampling
from plotters.plot_hardness_distributions import run as plot_hardness_distributions

if __name__ == "__main__":
    print("Running experiments! 🐳")

    simulate_pass_at_k_estimation()
    simulate_synthetic_pass_at_k_estimation()
    plot_attempts_histogram()
    plot_estimate_pass_at_k()
    plot_heatmap()
    plot_estimate_mse()
    plot_synthetic_estimate_mse()
    plot_synthetic_beta_sampling()
    plot_hardness_distributions()

    print("Done ✅")

