from runners.simulate_pass_at_k_estimation import run as simulate_pass_at_k_estimation
from plotters.plot_estimate_mse import run as plot_estimate_mse
from plotters.plot_heatmap import run as plot_heatmap

if __name__ == "__main__":
    print("Running experiments! 🐳")

    # simulate_pass_at_k_estimation()
    plot_heatmap()
    plot_estimate_mse()

    print("Done ✅")

