# Efficient Prediction of Pass@k Scaling in Large Language Models

![combined_statistical_plots.png](experiments/figs/pass_at_k/problem=jailbreaking-budget=10000.png)

## Setup & Execution

1. Build the docker image

`docker build -f experiments/Dockerfile -t experiments .`

2. Run all experiments and generate all plots (2-4 hours).

`docker run --rm -it -v "$PWD:/app" experiments`