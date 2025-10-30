"""
Probe comparison data extracted from analysis results.

This file contains MSE and correlation data for different probe configurations
comparing various VLM hidden layers for action step 0.

Data extracted from probe comparison plots for:
- Correct task: Action predictions from correct task VLM layers
- Wrong task: Action predictions from wrong task VLM layers
"""

# Correct task data - comparing different VLM hidden layers on correct task
correct_task_data = {
    "mean_pooled_layer_1": {
        "mse": 0.0557,
        "mean_correlation": 0.882,
    },
    "mean_pooled_layer_3": {
        "mse": 0.0537,
        "mean_correlation": 0.886,
    },
    "mean_pooled_layer_6": {
        "mse": 0.0540,
        "mean_correlation": 0.887,
    },
    "mean_pooled_layer_9": {
        "mse": 0.054,
        "mean_correlation": 0.889,
    },
    "mean_pooled_layer_12": {
        "mse": 0.0570,
        "mean_correlation": 0.882,
    },
    "last_vector_layer_1": {
        "mse": 0.1140,
        "mean_correlation": 0.752,
    },
    "last_vector_layer_3": {
        "mse": 0.0810,
        "mean_correlation": 0.820,
    },
    "last_vector_layer_6": {
        "mse": 0.0830,
        "mean_correlation": 0.819,
    },
    "last_vector_layer_9": {
        "mse": 0.0778,
        "mean_correlation": 0.830,
    },
    "last_vector_layer_12": {
        "mse": 0.0781,
        "mean_correlation": 0.840,
    },
}

# Wrong task data - comparing different VLM hidden layers on wrong task
wrong_task_data = {
    "mean_pooled_layer_1": {
        "mse": 0.0552,
        "mean_correlation": 0.882,
    },
    "mean_pooled_layer_3": {
        "mse": 0.0536,
        "mean_correlation": 0.887,
    },
    "mean_pooled_layer_6": {
        "mse": 0.0539,
        "mean_correlation": 0.887,
    },
    "mean_pooled_layer_9": {
        "mse": 0.0527,
        "mean_correlation": 0.889,
    },
    "mean_pooled_layer_12": {
        "mse": 0.0569,
        "mean_correlation": 0.882,
    },
    "last_vector_layer_1": {
        "mse": 0.1046,
        "mean_correlation": 0.767,
    },
    "last_vector_layer_3": {
        "mse": 0.0773,
        "mean_correlation": 0.831,
    },
    "last_vector_layer_6": {
        "mse": 0.0764,
        "mean_correlation": 0.833,
    },
    "last_vector_layer_9": {
        "mse": 0.0819,
        "mean_correlation": 0.824,
    },
    "last_vector_layer_12": {
        "mse": 0.0722,
        "mean_correlation": 0.845,
    },
}
