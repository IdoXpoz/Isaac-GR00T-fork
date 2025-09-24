import torch
from data_extraction.utils.vlm_features import apply_mean_pooling_and_last_vector


def extract_single_step_data_raise_hands(policy, step_data, dataset_info):
    """
    Extract VLM and diffusion outputs in the user's specified format.

    Returns:
        data_dict: Dictionary with dataset, step_data, vlm_output, final_output
    """
    step_data["annotation.human.coarse_action"] = ["unlocked_waist: raise both hands up"]

    selected_layers = [1, 3, 6, 9, 12]
    with torch.no_grad():
        # Extract VLM backbone features (without action head)
        vlm_output = policy.get_VLM_selected_layers_output(step_data, selected_layers)

        # take every returned layer, get the mean pool and last vector
        for i in range(len(vlm_output)):
            vlm_output[i]["mean_pooled"], vlm_output[i]["last_vector"] = apply_mean_pooling_and_last_vector(
                vlm_output[i]["backbone_features"]
            )

        # Extract action head output
        # action = policy.get_action(step_data)
        # action_right_arm = action["action.right_arm"]
        # turn action_right_arm to 1d array
        # action_right_arm = action_right_arm.reshape(-1)

        # Create the data in user's specified format
        data_dict = {
            "sample_index": dataset_info["sample_index"],
            "global_index": dataset_info["global_index"],
            **{f"mean_pooled_layer_{selected_layers[i]}": vlm_output[i]["mean_pooled"] for i in range(len(vlm_output))},
            **{f"last_vector_layer_{selected_layers[i]}": vlm_output[i]["last_vector"] for i in range(len(vlm_output))},
            # "action_right_arm": action_right_arm,
        }

        print(f"data_dict: {data_dict}")

        return data_dict


def extract_single_step_full_inference_using_selected_vlm_layers(policy, step_data, dataset_info):
    """
    Run full inference using different VLM layers and extract right_arm actions.

    For each selected layer, runs the full policy inference pipeline using that layer's
    output for action prediction, then extracts and saves the right_arm action outputs.

    Args:
        policy: The GR00T policy instance
        step_data: Dictionary containing step observation data
        dataset_info: Dictionary containing sample metadata

    Returns:
        data_dict: Dictionary with dataset info and action outputs from each layer
    """

    selected_layers = [1, 3, 6, 9, 12]

    with torch.no_grad():
        # Create base data dictionary with metadata
        data_dict = {
            "sample_index": dataset_info["sample_index"],
            "global_index": dataset_info["global_index"],
        }

        # Run inference for each selected layer
        for layer in selected_layers:
            # Run full inference using the selected VLM layer
            action = policy.get_action_using_selected_vlm_layer(step_data, layer)

            # Extract right_arm action and convert to 1d array
            action_right_arm = action["action.right_arm"]
            action_right_arm = action_right_arm.reshape(-1)

            # Save with descriptive name
            data_dict[f"action_right_arm_layer_{layer}"] = action_right_arm

        print(f"data_dict keys: {list(data_dict.keys())}")
        print(f"action shapes: {[(k, v.shape) for k, v in data_dict.items() if k.startswith('action_')]}")

        return data_dict
