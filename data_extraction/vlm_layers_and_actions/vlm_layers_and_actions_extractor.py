import torch
from data_extraction.utils.vlm_features import apply_mean_pooling_and_last_vector


# Define extraction functions in user's specified format
def extract_single_step_data(policy, step_data, dataset_info):
    """
    Extract VLM and diffusion outputs in the user's specified format.

    Returns:
        data_dict: Dictionary with dataset, step_data, vlm_output, final_output
    """

    # Extract action head output
    action = policy.get_action(step_data)
    action_right_arm = action["action.right_arm"]
    # turn action_right_arm to 1d array
    action_right_arm = action_right_arm.reshape(-1)

    # Change action and extract VLM features
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
