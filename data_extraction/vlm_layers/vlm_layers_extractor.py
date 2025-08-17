import torch
import pickle
from data_extraction.utils.vlm_features import apply_mean_pooling_and_last_vector


# Define extraction functions in user's specified format
def extract_single_step_data(policy, step_data, dataset_info):
    """
    Extract VLM and diffusion outputs in the user's specified format.

    Returns:
        data_dict: Dictionary with dataset, step_data, vlm_output, final_output
    """
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
            "dataset": dataset_info,  # Dataset name and info
            "step_data": step_data,  # Original input data
            **{f"mean_pooled_layer_{i}": vlm_output[i]["mean_pooled"] for i in range(len(vlm_output))},
            **{f"last_vector_layer_{i}": vlm_output[i]["last_vector"] for i in range(len(vlm_output))},
        }

        return data_dict


def save_all_extraction_data(all_data_list, output_file, MODEL_PATH, EMBODIMENT_TAG):
    """Save all extracted data to a single file in user's format."""
    # Extract only backbone_features from vlm_output
    backbone_features_list = []
    for data in all_data_list:
        if "backbone_features" in data["vlm_output"]:
            backbone_features_list.append(data["vlm_output"]["backbone_features"])
        else:
            backbone_features_list.append(None)  # Handle missing data

    # Combine all data
    combined_data = {
        "dataset": [data["dataset"] for data in all_data_list],
        "step_data": [data["step_data"] for data in all_data_list],
        "backbone_features": backbone_features_list,  # Only backbone_features from vlm_output
        "extraction_info": {
            "total_samples": len(all_data_list),
            "model_path": MODEL_PATH,
            "embodiment_tag": EMBODIMENT_TAG,
        },
    }

    # Move tensors to CPU for saving
    # Convert backbone_features to CPU if they are tensors
    for i in range(len(combined_data["backbone_features"])):
        if combined_data["backbone_features"][i] is not None and torch.is_tensor(combined_data["backbone_features"][i]):
            combined_data["backbone_features"][i] = combined_data["backbone_features"][i].cpu()

    # Save to file
    with open(output_file, "wb") as f:
        pickle.dump(combined_data, f)

    print(f"💾 Saved all data to {output_file}")
    print(f"   - Total samples: {len(all_data_list)}")
    print(f"   - Data keys: {list(combined_data.keys())}")
    print(f"   - Saved only backbone_features from vlm_output")

    return len(all_data_list)


print("✅ Extraction functions defined!")
