import numpy as np


def apply_mean_pooling_and_last_vector(backbone_features):
    """
    Apply mean pooling and extract last vector from backbone features.

    Args:
        backbone_features: Flattened list/array of backbone features

    Returns:
        tuple: (mean_pooled_features, last_vector_features) both shape [hidden_size]
    """

    try:
        # Convert to numpy array
        features_array = np.array(backbone_features)

        # Handle different input shapes
        if len(features_array.shape) == 3:
            # Shape is [batch, seq_len, hidden_size]
            if features_array.shape[0] == 1:
                # Remove batch dimension -> [seq_len, hidden_size]
                features_array = features_array[0]
            else:
                # Multiple batch items, take the first one
                features_array = features_array[0]

        elif len(features_array.shape) == 2:
            # Already in [seq_len, hidden_size] format
            pass
        else:
            print(f"⚠️  Unexpected shape after reshape: {features_array.shape}")
            return None, None

        # Ensure we have a 2D tensor [seq_len, hidden_size]
        if len(features_array.shape) != 2:
            print(f"⚠️  Expected 2D features after processing, got {features_array.shape}")
            return None, None

        seq_len, hidden_size = features_array.shape

        if seq_len == 0 or hidden_size == 0:
            print(f"⚠️  Invalid dimensions: seq_len={seq_len}, hidden_size={hidden_size}")
            return None, None

        # Apply mean pooling across sequence dimension (axis 0)
        mean_pooled = np.mean(features_array, axis=0)  # [seq_len, hidden_size] -> [hidden_size]

        # Extract last vector (final sequence position)
        last_vector = features_array[-1]  # [seq_len, hidden_size] -> [hidden_size]

        # Validate output shapes
        if mean_pooled.shape != (hidden_size,) or last_vector.shape != (hidden_size,):
            print(
                f"⚠️  Output shape mismatch: mean_pooled={mean_pooled.shape}, last_vector={last_vector.shape}, expected=({hidden_size},)"
            )
            return None, None

        return mean_pooled, last_vector

    except Exception as e:
        print(f"⚠️  Error processing features: {e}")
        return None, None
