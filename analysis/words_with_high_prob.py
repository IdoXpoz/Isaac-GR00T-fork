import torch

from .utils import (
    get_tokenizer_for_policy,
    get_language_model_head,
    project_hidden_states_to_vocab,
    logits_to_probabilities,
)


def get_high_confidence_tokens(vocab_probs, tokenizer, min_probability=0.1):
    """Get tokens with probability above threshold for each position

    Args:
        vocab_probs: Probability distribution tensor [seq_len, vocab_size]
        tokenizer: Tokenizer for decoding
        min_probability: Minimum probability threshold (default: 0.1)

    Returns:
        List of dicts with position, word, and probability for high-confidence tokens
    """
    seq_len, vocab_size = vocab_probs.shape
    high_confidence_results = []

    for pos in range(seq_len):
        # Get all probabilities above threshold
        high_prob_mask = vocab_probs[pos] > min_probability
        high_prob_indices = torch.where(high_prob_mask)[0]
        high_prob_values = vocab_probs[pos][high_prob_indices]

        # Convert to words
        position_tokens = []
        for i, token_idx in enumerate(high_prob_indices):
            try:
                word = tokenizer.decode([token_idx.item()]).strip()
                # Filter out empty or special tokens
                if word and len(word) > 0 and word not in ["<", ">", "|", " "]:
                    position_tokens.append({"word": word, "probability": high_prob_values[i].item()})
            except:
                continue

        # Sort by probability (highest first)
        position_tokens.sort(key=lambda x: x["probability"], reverse=True)

        if position_tokens:
            high_confidence_results.append({"position": pos, "tokens": position_tokens})

    return high_confidence_results


def get_high_confidence_tokens_from_hidden_states(hidden_states, policy, min_probability=0.1):
    """Get high-confidence tokens from hidden states - complete pipeline with setup"""
    # Get components
    tokenizer = get_tokenizer_for_policy(policy)
    lm_head = get_language_model_head(policy)

    if lm_head is None:
        return None

    # Project to vocabulary space
    logits = project_hidden_states_to_vocab(hidden_states, lm_head)
    vocab_probs = logits_to_probabilities(logits)

    # Get high-confidence results
    return get_high_confidence_tokens(vocab_probs, tokenizer, min_probability)


def display_high_confidence_tokens(high_confidence_results, min_probability=0.1):
    """Display high-confidence token predictions in a readable format"""
    print("\n" + "=" * 70)
    print(f"HIGH-CONFIDENCE TOKENS (probability > {min_probability:.1f}):")
    print("=" * 70)

    total_high_conf_tokens = 0

    for result in high_confidence_results:
        pos = result["position"]
        tokens = result["tokens"]
        total_high_conf_tokens += len(tokens)

        print(f"\nPosition {pos:2d}:")
        for token_info in tokens:
            word = token_info["word"]
            prob = token_info["probability"]
            print(f"  • {word:<15} ({prob:.3f})")

    print("\n" + "=" * 70)
    print(f"SUMMARY:")
    print(f"- Total positions with high-confidence tokens: {len(high_confidence_results)}")
    print(f"- Total high-confidence tokens: {total_high_conf_tokens}")
    print(f"- Average tokens per position: {total_high_conf_tokens/len(high_confidence_results):.1f}")
    print("=" * 70)


def get_and_display_high_confidence_tokens(policy, observations, min_probability=0.1):
    """Complete pipeline: get high-confidence tokens from hidden states and display them"""
    hidden_states = policy.get_VLM_selected_layer_output(observations)["backbone_features"]
    high_conf_results = get_high_confidence_tokens_from_hidden_states(hidden_states, policy, min_probability)
    display_high_confidence_tokens(high_conf_results, min_probability)
