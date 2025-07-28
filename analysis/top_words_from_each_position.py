import torch

from .utils import (
    get_tokenizer_for_policy,
    get_language_model_head,
    project_hidden_states_to_vocab,
    logits_to_probabilities,
    decode_tokens_to_words,
)


def get_top_k_tokens_per_position(vocab_probs, k):
    """Get top-k token indices and probabilities for each position"""
    seq_len = vocab_probs.shape[0]
    position_results = []

    for pos in range(seq_len):
        top_values, top_indices = torch.topk(vocab_probs[pos], k)
        position_results.append(
            {
                "position": pos,
                "token_indices": top_indices.cpu().numpy(),
                "probabilities": top_values.float().cpu().numpy(),
            }
        )

    return position_results


def format_vocab_projection_results(position_results, tokenizer, top_k=3):
    """Format vocabulary projection results into final structure"""
    final_results = []

    for pos_result in position_results:
        words = decode_tokens_to_words(pos_result["token_indices"], tokenizer)

        if len(words) > 0:
            final_results.append(
                {
                    "position": pos_result["position"],
                    "words": words[:top_k],
                    "probabilities": pos_result["probabilities"][: len(words[:top_k])],
                }
            )

    return final_results


def get_top_words_from_each_position_of_hidden_states(hidden_states, policy, top_k=3):
    """Simple vocabulary projection - project hidden states back to vocabulary space"""
    # Get components
    tokenizer = get_tokenizer_for_policy(policy)
    lm_head = get_language_model_head(policy)

    if lm_head is None:
        return None

    # Project to vocabulary space
    logits = project_hidden_states_to_vocab(hidden_states, lm_head)
    vocab_probs = logits_to_probabilities(logits)

    # Get top-k results
    position_results = get_top_k_tokens_per_position(vocab_probs, top_k)
    final_results = format_vocab_projection_results(position_results, tokenizer, top_k)

    return final_results


def display_top_words_from_each_position(vocab_results):
    print("\n" + "=" * 60)
    print("TOP WORDS AT EACH POSITION (what the VLM is 'thinking'):")
    print("=" * 60)

    for result in vocab_results:
        pos = result["position"]
        words = result["words"]
        probs = result["probabilities"]

        word_prob_pairs = [f"{word}({prob:.3f})" for word, prob in zip(words, probs)]
        print(f"Position {pos:2d}: {', '.join(word_prob_pairs)}")

    print("\n" + "=" * 60)
    print("INTERPRETATION GUIDE:")
    print("- Early positions (0-10): Often represent visual concepts from the image")
    print("- Later positions (10+): Often represent language concepts from instruction")
    print("- High probabilities (>0.1): Strong semantic activations")
    print("- Related words together: Semantic clusters")
    print("=" * 60)


def get_and_display_top_words_from_each_position_of_hidden_states(policy, observations, top_k=3):
    """Complete vocabulary projection in one simple function"""
    hidden_states = policy.get_VLM_selected_layer_output(observations)["backbone_features"]
    top_words_from_each_position = get_top_words_from_each_position_of_hidden_states(hidden_states, policy, top_k)
    display_top_words_from_each_position(top_words_from_each_position)
