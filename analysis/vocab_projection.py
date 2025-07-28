import torch
from transformers import AutoTokenizer

from gr00t.model.backbone.eagle_backbone import DEFAULT_EAGLE_PATH


def get_tokenizer_for_policy(policy):
    """Get the appropriate tokenizer for the policy's VLM model"""
    try:
        tokenizer = AutoTokenizer.from_pretrained(DEFAULT_EAGLE_PATH, trust_remote_code=True)
        print("✅ Using Eagle VLM tokenizer (Qwen3-based)")
        return tokenizer
    except Exception as e:
        print(f"Could not load Eagle tokenizer ({e}), trying Qwen3...")
        try:
            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
            print("✅ Using Qwen3 tokenizer")
            return tokenizer
        except:
            print("⚠️  Using GPT2 tokenizer as fallback")
            return AutoTokenizer.from_pretrained("gpt2")


def get_language_model_head(policy):
    """Find and return the language model head for vocabulary projection"""
    vlm_model = policy.model.backbone.eagle_model

    if hasattr(vlm_model, "language_model") and hasattr(vlm_model.language_model, "lm_head"):
        lm_head = vlm_model.language_model.lm_head
        print("Found language model head in vlm_model.language_model.lm_head")
        return lm_head
    elif hasattr(vlm_model, "lm_head"):
        lm_head = vlm_model.lm_head
        print("Found language model head in vlm_model.lm_head")
        return lm_head
    else:
        print("Could not find language model head for vocab projection")
        print(f"Available attributes: {list(vlm_model.__dict__.keys())}")
        return None


def project_hidden_states_to_vocab(hidden_states, lm_head):
    """Project hidden states to vocabulary space and return logits"""
    with torch.no_grad():
        # Handle batch dimension
        if hidden_states.dim() == 3:  # [batch, seq_len, hidden_dim]
            hidden_states = hidden_states[0]  # Take first batch

        print(f"Projecting hidden states of shape {hidden_states.shape} to vocabulary space...")

        # Ensure hidden_states and lm_head are on the same device
        lm_head_device = next(lm_head.parameters()).device
        hidden_states = hidden_states.to(lm_head_device)

        print(f"Hidden states device: {hidden_states.device}")
        print(f"LM head device: {lm_head_device}")

        # Project to vocabulary space
        logits = lm_head(hidden_states)  # [seq_len, vocab_size]
        print(f"Vocabulary projection shape: {logits.shape}")

        return logits


def logits_to_probabilities(logits):
    """Convert logits to probabilities using softmax"""
    return torch.softmax(logits, dim=-1)


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


def decode_tokens_to_words(token_indices, tokenizer, filter_special=True):
    """Convert token indices to readable words"""
    words = []
    for idx in token_indices:
        try:
            word = tokenizer.decode([idx]).strip()
            if filter_special:
                # Filter out special tokens and empty strings
                if word and len(word) > 0 and word not in ["<", ">", "|", " "]:
                    words.append(word)
            else:
                words.append(word)
        except:
            if not filter_special:
                words.append(f"<token_{idx}>")  # Fallback representation
    return words


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


def get_hidden_states_and_run_vocab_projection(policy, observations, top_k=3):
    """Complete vocabulary projection in one simple function"""
    hidden_states = policy.get_backbone_features(observations)["backbone_features"]
    return get_top_words_from_each_position_of_hidden_states(hidden_states, policy, top_k)


def display_vocab_projection_results(vocab_results):
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
