import torch
from transformers import AutoTokenizer


def get_top_words_from_each_position_of_hidden_states(hidden_states, policy, top_k=3):
    """Simple vocabulary projection - project hidden states back to vocabulary space"""

    # Get the correct tokenizer (same as Eagle VLM uses)
    try:
        from gr00t.model.backbone.eagle_backbone import DEFAULT_EAGLE_PATH

        tokenizer = AutoTokenizer.from_pretrained(DEFAULT_EAGLE_PATH, trust_remote_code=True)
        print("✅ Using Eagle VLM tokenizer (Qwen3-based)")
    except Exception as e:
        print(f"Could not load Eagle tokenizer ({e}), trying Qwen3...")
        try:
            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
            print("✅ Using Qwen3 tokenizer")
        except:
            print("⚠️  Using GPT2 tokenizer as fallback")
            tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Access the VLM model
    vlm_model = policy.model.backbone.eagle_model

    # Try to find the language model head
    if hasattr(vlm_model, "language_model") and hasattr(vlm_model.language_model, "lm_head"):
        lm_head = vlm_model.language_model.lm_head
        print("Found language model head in vlm_model.language_model.lm_head")
    elif hasattr(vlm_model, "lm_head"):
        lm_head = vlm_model.lm_head
        print("Found language model head in vlm_model.lm_head")
    else:
        print("Could not find language model head for vocab projection")
        print(f"Available attributes: {list(vlm_model.__dict__.keys())}")
        return None

    with torch.no_grad():
        # Handle batch dimension
        if hidden_states.dim() == 3:  # [batch, seq_len, hidden_dim]
            hidden_states = hidden_states[0]  # Take first batch

        print(f"Projecting hidden states of shape {hidden_states.shape} to vocabulary space...")

        # Ensure hidden_states and lm_head are on the same device
        lm_head_device = next(lm_head.parameters()).device
        hidden_states = hidden_states.to(lm_head_device)

        print(f"Hidden states device after: {hidden_states.device}")
        print(f"LM head device after: {lm_head_device}")

        # Project to vocabulary space
        logits = lm_head(hidden_states)  # [seq_len, vocab_size]
        vocab_probs = torch.softmax(logits, dim=-1)

        print(f"Vocabulary projection shape: {vocab_probs.shape}")

        # Get top-k words for each position
        seq_len = vocab_probs.shape[0]
        results = []

        for pos in range(seq_len):
            top_values, top_indices = torch.topk(vocab_probs[pos], top_k)
            top_words = []

            for idx in top_indices:
                try:
                    word = tokenizer.decode([idx.item()]).strip()
                    if word and len(word) > 0 and word not in ["<", ">", "|", " "]:
                        top_words.append(word)
                except:
                    pass

            if len(top_words) > 0:
                results.append(
                    {
                        "position": pos,
                        "words": top_words[:top_k],
                        "probabilities": top_values.float().cpu().numpy()[: len(top_words)],
                    }
                )

        return results


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
