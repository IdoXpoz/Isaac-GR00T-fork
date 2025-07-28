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
