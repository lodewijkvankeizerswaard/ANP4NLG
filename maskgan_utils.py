

def greedy_sample(logits):
    batch_size, seq_len, _ = logits.size()
    max_values, max_indices = logits.max(dim=2)
    return max_indices

def perplexity(truths, sampled, log_probs):
    batch_size, seq_len, vocab_size = log_probs.size()
    # sampled = greedy_sample(log_probs)
    _ppl = {
        'ground-truth': ppl(truths, log_probs).mean(),
        'sampled': ppl(sampled, log_probs).mean(),
    }
    return _ppl