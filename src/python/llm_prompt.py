def build_llm_prompt(diagnostics):
    """
    Builds a structured prompt for LLM-based regularization selection.
    """

    prompt = f"""
You are an expert in numerical linear algebra and inverse problems.

You are given diagnostics of an ill-posed linear inverse problem Ax = y.

Diagnostics:
- Condition number: {diagnostics['condition_number']}
- Spectral decay trend: {diagnostics['singular_decay_trend']}
- Spectral decay ratio (sigma_max / sigma_min): {diagnostics['spectral_decay_ratio']}

Tikhonov regularization:
- Candidate lambdas tested
- Best lambda based on reconstruction error: {diagnostics['tikhonov']['best_lambda']}

TSVD regularization:
- Candidate truncation ranks tested
- Best k based on reconstruction error: {diagnostics['tsvd']['best_k']}

Task:
1. Decide which regularization method is more appropriate (Tikhonov or TSVD).
2. Select the corresponding parameter (lambda or k).
3. Justify your choice in terms of spectral stability and noise amplification.

Respond strictly in the following JSON format:

{{
  "method": "tikhonov or tsvd",
  "parameter": numeric_value,
  "justification": "short explanation"
}}
"""
    return prompt
