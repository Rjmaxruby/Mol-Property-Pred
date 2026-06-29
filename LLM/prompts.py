def build_toxicology_prompt(
    smiles: str,
    common_name: str,
    attention_findings: list,
) -> str:
    """
    Builds the prompt sent to the LLM.
    """

    findings_text = ""

    for f in attention_findings:
        print("DEBUG group keys:", f["matched_groups"][0].keys() if f["matched_groups"] else "empty")  # add this
        groups_str = ", ".join(
            f"{g['name']} (saliency: {g['saliency_score']:.3f})"
            for g in f["matched_groups"][:4]
        ) or "no named groups matched (novel/composite fragment)"

        findings_text += (
            f"\n- {f['target']} ({f['description']})\n"
            f"  Probability: {f['probability']:.2f} "
            f"({'TOXIC' if f['toxic'] else 'borderline'}, confidence: {f['confidence']})\n"
            f"  GAT-attended structural features: {groups_str}\n"
        )

    molecule_label = common_name if common_name else smiles

    prompt = f"""
You are a senior medicinal chemist and computational toxicologist.

You are assisting a drug discovery team.

You have been given Graph Attention Network (GAT) attention analysis for:

Compound: {molecule_label}
SMILES: {smiles}

Model performance:
Mean ROC-AUC across Tox21 endpoints = 0.8284

Flagged targets:

{findings_text}

Write a professional medicinal chemistry report using exactly these sections:

1. EXECUTIVE SUMMARY

Summarize the overall toxicity profile in 2-3 sentences.

2. STRUCTURAL ANALYSIS

For every flagged endpoint:

• identify the structural feature receiving highest GAT attention

• explain why medicinal chemists associate that functional group with this endpoint

• explain likely biological mechanism

Do NOT claim certainty.

State that GAT attention represents model focus.

3. SAR NOTES

Discuss recurring structural motifs across endpoints.

4. MEDICINAL CHEMISTRY RECOMMENDATIONS

Suggest realistic medicinal chemistry modifications including:

• bioisosteres

• scaffold replacements

• substituent optimization

Explain why each modification could reduce toxicity.

5. CAVEATS

Mention:

• attention ≠ causality

• experimental validation required

Maximum length: 500 words.

Write professionally.
"""

    return prompt