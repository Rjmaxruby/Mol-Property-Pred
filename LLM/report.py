from LLM.prompts import build_toxicology_prompt
from LLM.client import generate

def call_groq_narrative(
    
    smiles,
    common_name,
    attention_findings,
):
    print("DEBUG findings_for_groq:", attention_findings[0]["matched_groups"][:2])  # add this
    
    prompt = build_toxicology_prompt(
        smiles,
        common_name,
        attention_findings,
    )

    return generate(prompt)