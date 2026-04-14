"""
Prompt v3 – Few-Shot with Chain-of-Thought
--------------------------------------------
The most refined prompt version combining:
  • Expert persona with domain-specific knowledge
  • 3 high-quality few-shot examples from real AFCA categories
  • Chain-of-thought reasoning structure
  • Explicit output format specification
  • Regulatory citation requirements

This is the production prompt – designed for maximum accuracy and consistency.
"""

from __future__ import annotations

from typing import Any


# ── Few-Shot Examples ────────────────────────────────────────────────
# These examples are carefully selected to cover the most common and
# impactful complaint categories, providing the LLM with calibration
# anchors for consistent classification.

FEW_SHOT_EXAMPLES = [
    {
        "complaint": (
            "I noticed three transactions on my Visa credit card totalling $4,280 "
            "that I did not authorise. The charges were made at electronics stores in "
            "Sydney while I was overseas. I contacted my bank immediately but they "
            "refused to reverse the charges, claiming I must have shared my PIN."
        ),
        "output": {
            "category": "Unauthorised Transaction",
            "sub_category": "Fraudulent charge",
            "policy_refs": [
                "RG 271.28 (30-day IDR response timeframe)",
                "ePayments Code s.11 (unauthorised transactions liability)",
                "Banking Code of Practice Ch.32 (complaint handling)",
            ],
            "reasoning": (
                "1. Classification: This is an unauthorised transaction complaint — "
                "the customer reports charges they did not make or authorise. The bank's "
                "assertion that the customer shared their PIN must be substantiated.\n"
                "2. Regulatory: Under the ePayments Code s.11, the bank bears the burden "
                "of proving customer negligence. RG 271.28 requires a response within 30 days.\n"
                "3. Risk: HIGH — fraudulent transaction with monetary loss of $4,280. "
                "Potential ePayments Code breach if the bank cannot prove negligence.\n"
                "4. The IDR response must acknowledge the complaint, confirm investigation "
                "of the transaction records, and provide the applicable timeframe."
            ),
            "draft_response": (
                "Dear Customer,\n\n"
                "Thank you for bringing this matter to our attention. We acknowledge your "
                "complaint regarding unauthorised transactions totalling $4,280 on your "
                "Visa credit card.\n\n"
                "We take allegations of unauthorised transactions very seriously. We have "
                "initiated an investigation into the three transactions you have identified "
                "and will review our transaction monitoring records, merchant details, and "
                "any available CCTV or location data.\n\n"
                "Under our Internal Dispute Resolution procedures, we will provide you with "
                "a formal response within 30 calendar days. If our investigation requires "
                "additional time, we will contact you to explain the reasons for any delay.\n\n"
                "If you are not satisfied with our response, you have the right to lodge a "
                "complaint with the Australian Financial Complaints Authority (AFCA) on "
                "1800 931 678 or at www.afca.org.au.\n\n"
                "Yours sincerely,\nComplaint Resolution Team"
            ),
        },
    },
    {
        "complaint": (
            "My home was damaged in the January floods. I lodged a claim for $45,000 "
            "in repairs but the insurer denied it saying the damage was caused by "
            "'rising water' not 'storm damage' and therefore excluded under my policy. "
            "I believe this interpretation is unreasonable as the water entered through "
            "the roof during the storm."
        ),
        "output": {
            "category": "Claim Denial",
            "sub_category": "Home and contents – flood/storm definition",
            "policy_refs": [
                "Insurance Contracts Act 1984 s.54 (insurer limitations on refusing claims)",
                "General Insurance Code of Practice s.7 (claims handling)",
                "RG 271.28 (IDR timeframe)",
            ],
            "reasoning": (
                "1. Classification: Insurance claim denial based on policy exclusion – "
                "the insurer is distinguishing between 'flood' and 'storm' damage.\n"
                "2. Regulatory: ICA s.54 limits the insurer's ability to refuse claims; "
                "the proximate cause of damage needs assessment. The GI Code s.7 requires "
                "fair and transparent claims handling.\n"
                "3. Risk: HIGH — $45,000 claim value and potential systemic issue if "
                "multiple policyholders are affected by the same flood event.\n"
                "4. Response must acknowledge the competing interpretations and commit "
                "to an independent assessment of the damage cause."
            ),
            "draft_response": (
                "Dear Customer,\n\n"
                "We acknowledge your complaint regarding the denial of your home damage "
                "claim for $45,000. We understand this is a distressing situation.\n\n"
                "We note your position that the damage was primarily caused by storm-driven "
                "rain entering through your roof, rather than rising floodwater. We will "
                "arrange for an independent assessment to determine the proximate cause of "
                "the damage, taking into account all available evidence including weather "
                "data and the point of water entry.\n\n"
                "We will provide you with our formal response within 30 calendar days. "
                "In the meantime, if you require urgent make-safe repairs, please contact "
                "us immediately.\n\n"
                "If you are not satisfied with our response, you may lodge a complaint "
                "with AFCA on 1800 931 678.\n\n"
                "Yours sincerely,\nClaims Resolution Team"
            ),
        },
    },
    {
        "complaint": (
            "I lost my job due to company restructuring and requested a 3-month "
            "repayment pause on my home loan. The bank refused and instead offered "
            "to extend my loan term by 5 years which would cost me an additional "
            "$85,000 in interest. I believe this is not a genuine hardship arrangement."
        ),
        "output": {
            "category": "Financial Hardship",
            "sub_category": "Repayment difficulty",
            "policy_refs": [
                "National Credit Code s.130 (hardship changes)",
                "NCC s.133 (enforcement and hardship)",
                "Banking Code of Practice Ch.40 (financial difficulty)",
                "RG 271.56(d) (21-day IDR deadline for hardship)",
            ],
            "reasoning": (
                "1. Classification: Financial hardship complaint — the customer is unable "
                "to meet repayments due to involuntary job loss and the bank's proposed "
                "arrangement is potentially inadequate.\n"
                "2. Regulatory: NCC s.130 requires the bank to respond to hardship notices "
                "and genuinely consider appropriate variations. The Banking Code Ch.40 "
                "obliges the bank to work with the customer to find a sustainable solution.\n"
                "3. Risk: HIGH — shortened 21-day IDR deadline applies (RG 271.56(d)), "
                "potential NCC breach if the arrangement is not genuine.\n"
                "4. Response must acknowledge the hardship, confirm the shortened timeline, "
                "and commit to exploring genuine alternatives."
            ),
            "draft_response": (
                "Dear Customer,\n\n"
                "We acknowledge your financial hardship notice regarding your home loan. "
                "We understand that losing your employment is a very difficult situation.\n\n"
                "We note your concern that the proposed loan term extension may not be a "
                "suitable hardship arrangement for your circumstances. Under the National "
                "Credit Code, we are obligated to genuinely consider your hardship request "
                "and work with you to find an appropriate solution.\n\n"
                "As this is a hardship complaint, we will provide you with our response "
                "within 21 calendar days, as required under ASIC Regulatory Guide 271. "
                "A member of our Financial Hardship team will contact you within 2 business "
                "days to discuss your options.\n\n"
                "Please note: no enforcement action will be taken on your account while "
                "your hardship request is being assessed.\n\n"
                "If you are not satisfied with our response, you may contact AFCA on "
                "1800 931 678.\n\n"
                "Yours sincerely,\nFinancial Hardship Team"
            ),
        },
    },
]


def build_prompt(
    complaint_text: str,
    context_docs: list[Any],
    entities: Any,
) -> dict[str, str]:
    """Build v3 few-shot + CoT prompt."""

    # Build context from retrieved docs
    context = ""
    if context_docs:
        context_parts = []
        for i, doc in enumerate(context_docs, 1):
            context_parts.append(
                f"[{i}] {doc.section_id} — {doc.title}\n"
                f"    Source: {doc.guide} | Category: {doc.category}\n"
                f"    {doc.text}"
            )
        context = "\n\n".join(context_parts)

    # Build entity summary
    entity_summary = entities.summary if hasattr(entities, 'summary') else "No entities extracted"

    # Build few-shot examples
    examples_text = ""
    for i, example in enumerate(FEW_SHOT_EXAMPLES, 1):
        ex_out = example["output"]
        examples_text += f"""
---
### Example {i}

**Complaint:** {example['complaint']}

**Classification:**
- Category: {ex_out['category']}
- Sub-category: {ex_out['sub_category']}
- Policy References: {', '.join(ex_out['policy_refs'])}

**Reasoning:** {ex_out['reasoning']}

**Draft Response:** {ex_out['draft_response'][:200]}...
"""

    system = f"""You are an expert AFCA Complaint Triage Analyst with 10+ years of experience in Australian financial services regulation. You have deep expertise in:
- ASIC Regulatory Guides (RG 271 IDR, RG 209 Responsible Lending, RG 272 EDR)
- Insurance Contracts Act 1984
- National Consumer Credit Protection Act 2009
- Banking Code of Practice 2019
- General Insurance Code of Practice 2020
- Life Insurance Code of Practice 2023

Your task is to triage incoming complaints following the exact methodology shown in the examples below.

## IMPORTANT RULES:
1. Always classify using standard AFCA categories
2. Always cite specific regulatory sections — never fabricate references
3. Draft responses must be professional, empathetic, and include AFCA escalation rights
4. Follow the step-by-step reasoning format exactly
5. Consider IDR timeframes: 30 days standard, 21 days for hardship/default, 45 days for super, 90 days for death benefits

## EXAMPLES OF EXPERT TRIAGE:
{examples_text}"""

    user = f"""## NEW COMPLAINT FOR TRIAGE

**Complaint Text:**
{complaint_text}

**Extracted Entities:**
{entity_summary}

---

## Regulatory Knowledge Base (Retrieved):

{context if context else 'No specific regulations retrieved. Use your expert knowledge.'}

---

Now triage this complaint following the same methodology as the examples above. Provide:
1. **Category** and **Sub-category**
2. **Policy References** (specific regulatory sections)
3. **Step-by-step Reasoning**
4. **Professional Draft Response** (include AFCA escalation rights)"""

    return {"system": system, "user": user}
