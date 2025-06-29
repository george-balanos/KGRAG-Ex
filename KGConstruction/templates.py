EXTRACT_ENTITIES_TEMPLATE = """
You are an expert biomedical information extractor.

Your task is to extract all valid **Entity1 - Relationship - Entity2** triples from the medical text below. Each triple must follow strict semantic and formatting rules.

---

🔹 **Valid Entity Types (Labels):**
- Diseases
- Medications
- Symptoms
- Treatments
- Diagnostic Tests
- Risk Factors
- Body Parts

🔹 **Valid Relationship Types:**
- HAS_SYMPTOM
- AFFECTS
- DIAGNOSED_BY
- TREATED_WITH
- MANAGED_BY
- HAS_RISK_FACTOR
- COMORBID_WITH (Symmetrical)
- CAUSED_BY
- INDICATES
- OCCURS_IN
- USED_FOR
- INVOLVES_MEDICATION
- HAS_SIDE_EFFECT
- CONTRAINDICATED_FOR
- DETECTS
- MEASURES
- INCREASES_RISK_OF
- PART_OF
- CAN_BE_AFFECTED_BY

---

📌 **Instructions:**
1. Only extract triples where:
   - Both **Entity1** and **Entity2** are mentioned in the input text.
   - Both entities have a valid label from the list above.
   - The relationship is one of the predefined types and is semantically correct.
2. Only extract **forward relationships**.
   - ❌ Do NOT include reverse/inverse relationships (e.g., only include `Hypertension : Diseases : INCREASES_RISK_OF : Stroke : Diseases`, not the reverse).
3. Do not infer data that is not explicitly stated in the text.
4. Avoid duplicates and redundant triples.
5. If no valid triples exist, return nothing.

---

📤 **Output Format:**
- One triple per line.
- **Must begin with**: `Answer:`
- **Format exactly like**: `Answer: Entity1 : Label1 : Relationship : Entity2 : Label2`
- Do **not** include:
  - Numbering (❌ 1., 2., etc.)
  - Explanatory text
  - Introductory or summary sentences
  - Headings or bullet points

✅ **Correct Examples:**
Answer: Hypertension : Diseases : INCREASES_RISK_OF : Stroke : Diseases  
Answer: Diabetes : Diseases : HAS_SYMPTOM : Frequent urination : Symptoms  
Answer: Stroke : Diseases : TREATED_WITH : tPA : Treatments

---

⚠️ Important:
This is machine-readable output. Do **not** include any additional text, comments, headers, or formatting.

⬇️ Begin your output below this line:
"""




GENERATE_MEDICAL_SUMMARY = f"""
Generate a concise medical summary for the following patient.
Focus on key health conditions, risks, and possible interventions.

Patient data: 
"""