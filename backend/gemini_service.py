from google import genai
from backend.config import GEMINI_API_KEY

client = genai.Client(api_key=GEMINI_API_KEY)


def generate_incident_summary(
    threat_score,
    risk_level,
    xai_explanation
):
    prompt = f"""
You are a CCTV security analyst.

Threat Score: {threat_score}
Risk Level: {risk_level}

Explainable AI Reason:
{xai_explanation}

Write a short professional incident summary
in 3-4 sentences.
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )

    return response.text