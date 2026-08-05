import google.generativeai as genai

from backend.config import GEMINI_API_KEY


# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)

model = genai.GenerativeModel("gemini-1.5-flash")


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

    response = model.generate_content(prompt)

    return response.text