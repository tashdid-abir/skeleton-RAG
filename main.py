import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

API_KEY = os.environ["OPENROUTER_API_KEY"]
MODEL = os.environ["OPENROUTER_MODEL"]

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=API_KEY,
)

PASSAGES = """
[P1] Cell phone or any other unauthorized electronic devices are not allowed in the examination hall.

[P2] Students shall not be allowed to sit for the examination if they are late by 30 minutes or more.

[P3] No examinee shall be allowed to leave the examination hall or submit an answer script within 45 minutes of the start of the examination.

[P4] No examinee is allowed to go outside for any reason within the first hour of the start of the examination.

[P5] No examinee is allowed to borrow or lend any material within the examination hall.

[P6] Students must deposit bags, purses, and any paper not approved by the invigilator in an area specified by the invigilator before the examination starts.

[P7] Possession of any written unauthorized chit, paper, book, material, or aid related to the examination subject is treated as an unfair means.

[P8] Pencil is not allowed for writing in the answer script. It may be used only for drawing figures.

[P9] Students are not allowed to wear caps in the examination hall except for religious or medical reasons.

[P10] Students are not allowed to wear sunglasses in the examination hall except for medical reasons.

[P11] Students with a medical or special requirement may notify the invigilator in advance.

[P12] Any non-compliance with examination rules—including verbal communication, visual copying, unauthorized materials, or other unfair means—is punishable and will result in disciplinary action.
"""

QUESTION = "Can I leave the examination hall 30 minutes after it begins?"

response = client.chat.completions.create(
    model=MODEL,
    messages=[
        {
            "role": "system",
            "content": 
            """
                You answer questions about university examination rules.

                Use only the supplied policy passages.
                If the passages do not contain the answer, reply exactly: unanswerable.
                Do not use outside knowledge or make assumptions.
                Keep the answer concise.

                After every answer, add a new line in this format:
                Citations: P1, P2

                Cite only passage IDs that directly support the answer.
            """,
        },
        
        {
            "role": "user",
            "content": 
            f"""
                Policy passages:

                {PASSAGES}

                Question: {QUESTION}
            """,
        },
    ],
)

print(response.choices[0].message.content)