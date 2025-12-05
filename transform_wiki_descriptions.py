import os
import pandas as pd
import random

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OPEN_AI_KEY = os.getenv("OPEN_AI_KEY")

client = OpenAI(api_key=OPEN_AI_KEY)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH = os.path.join(BASE_DIR, "Data", "val_common_wiki_test.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "Data", "transformed_wiki_descriptions.csv")

SEED = 42


def generate_user_description(row):
    """
    Generate a user description that is similar to wiki description.
    """
    prompt = f"""
    You are given a single wiki plant description.

    Name: {row['name']}
    Wiki Description: {row['description']}

    Please transform the wiki description into a SINGLE plant description from the perspective of a *user shopping for a plant*.  
    The description must:

    - be inspired by plant traits,
    - express what the user is looking for in a plant (tone: “I’m looking for…”),
    - NOT mention any specific plant names,
    - NOT include any scientific names (including those in parentheses),
    - NOT copy any full sentences from the provided descriptions,
    - be written in natural consumer-friendly language,
    - be 2–3 sentences long,
    - be general enough that more than one of the five plants could match it.

    Return ONLY the description text, written as if the user is asking for a plant recommendation.
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150,
        temperature=0.7,
    )

    return response.choices[0].message.content.strip()


def main():
    df = pd.read_csv(INPUT_PATH)
    df = df.rename(columns={df.columns[0]: "name", df.columns[1]: "description"})

    all_rows = []

    # FIXED: iterate over rows, not columns
    for i, row in df.iterrows():
        if i > 100:
            break
        user_desc = generate_user_description(row)
        print("Generated user description")

        all_rows.append({
            "user_description": user_desc,
            "wiki_name": row["name"],
            "wiki_description": row["description"]
        })

    out_df = pd.DataFrame(all_rows)
    out_df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nGenerated {len(out_df)} rows and saved to {OUTPUT_PATH}")
    print(out_df.head())


if __name__ == "__main__":
    main()
