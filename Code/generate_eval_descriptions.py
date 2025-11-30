import os
import pandas as pd
import random

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OPEN_AI_KEY = os.getenv("OPEN_AI_KEY")

client = OpenAI(api_key=OPEN_AI_KEY)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH = os.path.join(BASE_DIR, "Data", "generated_descriptions.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "Data", "eval_descriptions.csv")

NUM_SETS = 40           # number of seed sets to generate
SET_SIZE = 5            # 5 recommended items per seed
SEED = 42               # reproducibility


def generate_seed_description(sampled_rows):
    """
    Generate one seed description that is similar to the 5 chosen plants.
    """
    prompt = f"""
    You are given descriptions of 5 different plants.

    {chr(10).join([f"{i+1}. Name: {row['name']}\n   Description: {row['description']}" 
                    for i, row in enumerate(sampled_rows)])}

    Please write a SINGLE plant description from the perspective of a *user shopping for a plant*.  
    The description must:

    - be inspired by traits that overlap across these 5 plants,
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
    random.seed(SEED)

    for set_idx in range(NUM_SETS):
        # sample 5 random plants for this set
        sampled = df.sample(SET_SIZE).reset_index(drop=True)

        # generate one seed description
        seed_desc = generate_seed_description(sampled.to_dict("records"))
        print(f"[{set_idx+1}/{NUM_SETS}] Generated seed description")

        # create 5 rows
        for _, row in sampled.iterrows():
            all_rows.append({
                "seed_description": seed_desc,
                "recommended_title": row["name"],
                "recommended_description": row["description"]
            })

    out_df = pd.DataFrame(all_rows)
    out_df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nGenerated {len(out_df)} rows and saved to {OUTPUT_PATH}")
    print(out_df.head())


if __name__ == "__main__":
    main()
