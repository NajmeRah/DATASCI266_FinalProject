import os
import pandas as pd
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OPEN_AI_KEY = os.getenv("OPEN_AI_KEY")
client = OpenAI(api_key=OPEN_AI_KEY)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH = os.path.join(BASE_DIR, "Data", "val_common_wiki_test.csv")
GENERATED_PATH = os.path.join(BASE_DIR, "Data", "perenualData_clean.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "Data", "transformed_wiki_descriptions.csv")

MAX_WORKERS = 10          # number of parallel OpenAI calls


def generate_user_description(row):
    """
    Generate a user description that is similar to wiki description.
    With retry logic.
    """
    prompt = f"""
    You are given a single wiki plant description.

    Name: {row['name']}
    Wiki Description: {row['description']}

    Please transform the wiki description into a SINGLE plant description from the perspective 
    of a *user shopping for a plant*.

    The description must:
    - be inspired by plant traits,
    - express what the user is looking for in a plant (tone: “I’m looking for…”),
    - NOT mention any specific plant names,
    - NOT include any scientific names (including those in parentheses),
    - NOT copy any full sentences from the provided descriptions,
    - be written in natural consumer-friendly language,
    - be 2–3 sentences long,
    - be general enough that more than one of the five plants could match it.

    Return ONLY the description text.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"Error on {row['name']}: {e}")

    return None


def process_row(row):
    """Function executed in parallel for each DataFrame row."""
    user_desc = generate_user_description(row)
    return {
        "user_description": user_desc,
        "name": row["name"],
        "wiki_description": row["description"],
        "perenual_data_text": row["perenual_data_text"],
    }


def main():
    # Load wiki data
    df = pd.read_csv(INPUT_PATH)
    df = df.rename(columns={df.columns[0]: "name", df.columns[1]: "description"})
    df = df.drop_duplicates(subset="name", keep="last")

    # Load generated descriptions
    gen_df = pd.read_csv(GENERATED_PATH)
    gen_df = gen_df.rename(columns={gen_df.columns[1]: "name", gen_df.columns[25]: "perenual_data_text"})
    gen_df = gen_df[["name", "perenual_data_text"]]
    gen_df = gen_df.drop_duplicates(subset="name", keep="last")

    # Merge on name
    merged = df.merge(gen_df, on="name", how="left")

    results = []

    # --- PARALLEL EXECUTION ---
    print(f"Starting parallel generation with {MAX_WORKERS} workers...")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_row, row): idx for idx, row in merged.iterrows()}

        for future in as_completed(futures):
            idx = futures[future]
            try:
                result = future.result()
                results.append(result)
                print(f"Finished row {idx}: {result['name']}")
            except Exception as exc:
                print(f"Row {idx} generated an exception: {exc}")

    # Convert results to DataFrame
    out_df = pd.DataFrame(results)
    out_df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nGenerated {len(out_df)} rows and saved to {OUTPUT_PATH}")
    print(out_df.head())


if __name__ == "__main__":
    main()

