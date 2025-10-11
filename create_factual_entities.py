from SPARQLWrapper import SPARQLWrapper, JSON
import pandas as pd
import time
import random
import urllib.error

endpoint_url = "https://query.wikidata.org/sparql"
sparql = SPARQLWrapper(endpoint_url)

def run_query(query, retries=5):
    for attempt in range(retries):
        try:
            sparql.setQuery(query)
            sparql.setReturnFormat(JSON)
            return sparql.query().convert()
        except urllib.error.HTTPError as e:
            if e.code in [429, 504]:
                wait = (2 ** attempt) + random.random()
                print(f"HTTP {e.code}, retrying in {wait:.1f}s...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Query failed after max retries")

def main():
    all_data = {}
    batch_size = 20
    offset = 0

    while len(all_data) < 1000:
        step1_query = f"""
        SELECT ?person ?personLabel WHERE {{
          ?person wdt:P31 wd:Q5 .  # Human
          FILTER NOT EXISTS {{ ?person wdt:P106 wd:Q36080 }}  # No fictional characters
          FILTER NOT EXISTS {{ ?person wdt:P106 wd:Q82955 }}  # No mythological characters
          FILTER NOT EXISTS {{ ?person wdt:P106 wd:Q183318 }} # No literary characters
          FILTER EXISTS {{ ?person rdfs:label ?mulLabel . FILTER(LANG(?mulLabel)="mul") }}
        }}
        LIMIT 50
        OFFSET {offset}
        """
        results = run_query(step1_query)
        candidate_ids = [r["person"]["value"].split("/")[-1] for r in results["results"]["bindings"]]
        if not candidate_ids:
            print("No more candidates available.")
            break

        for i in range(0, len(candidate_ids), batch_size):
            batch_ids = candidate_ids[i:i+batch_size]
            ids_str = " ".join(f"wd:{pid}" for pid in batch_ids)

            # Updated step2_query with human settlement filter and multilingual city names
            step2_query = f"""
            SELECT ?person ?personLabel ?birthDate ?deathDate 
                   (GROUP_CONCAT(DISTINCT ?birthPlaceLabel; SEPARATOR=" | ") AS ?birthPlaceNames)
            WHERE {{
              VALUES ?person {{ {ids_str} }}
              ?person wdt:P569 ?birthDate .
              ?person wdt:P570 ?deathDate .
              ?person wdt:P19 ?birthPlace .
              ?birthPlace wdt:P31 ?placeType .
              ?placeType (wdt:P279*) wd:Q486972 .  # Must be a human settlement

              ?birthPlace rdfs:label ?birthPlaceLabel .
              SERVICE wikibase:label {{ bd:serviceParam wikibase:language "mul,en". }}
            }}
            GROUP BY ?person ?personLabel ?birthDate ?deathDate
            """

            results = run_query(step2_query)
            for r in results["results"]["bindings"]:
                pid = r["person"]["value"].split("/")[-1]
                if pid not in all_data:
                    all_data[pid] = {
                        "id": pid,
                        "name": r["personLabel"]["value"],
                        "birth_date": r["birthDate"]["value"],
                        "death_date": r["deathDate"]["value"],
                        "city": r["birthPlaceNames"]["value"]
                    }
            time.sleep(2)

        offset += 50
        print(f"Collected {len(all_data)} unique people so far...")

    df = pd.DataFrame(list(all_data.values()))
    df.to_csv("factual_people.csv", index=False)
    print(df.head())
    print(f"Retrieved {len(df)} unique people")

def process_entities():
    df = pd.read_csv("factual_people.csv")

    # Filter out invalid dates
    df = df[~df["birth_date"].str.startswith("http", na=False)]
    df = df[~df["death_date"].str.startswith("http", na=False)]

    # Clean date fields to only year
    df["birth_date"] = df["birth_date"].astype(str).str[:4].apply(
        lambda x: str(int(x)) if x.isdigit() else x
    )
    df["death_date"] = df["death_date"].astype(str).str[:4].apply(
        lambda x: str(int(x)) if x.isdigit() else x
    )

    df.to_csv("factual_people.csv", index=False)

if __name__ == "__main__":
    main()
    process_entities()
