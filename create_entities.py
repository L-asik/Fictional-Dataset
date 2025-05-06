import os 
import pandas as pd
import random
import numpy as np

path_city = os.path.join("data","pl.csv")
path_surname = os.path.join("data","nazwiska_męskie-osoby_żyjące_w_podziale_na_województwo_zameldowania.csv")
path_first_name = os.path.join("data","8_-_WYKAZ_IMION_MĘSKICH_OSÓB_ŻYJĄCYCH_WG_POLA_IMIĘ_PIERWSZE_WYSTĘPUJĄCYCH_W_REJESTRZE_PESEL_BEZ_ZGONÓW.csv")

SEED = 42

def init_seeds(seed):
    """
    Initialize the random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)

def create_cities_base(path, number_of_cities):

    df = pd.read_csv(path, sep=",")
    if number_of_cities > len(df):
        raise ValueError("number_of_cities is greater than the number of rows in the dataframe")
    df = df.sort_values(by="population_proper", ascending=False).head(number_of_cities)
    return df["city"].tolist()

def create_surnames_base(path, number_of_surnames, threshold=0.05):
    df = pd.read_csv(path, sep=",")
    if number_of_surnames > len(df):
        raise ValueError("number_of_surnames is greater than the number of rows in the dataframe")
    df = df.sort_values(by="Liczba", ascending=True)
    thresholded = df[df["Liczba"] < threshold * df["Liczba"].max()]
    if len(thresholded) < number_of_surnames:
        df = df.head(number_of_surnames)
    else:
        df = thresholded.sample(n=number_of_surnames, replace=False, random_state=SEED)
    list_of_surnames = df["Nazwisko aktualne"].tolist()
    for x in range(len(list_of_surnames)):
        list_of_surnames[x] = list_of_surnames[x][0] + list_of_surnames[x][1:].lower()

    return list_of_surnames

def create_first_names_base(path, number_of_first_names, upper_threshold=1, lower_threshold=0.10):
    df = pd.read_csv(path, sep=",")
    if number_of_first_names > len(df):
        raise ValueError("number_of_first_names is greater than the number of rows in the dataframe")
    
    df = df.sort_values(by="LICZBA WYSTĄPIEŃ", ascending=True)
    
    min_count = lower_threshold * df["LICZBA WYSTĄPIEŃ"].max()
    max_count = upper_threshold * df["LICZBA WYSTĄPIEŃ"].max()
    # print(min_count)
    # print(max_count)

    thresholded = df[
        (df["LICZBA WYSTĄPIEŃ"] >= min_count) & 
        (df["LICZBA WYSTĄPIEŃ"] < max_count)
    ]
    # print("Thresholded")
    # print(len(thresholded))

    if len(thresholded) < number_of_first_names:
        thresholded = thresholded
    else:
        thresholded = thresholded.sample(n=number_of_first_names, replace=False, random_state=SEED)
    list_of_first_names = thresholded["IMIĘ PIERWSZE"].tolist()
    for x in range(len(list_of_first_names)):
        list_of_first_names[x] = list_of_first_names[x][0] + list_of_first_names[x][1:].lower()
    return list_of_first_names



def create_birth_death(num_samples, seed=SEED):
    rng = np.random.default_rng(seed)  
    birthdates = rng.uniform(low=1800, high=1940, size=num_samples).astype(int)
    
    life_expectancy = rng.normal(loc=65, scale=10, size=num_samples).astype(int)
    life_expectancy = np.clip(life_expectancy, 0, None)
    deathdates = birthdates + life_expectancy
    
    return birthdates, deathdates

def create_retalations(firstnames, surnames, cities, birthdates, deathdates, samples):
    people = {}
    for i in range(samples):
        first_name = firstnames[i % len(firstnames)]
        last_name = surnames[i]
        name = f"{first_name} {last_name}"
        city = cities[i % len(cities)]
        birthdate = birthdates[i]
        deathdate = deathdates[i]
        people[i] = {
            "name": name,
            "city": city,
            "birth_date": str(birthdate),
            "death_date": str(deathdate)
        }
    return people

def main():
    init_seeds(SEED)
    firstnaems = create_first_names_base(path_first_name, 400)
    surnames = create_surnames_base(path_surname, 400)
    cities = create_cities_base(path_city, 20)
    birthdates, deathdates = create_birth_death(400)
    print(len(firstnaems), len(surnames), len(cities), len(birthdates), len(deathdates))
    people = create_retalations(firstnaems, surnames, cities, birthdates, deathdates, 400)
    df = pd.DataFrame.from_dict(people, orient="index")
    path_save = os.path.join("data", "people.csv")  
    df.to_csv(path_save, index=False)

if __name__ == "__main__":
    main()