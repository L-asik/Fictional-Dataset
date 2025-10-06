import pandas as pd 
import os 
from sklearn.model_selection import train_test_split
import json
import random
from fine_tune_qa.utils.globs import _LANG_NAME

def read_files_and_split(language):
    src_lang_path = language+"_formats.json"
    src_lang_path = os.path.join("templates_factual", src_lang_path)
    src_json = pd.read_json(src_lang_path)

    path = os.path.join("data", "factual_people.csv")
    src_lang_train = pd.read_csv(path)

    src_lang_train, src_lang_test = train_test_split(src_lang_train, test_size=0.5, random_state=42)
    src_val = src_lang_train.sample(frac=0.1, random_state=42)
    # tgt_lang_val, tgt_lang_test = train_test_split(tgt_lang_test, test_size=0.75, random_state=42)
    return src_json, src_lang_train, src_val, src_lang_test




def fill_the_templates(dataframe, template_json, language, typ, for_CLA_format=False, parallel = None):
    dir = os.path.join("factual_data", language)
    os.makedirs(dir, exist_ok=True)


    path = os.path.join(dir, f"{typ}.json") 

    print(path)
    all_data = []
    idx = 0
    with open(path, "w", encoding="utf-8") as outfile:
        for person in dataframe.iterrows():
            for key in template_json.keys():
                for value in template_json[key]:
                    dictr = {}
                    tags = []
                    question = value["question"]
                    answer = value["answer"]
                    question = question.replace("{name}", person[1]["name"])
                    answer = answer.replace("{name}", person[1]["name"])
                    tags.append(person[1]["name"])
                    if key =="Place of birth":
                        answer = answer.replace("{location}", person[1]["city"])
                        tags.append(person[1]["city"])
                    if key=="Birth":
                        answer = answer.replace("{date}", str(person[1]["birth_date"]))
                        tags.append(str(person[1]["birth_date"]))
                    if key=="Death":
                        answer = answer.replace("{date}", str(person[1]["death_date"]))
                        tags.append(str(person[1]["death_date"]))

                    dictr["prompt"] = question
                    dictr["answer"] = answer
                    if typ != "train":
                        dictr["tags"] = tags
                    all_data.append(dictr)
                    idx += 1
        json.dump(all_data, outfile, ensure_ascii=False, indent=2)
                    

def main():
    languages = ["pl", "cs", "en", "zh"]
    for language in languages:
        src_json, src_lang_train, src_lang_val, src_lang_test = read_files_and_split(language)
        

        #training set containing all of the people and questions and answers in English
        fill_the_templates(src_lang_train, src_json, language, "train", for_CLA_format=False)

        #test set
        fill_the_templates(src_lang_test, src_json, language, "test", for_CLA_format=False)

        #validation set
        fill_the_templates(src_lang_val, src_json, language, "val", for_CLA_format=False)



if __name__ == "__main__":
    main()