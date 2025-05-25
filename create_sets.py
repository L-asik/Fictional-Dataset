import pandas as pd 
import os 
from sklearn.model_selection import train_test_split
import json
import random

_LANG_NAME={
    "af": "Afrikaans",
	"ar": "Arabic",
	"az": "Azerbaijani",
	"bn": "Bengali",
	"cs": "Czech",
	"de": "German",
	"en": "English",
	"es": "Spanish",
	"et": "Estonian",
	"fa": "Persian",
	"fi": "Finnish",
	"fr": "French",
	"gl": "Galician",
	"gu": "Gujarati",
	"he": "Hebrew",
	"hi": "Hindi",
	"hr": "Croatian",
	"id": "Indonesian",
	"it": "Italian",
	"ja": "Japanese",
	"ka": "Georgian",
	"kk": "Kazakh",
	"km": "Khmer",
	"ko": "Korean",
	"lt": "Lithuanian",
	"lv": "Latvian",
	"mk": "Macedonian",
	"ml": "Malayalam",
	"mn": "Mongolian",
	"mr": "Marathi",
	"my": "Burmese",
	"ne": "Nepali",
	"nl": "Dutch",
	"pl": "Polish",
	"ps": "Pashto",
	"pt": "Portuguese",
	"ro": "Romanian",
	"ru": "Russian",
	"si": "Sinhala",
	"sl": "Slovene",
	"sv": "Swedish",
	"sw": "Swahili",
	"ta": "Tamil",
	"te": "Telugu",
	"th": "Thai",
	"tl": "Tagalog",
	"tr": "Turkish",
	"uk": "Ukrainian",
	"ur": "Urdu",
	"vi": "Vietnamese",
	"xh": "Xhosa",
	"zh": "Chinese",
}
def read_files_and_split(languages):
    src_lang_path = languages[0]+"_formats.json"
    src_lang_path = os.path.join("templates", src_lang_path)
    src_json = pd.read_json(src_lang_path)
    tgt_lang_path = languages[1]+"_formats.json"
    tgt_lang_path = os.path.join("templates", tgt_lang_path)
    tgt_json = pd.read_json(tgt_lang_path)

    path = os.path.join("data", "people.csv")
    src_lang_train = pd.read_csv(path)

    tgt_lang_train, tgt_lang_test = train_test_split(src_lang_train, test_size=0.6, random_state=42)
    tgt_lang_val, tgt_lang_test = train_test_split(tgt_lang_test, test_size=0.75, random_state=42)
    tgt_lang_train.sort_index(inplace=True)
    
    return src_json, tgt_json, src_lang_train, tgt_lang_train, tgt_lang_test, tgt_lang_val




def fill_the_templates(dataframe, template_json, language, typ, for_CLA_format=False, parallel = None):
    dir = os.path.join("data", language)
    os.makedirs(dir, exist_ok=True)
    if for_CLA_format:
        path = os.path.join(dir, f"{typ}_cla.json")
    else:
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
                    if key =="Place of living":
                        answer = answer.replace("{location}", person[1]["city"])
                        tags.append(person[1]["city"])
                    if key=="Birth":
                        answer = answer.replace("{date}", str(person[1]["birth_date"]))
                        tags.append(str(person[1]["birth_date"]))
                    if key=="Death":
                        answer = answer.replace("{date}", str(person[1]["death_date"]))
                        tags.append(str(person[1]["death_date"]))
                    if for_CLA_format:
                        if parallel is not None:
                            sent0 = parallel[0][idx]
                            sent1 = parallel[1][idx]
                                
                        else: 
                            sent0 = ""
                            sent1 = ""
                        clm_text = "Question: "+ question + " Answer: "+ answer
                        dictr = {   
                            "sent0": sent0,
                            "sent1": sent1,
                            "clm_text": clm_text,
                            "clm_prompt_len": len(clm_text) - len(answer),
                        }
                        
                    else:
                        dictr["prompt"] = question
                        dictr["answer"] = answer
                        if typ != "train":
                            dictr["tags"] = tags
                    all_data.append(dictr)
                    idx += 1
        json.dump(all_data, outfile, ensure_ascii=False, indent=2)
                    
def parallel_prep(parallel, templates):
    sents = {}

    for id, template in enumerate(templates):
        sents[id] = []
        for person in parallel.iterrows():
            for key in template.keys():
                for value in template[key]:
                    question = value["question"]
                    question = question.replace("{name}", person[1]["name"])
                    sents[id].append(question)
    return sents

def main():
    languages = ["en","zh"]
    src_json, tgt_json, src_lang_train, tgt_lang_train, tgt_lang_test, tgt_lang_val = read_files_and_split(languages)
    parallel = parallel_prep(tgt_lang_train,[tgt_json, src_json])

    #training set containing all of the people and questions and answers in English
    fill_the_templates(src_lang_train, src_json, languages[0], "train", for_CLA_format=False)
    #training set containing half of the people and questions and answers in Chinese
    fill_the_templates(tgt_lang_train, tgt_json, languages[1], "train", for_CLA_format=False)
    #test set containing the other half of the people and questions and answers in Chinese
    fill_the_templates(tgt_lang_test, tgt_json, languages[1], "test", for_CLA_format=False)
    # #control set containing the same half as the chinese test set but in English
    # fill_the_templates(tgt_lang_test, src_json, languages[0], "test", for_CLA_format=False)

    #training sets that are in the format that is compatible with the CLA pipeline
    fill_the_templates(src_lang_train, src_json, languages[0], "train", for_CLA_format=True)
    fill_the_templates(tgt_lang_train, tgt_json, languages[1], "train", for_CLA_format=True)

    #validation set
    fill_the_templates(tgt_lang_val, tgt_json, languages[1], "val", for_CLA_format=False)
    # fill_the_templates(tgt_lang_val, src_json, languages[0], "val", for_CLA_format=False)

    #fill the parallel sentences using questions that overlap in both languages
    # fill_the_templates(tgt_lang_train, tgt_json, languages[1], "train", for_CLA_format=True, parallel=parallel)

    # fill_the_templates(tgt_lang_test, tgt_json, languages[1], "test", for_CLA_format=True)

if __name__ == "__main__":
    main()