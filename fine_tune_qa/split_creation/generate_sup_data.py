import json
import random
import os
from tqdm import tqdm
import argparse
import re

from lang_dict import _LANG_NAME

def get_random_item(in_list:list):
    idx = random.randint(0,len(in_list)-1)
    return in_list[idx]

def read_tsv_data(tsv_path:str):
    samples = []
    with open(tsv_path,"r", encoding="utf-8") as f:
        for line in f.readlines():
            samples.append(line.strip().split("\t"))
    return samples
def read_json(path:str):
    with open(path, "r") as f:
        samples = json.load(f)
    return samples

def add_lang_to_samples_clika(train_path, lang, src_lang="en"):
    
    lang = _LANG_NAME[lang]
    src_lang = _LANG_NAME[src_lang]
    langs = []
    for x in range(2001):
        langs.append(src_lang)
    count = 0
    for x in range(0,8200):
        if count == 0:
            langs.append(lang)
            if x%41 == 0:
                count = 4
            else:
                count = 3 
            # print(count)
        else:
            langs.append(src_lang)
            count -= 1
    return langs

def read_opus_data(path:str):
    samples = []
    with open(path) as file:
        get_lines = file.readlines()
        for line in get_lines:   
            samples.append(line.strip())
    return samples

def is_chinese(text):
    for char in text:
        if '\u4e00' <= char <= '\u9fff':
            return True
    return False

def add2list_p(keys, data_list, reverse_prob):
    assert(len(keys)==2 and keys[0]["id"] == keys[1]["id"])
    k1 = keys[0]
    k2 = keys[1]

    c_d = {0:0, 1:0}

    for k in k1.keys():
        if k != "instruction" or k1[k]==None or len(k1[k]) == 0 or k2[k]==None or len(k2[k])==0 or (k1[k] == k2[k]):
            continue
        
        if random.random()<reverse_prob:
            data_list.append([k2[k], k1[k]])
            c_d[1] += 1
        else:
            data_list.append([k1[k], k2[k]])
            c_d[0] += 1
            
    return data_list, c_d

def read_json(path:str):
    with open(path, "r") as f:
        samples = json.load(f)
    return samples


def build_factual_train(languages, out_dir_train, out_dir_test, add_lang):

    assert len(languages)==2, f"Number of languages({len(languages)}) input must be equal to 2."
    path = "../data/fictional_data/Fictional-Dataset/factual_data"

    tgt_lang = languages[1]

    path_ = os.path.join(path, tgt_lang, "test.json")
    with open(path_, "r") as f:
        samples_test = json.load(f)
        if add_lang == True:
            for x in range(len(samples_test)):
                samples_test[x]["language"] = _LANG_NAME[tgt_lang]
        out_path = os.path.join(out_dir_test, f"{tgt_lang}.factual.test.json")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        print(out_path)
        with open(out_path, "w") as f:
            for sample in samples_test:
                f.write(json.dumps(sample, ensure_ascii=False)+"\n")


    tgt_file = os.path.join(path, languages[1], "val.json")
    tgt_data = read_json(tgt_file)
    all_samples = tgt_data
    indices = list(range(len(all_samples )))
    random.shuffle(indices)
    random.seed(42)

    all_samples = [all_samples[i] for i in indices]
    overfited_test = all_samples[:64]

    # train set in simple format
    src_file = os.path.join(path, languages[0], "train.json")
    tgt_file = os.path.join(path, languages[1], "train.json")
    src_data = read_json(src_file)
    tgt_data = read_json(tgt_file)
    if add_lang == True:
        for x in range(len(src_data)):
            src_data[x]["language"] = _LANG_NAME[languages[0]]
        for x in range(len(tgt_data)):
            tgt_data[x]["language"] = _LANG_NAME[languages[1]]
    all_samples = tgt_data+src_data
    indices = list(range(len(all_samples )))
    random.shuffle(indices)
    random.seed(42)
    all_samples = [all_samples[i] for i in indices]
    train_simple = os.path.join(out_dir_train, f"{languages[-1]}.factual.simple.json")
    with open(train_simple, "w") as f:
        for sample in all_samples:
            f.write(json.dumps(sample, ensure_ascii=False)+"\n")

    # validation set simple format
    src_file = os.path.join(path, languages[0], "val.json")
    tgt_file = os.path.join(path, languages[1], "val.json")
    src_data = read_json(src_file)
    tgt_data = read_json(tgt_file)
    if add_lang == True:
        for x in range(len(src_data)):
            src_data[x]["language"] = _LANG_NAME[languages[0]]
        for x in range(len(tgt_data)):
            tgt_data[x]["language"] = _LANG_NAME[languages[1]]
    all_samples = tgt_data + src_data
    validation_path = os.path.join(out_dir_test, f"{languages[1]}.factual.validation.json")
    with open(validation_path, "w") as f:
        for sample in all_samples:
            f.write(json.dumps(sample, ensure_ascii=False)+"\n")





if __name__ == '__main__':
    parser = argparse.ArgumentParser('Script to build training samples for AFP.')

    parser.add_argument('-s', '--seed', type=int, default=0)
    parser.add_argument('-m', '--mode', type=str, default="bilingual", choices=["bilingual", "crosslingual", "translation", "fictional-people", "fictional-clika", "factual-people"])
    parser.add_argument('-l', '--languages', type=str, default=["en", "zh"], nargs="*", help="The list of language names to align, where the first one is the pivotal language.")
    parser.add_argument('-x', '--cross-probability', type=float, default=0.5)
    parser.add_argument('-c', '--change-probability', type=float, default=0.5)
    parser.add_argument('-r', '--repeat', type=float, default=1)
    parser.add_argument('-t', '--translation-file', type=str, default="CrossLingualAlignment/data/opus.en2zh.tsv")
    parser.add_argument('-f', '--source_file', type=str, default="CrossLingualAlignment/data/opus_train_es.en")
    parser.add_argument('-g', '--target_file', type=str, default="CrossLingualAlignment/data/opus_train_es.es")
    parser.add_argument('-b', '--bactrian-dir', type=str, default="./Bactrian-X/data")
    parser.add_argument('-o', '--output-dir-train', type=str, default="../data/train_data")
    parser.add_argument('-v', '--output-dir-test', type=str, default="../data/test_data")
    parser.add_argument('-a', '--add-lang', type=bool, default=False)
    args = parser.parse_args()

    random.seed(args.seed)
    if args.mode == "factual-people":
        build_factual_train(
            languages=args.languages,
            out_dir_train=args.output_dir_train,
            out_dir_test=args.output_dir_test,
            add_lang=args.add_lang
        )
