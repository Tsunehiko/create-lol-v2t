import os
import re
import json
from collections import defaultdict, Counter  # noqa
import en_core_web_lg
from spacy.attrs import ORTH, NORM  # noqa
from tqdm import tqdm


nlp = en_core_web_lg.load()
rule_file = "/home/Tanaka/generate-commentary/dataset/abbreviation.json"
with open(rule_file, 'r') as f:
    abbreviation = json.load(f)
for text, rules in abbreviation.items():
    case = [{ORTH: rule} for rule in rules]
    nlp.tokenizer.add_special_case(text, case)
dict_file = "/home/Tanaka/generate-commentary/dataset/dictionary.json"
with open(dict_file, 'r') as f:
    noun_cls = json.load(f)
noun_cls_pre = defaultdict(list)
noun_cls_post = defaultdict(list)
for cls, nouns in noun_cls.items():
    for noun in nouns:
        if len(noun.split()) > 1:
            noun_cls_pre[cls].append(noun.lower())
        else:
            noun_cls_post[cls].append(noun.lower())


def preprocess(sentence):
    sentence = sentence.replace('-', "")
    for cls, nouns in noun_cls_pre.items():
        for noun in nouns:
            if noun in sentence.lower():
                if noun == 'i g ':
                    sentence = sentence.lower().replace(noun, cls + ' ')
                else:
                    sentence = sentence.lower().replace(noun, cls)
    doc = nlp(sentence)
    tokens = [e.text for e in doc]
    processed_tokens = []
    for s in tokens:
        isProcessed = False
        for cls, nouns in noun_cls_post.items():
            for noun in nouns:
                if s == noun and not isProcessed:
                    processed_tokens.append('<' + cls + '>')
                    isProcessed = True
                    break
        for cls in noun_cls_pre:
            if s == cls and not isProcessed:
                processed_tokens.append('<' + cls + '>')
                isProcessed = True
                break
        if not isProcessed:
            s = re.sub(r'\d+\.*\d*', '0', s)
            processed_tokens.append(s)
    return ' '.join(processed_tokens)


dataset_dir = "/home/Tanaka/generate-commentary/dataset/lol/annotation/deepsegment"
splits = ['validation', 'testing', 'training']

output_dir = "/home/Tanaka/generate-commentary/dataset/lol/annotation/evaluate_1"
new_dict = {}
for split in splits:
    print("-" * 10 + f' {split} ' + "-" * 10)
    dataset_file = os.path.join(dataset_dir, split + '.json')
    with open(dataset_file, 'r') as data_file:
        data = json.load(data_file)

    new_dict = {}
    for vid, val in tqdm(data.items()):
        sentences = val['sentences']
        new_sentences = []
        for sentence in sentences:
            new_sentence = preprocess(sentence)
            new_sentences.append(new_sentence)
        new_dict.update({vid: {'duration': val['duration'], 'timestamps': val['timestamps'], 'sentences': new_sentences}})
    
    output_path = os.path.join(output_dir, split + '.json')
    with open(output_path, 'w') as f:
        json.dump(new_dict, f)
