import os
import re
import json
import datetime
from collections import defaultdict, Counter
import torchtext
import en_core_web_lg
from spacy.attrs import ORTH, NORM
import matplotlib.pyplot as plt


# dataset_dir = "/home/Tanaka/generate-commentary/dataset/lol/annotation/deepsegment"
dataset_dir = "/home/Tanaka/densecap/data/anet/ActivityNet/annotation"
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
stop_words = ["the", "a", "that", "'s", "able", "okay", "going", "so", "just", "now", "very", "yeah", "an", "oh", "too"]
not_use_list = ["$", "``", "''", ",", ":", "."]


def analyze_vocab(dataset_dir, max_length=20):
    # build vocab and tokenized sentences

    text_proc = torchtext.data.Field(sequential=True, init_token='<init>',
                                     eos_token='<eos>',
                                     tokenize=tokenize,
                                    #  tokenize='spacy',
                                    #  stop_words=stop_words,
                                     lower=True, batch_first=True,
                                     fix_length=max_length,
                                     preprocessing=torchtext.data.Pipeline(preprocessing)
                                     )

    dataset_files = [
        os.path.join(
            dataset_dir,
            x +
            '.json') for x in [
            'training',
            'validation']]

    sentences = []
    nsentence = 0
    for dataset_file in dataset_files:
        with open(dataset_file, 'r') as data_file:
            data = json.load(data_file)

        for vid, val in data.items():
            nsentence += len(val['sentences'])
            sentences.extend(val['sentences'])

    # build vocab on train and val
    sentences_proc = list(map(text_proc.preprocess, sentences))
    text_proc.build_vocab(sentences_proc, min_freq=5)
    # vocab = text_proc.vocab.freqs
    # from IPython import embed
    # embed()
    # exit()


pos_dict = defaultdict(list)


def tokenize(text):
    text = text.replace('-', "")
    for cls, nouns in noun_cls_pre.items():
        for noun in nouns:
            if noun in text.lower():
                if noun == 'i g ':
                    text = text.lower().replace(noun, cls + ' ')
                else:
                    text = text.lower().replace(noun, cls)
    doc = nlp(text)
    tokens = [e.text for e in doc]
    for e in doc:
        if e.tag_ not in not_use_list:
            pos_dict[e.tag_].append(e.text)
    return tokens


def preprocessing(s):
    for cls, nouns in noun_cls_post.items():
        for noun in nouns:
            if s == noun:
                return '<' + cls + '>'
    for cls in noun_cls_pre:
        if s == cls:
            return '<' + cls + '>'
    s = re.sub(r'\d+\.*\d*', '0', s)
    return s


analyze_vocab(dataset_dir)

en_jp = {'Conjunction': ['CC'], 'Numeral': ['CD'], 'Determiner': ['DT', 'PDT', 'PRP$', 'WP$'], 'Pronoun': ['EX', 'PRP', 'WP'], 'Adposition': ['IN', 'RP'],
         'Adjective': ['JJ', 'JJR', 'JJS', 'WRB'], 'Verb': ['MD', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'], 'Noun': ['NN', 'NNS'],
         'Proper Noun': ['NNP', 'NNPS'], 'Particle': ['POS', 'TO'], 'Adverb': ['RB', 'RBR', 'RBS'], 'Interjection': ['UH']}
log_dir = "/home/Tanaka/generate-commentary/tools/information/ActivityNet/"
log_dir = os.path.join(log_dir, str(datetime.date.today()))
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
pos_dict_size = {pos: len(pos_dict[pos]) for pos in pos_dict}
pos_jp_dict = {x: 0 for x in en_jp.keys()}
all_count = 0
for pos, count in pos_dict_size.items():
    all_count += count
    for jp, en_list in en_jp.items():
        if pos in en_list:
            pos_jp_dict[jp] += count
pos_dict_sorted = sorted(pos_jp_dict.items(), key=lambda x: x[1], reverse=True)
pos_list = [x[0] for x in pos_dict_sorted][::-1]
freq_list = [x[1] for x in pos_dict_sorted][::-1]
print(pos_jp_dict['Proper Noun'] / all_count)
print(pos_jp_dict['Numeral'] / all_count)
fig = plt.figure()
ax1 = fig.add_subplot(111, xlabel='frequency', ylabel='part-of-speech')
barlist = ax1.barh(pos_list, freq_list)
barlist[1].set_color('r')
barlist[2].set_color('r')
plt.tight_layout()
plt.savefig(os.path.join(log_dir, "pos_frequency" + ".pdf"))
