import os
import re
import json
import nltk
from collections import defaultdict, Counter
import torchtext
from nltk.corpus import stopwords  # noqa
import en_core_web_lg
from spacy.attrs import ORTH, NORM


def get_vocab_lol(dataset_dir, result_path, sampling_rate=100, max_length=20):
    dataset_files = [
        os.path.join(
            dataset_dir,
            x +
            '.json') for x in [
            'training',
            'validation']]

    sentences = []
    for dataset_file in dataset_files:
        with open(dataset_file, 'r') as data_file:
            data = json.load(data_file)

        for _, val in data.items():
            sentences.extend(val['sentences'])

    with open(result_path, mode='w') as f:

        for i, sentence in enumerate(sentences):
            if i % sampling_rate == 0 and len(sentence) > 100:
                f.write('-' * 50 + '\n')
                f.write(sentence + '\n')

        word_pos = []
        for sentence in sentences:
            morph = nltk.word_tokenize(sentence)
            pos = nltk.pos_tag(morph)
            word_pos.extend(pos)

        pos_word = defaultdict(list)
        for (word, pos) in word_pos:
            pos_word[pos].append(word)

        # extract_pos = ['NN', 'NNS', 'NNP', 'NNPS', 'PRP', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
        for tag in pos_word.keys():
            c = Counter(pos_word[tag]).most_common()
            # if tag != 'NNP':
            #     c = c[:20]
            f.write(f'---- {tag} ----\n')
            words = ""
            for word, count in c:
                words += f"{word}: {count}, "
            f.write(words + '\n')

    # for pos in pos_word:
    #     print(f'{pos}: {len(pos_word[pos])}')


dataset_dir = "/home/Tanaka/generate-commentary/dataset/lol/annotation/deepsegment"
result_path = "./result.txt"


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


def analyze_vocab(dataset_dir, max_length=20):
    # build vocab and tokenized sentences
    #  stop_words=set(stopwords.words("english")),
    text_proc = torchtext.data.Field(sequential=True, init_token='<init>',
                                     eos_token='<eos>', tokenize=tokenize,
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
    from IPython import embed
    embed()
    exit()


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
    # tokens = []
    # for token in doc:
    #     tokens.append(token.lemma_)
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
