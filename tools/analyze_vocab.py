import os
import json
import nltk
from collections import defaultdict, Counter
# nltk.download('all')


def get_vocab_lol(dataset_dir, result_path, sampling_rate=100, max_length=20):
    dataset_files = [os.path.join(dataset_dir, x + '.json') for x in ['training', 'validation']]

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
get_vocab_lol(dataset_dir, result_path)

# from IPython import embed
# embed()
# exit()
