import sys
import os
import re
import string
import nltk
from nltk import FreqDist
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import logging

logging.getLogger('nltk').setLevel(logging.ERROR)

nltk.download('stopwords')
nltk.download('punkt')


def extract_tag_name(string):
    parts = string.replace(".txt", "").split('_')
    return parts[1]


def load_freq_unigram(file):
    with open(file, 'r') as file:
        lines = file.readlines()
        word_freq_list = [line.strip().split() for line in lines]
        freq_dist = FreqDist()
        for word, freq in word_freq_list:
            freq_dist += FreqDist([word] * int(freq))
        
        return freq_dist

def load_freq_bigram(file):
    with open(file, 'r') as file:
        lines = file.readlines()
    word_freq_list = [line.strip().split() for line in lines]
    freq_dist = FreqDist()
    for word1, word2, freq in word_freq_list:
        freq_dist += FreqDist([(word1, word2)] * int(freq))

    return freq_dist

def load_unigrams(path):
    all_files = os.listdir(path)
    
    result = {}
    filtered_files = [file for file in all_files if file.startswith('unigrams_')]

    for file in filtered_files:
        tag = extract_tag_name(file)
        result[tag] = load_freq_unigram(f"{path}/{file}")
    
    return result
        

def load_bigrams(path):
    all_files = os.listdir(path)

    result = {}
    filtered_files = [file for file in all_files if file.startswith('bigrams_')]
    
    for file in filtered_files:
        tag = extract_tag_name(file)
        result[tag] = load_freq_bigram(f"{path}/{file}")
    
    return result

def unigram_tag(tag_unigrams, text):
    unigrams = word_tokenize(text) 
    prob = {}

    for tag, freq_dist in tag_unigrams.items():
        base_prop = 1.0
        total_unigrams = sum(freq_dist.values())

        for unigram in unigrams:
            base_prop *= (freq_dist[unigram] + 1) / (total_unigrams + len(freq_dist))

        prob[tag] = base_prop
 
    return max(prob, key=prob.get)


def bigram_tag(tag_digram_freqdist, text):
    words = word_tokenize(text)
    max_score = 0
    predicted_tag = None

    for tag, freqdist in tag_digram_freqdist.items():
        score = 0
        for word1, word2 in zip(words[:-1], words[1:]):
            digram = (word1, word2)
            score += freqdist.get(digram, 0)

        if score > max_score:
            max_score = score
            predicted_tag = tag

    return predicted_tag


def bigram_smooth_tag(tag_digram_freqdist, text, alpha=1):
    words = word_tokenize(text)
    vocabulary_size = len(set(word for freqdist in tag_digram_freqdist.values() for word in freqdist.keys()))
    max_score = float('-inf')
    predicted_tag = None

    for tag, freqdist in tag_digram_freqdist.items():
        score = 0
        total_digrams = sum(freqdist.values())
        for word1, word2 in zip(words[:-1], words[1:]):
            digram = (word1, word2)
            score += (freqdist.get(digram, 0) + alpha) / (total_digrams + alpha * vocabulary_size)

        if score > max_score:
            max_score = score
            predicted_tag = tag

    return predicted_tag


def load_questions(file):
    with open(file, 'r') as file:
        lines = file.readlines()
        lines = [line.replace("\t" , " ") for line in lines]
        return lines
        
def pre_process(text):
    remove_words = list(set(string.punctuation))
    remove_words.extend(stopwords.words())
    remove_words.extend(set("`'"))
    remove_words.extend(["``", "''"])

    words_question = word_tokenize(question)

    processed_tokens = []
    for token in words_question:
        ntoken = token.lower()
        if ntoken in remove_words:
            continue
        if re.match(r'^\d{4}$', ntoken):
            processed_tokens.append('_YEAR_')
        else:
            processed_tokens.append(ntoken)
    return ' '.join(processed_tokens)


if len(sys.argv) < 4:
    print("Uso: python lmclassifier.py algorithm folder_unigrams_bigrams file_questions.txt")
    exit(1)

algorithm = sys.argv[1]
folder_counts = sys.argv[2]
file = sys.argv[3]
is_preprocess = False
if folder_counts == "counts2":
    is_preprocess = True

questions = load_questions(file)

if algorithm == "unigram":
    unigrams = load_unigrams(folder_counts)
    for question in questions:
        if is_preprocess:
            question = pre_process(question)
        print(unigram_tag(unigrams, question))
elif algorithm == "bigram":
    bigrams = load_bigrams(folder_counts)
    for question in questions:
        if is_preprocess:
            question = pre_process(question)
        print(bigram_tag(bigrams, question))
elif algorithm == "smooth":
    bigrams = load_bigrams(folder_counts)
    for question in questions:
        if is_preprocess:
            question = pre_process(question)
        print(bigram_smooth_tag(bigrams, question))
else:
    print("Invalid algorithm")
    exit(1)
