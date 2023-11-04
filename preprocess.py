import nltk
import string
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
nltk.download('punkt')

with open('data/train.txt', 'r') as file, open('data-processed/train.txt', 'w') as out:
    lines = file.readlines()

    for line in lines:
        splited_line = line.strip().split('\t')
        if len(splited_line) != 3:
            continue
        tag, question, answer = line.strip().split('\t')
        
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
        new_question = ' '.join(processed_tokens)
        
        out.write(f"{tag}\t{new_question}\t{answer.lower()}\n")
    
    
    