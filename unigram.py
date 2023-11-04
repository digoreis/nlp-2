import nltk
import sys
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist

if len(sys.argv) < 3:
    print("Uso: python bigram.py file folder_out")
    exit(1)

    # Obtém os argumentos da linha de comando
file_input = sys.argv[1]
folder_output = sys.argv[2]

unigram_tags= {}
tag_question = {}
# Abra o arquivo para leitura
with open(file_input, 'r') as file:
    lines = file.readlines()

for line in lines:
    splited_line = line.strip().split('\t')
    if len(splited_line) != 3:
        continue
    tag, question, answer = line.strip().split('\t')
    if tag in tag_question:
        tag_question[tag].append(f"{question} {answer}")
    else:
        tag_question[tag] = [f"{question} {answer}"]

for tag, questions in tag_question.items():
    unigramas = []
    for question in questions:
        tokens = word_tokenize(question)  # Tokenize a pergunta em palavras
        unigramas.extend(tokens)  # Adicione as palavras à lista de unigramas
    unigram_tags[tag] = unigramas
    
freq_unigrams = {}

for tag, unigrams in unigram_tags.items():
    fdist = FreqDist(unigrams)  # Calcule a frequência dos unigramas
    freq_unigrams[tag] = fdist

for tag, freq_dist in freq_unigrams.items():
    # Nome do arquivo no formato unigrams_CATEGORIA.txt
    file_name = f'{folder_output}/unigrams_{tag}.txt'

    # Abre o arquivo no modo de escrita
    with open(file_name, 'w') as file:
        # Escreve os unigramas e suas frequências no arquivo
        for unigram, freq in freq_dist.items():
            line = f'{unigram} {freq}\n'
            file.write(line)