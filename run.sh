python3 preprocess.py
python3 split.py
python3 unigram.py data/train.txt counts2 
python3 bigram.py data/train.txt counts2 
python3 unigram.py data-processed/train.txt counts2 
python3 bigram.py data-processed/train.txt counts2 
python3 lmclassifier.py unigram counts data/eval-questions.txt > results_unigram_counts.txt
python3 lmclassifier.py bigram counts data/eval-questions.txt > results_bigram_counts.txt
python3 lmclassifier.py smooth counts data/eval-questions.txt > results_smooth_counts.txt
python3 lmclassifier.py unigram counts2 data/eval-questions.txt > results_unigram_counts2.txt
python3 lmclassifier.py bigram counts2 data/eval-questions.txt > results_bigram_counts2.txt
python3 lmclassifier.py smooth counts2 data/eval-questions.txt > results_smooth_counts2.txt
python3 evaluate.py -v data/eval-labels.txt results_unigram_counts.txt
python3 evaluate.py -v data/eval-labels.txt results_bigram_counts.txt
python3 evaluate.py -v data/eval-labels.txt results_smooth_counts.txt
python3 evaluate.py -v data/eval-labels.txt results_unigram_counts2.txt
python3 evaluate.py -v data/eval-labels.txt results_bigram_counts2.txt
python3 evaluate.py -v data/eval-labels.txt results_smooth_counts2.txt