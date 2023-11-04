file_name_questions = f'./data/eval-questions.txt'
file_name_tags = f'./data/eval-labels.txt'
        
with open('./data/eval.txt', 'r') as file, open(file_name_questions, 'w') as file_question,open(file_name_tags, 'w') as file_tag:
    lines = file.readlines()
    for line in lines:
        splited_line = line.strip().split('\t')
        if len(splited_line) != 3:
            continue
        tag, question, answer = line.strip().split('\t')
        
        line_question = f'{question} {answer}\n'
        file_question.write(line_question)
        
        line_tag = f'{tag}\n'
        file_tag.write(line_tag)