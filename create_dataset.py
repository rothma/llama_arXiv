# This script creates a dataset from arXiv papers by extracting conclusions and summaries.
# It uses the Llama language model to generate questions based on the extracted information.

# Steps:
# 1. Set environment variables for CUDA, rank, world size, master address, master port, and local rank.
# 2. Set parameters such as the directory name, topic, and maximum number of results.
# 3. Download arXiv papers based on the specified topic and maximum results.
# 4. Read the downloaded papers and extract conclusions.
# 5. Create a pandas DataFrame with the extracted conclusions and summaries.
# 6. Remove rows with no conclusion and remove new line characters.
# 7. Generate dialogs for each row in the DataFrame.
# 8. Build the Llama language model.
# 9. Call the Llama model to generate questions based on the dialogs.
# 10. Create a dataset from the Llama output.
# 11. Save the dataset as a CSV file.

import glob
import os
import pandas as pd

from llama import Llama

from arxiv_downloader import get_papers
from dataset_helpers import create_dataset_from_llama_output, call_llama, get_file_content, extract_conclusion_from_content

# set environment variables
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
os.environ['RANK'] = '0'
os.environ['WORLD_SIZE'] = '1'
os.environ['MASTER_ADDR'] = '127.0.1'
os.environ['MASTER_PORT'] = '29500'
os.environ['LOCAL_RANK'] = '0'

# set parameters
directory_name = "data/"
topic = ("Quantum Machine Learning")
max_results = 5

# Llama parameters
ckpt_dir = '../llama/llama-2-7b-chat'
tokenizer_path ='../llama/tokenizer.model'
max_seq_len = 2048
max_batch_size = 8

# Start execution
arxiv_summary_filename = f'{directory_name}arxiv_data.csv'
get_papers(topic, max_results, arxiv_summary_filename)

file_list = sorted(glob.glob(directory_name + '*.tar.gz'), key=lambda x: int(x.split('_')[-1].split('.tar.gz')[0]))
df = pd.read_csv(directory_name + 'arxiv_data.csv')

conclusion_list = []
for n, file in enumerate(file_list):
    try:
        paper_content = get_file_content(file)
        conclusion = extract_conclusion_from_content(paper_content)
        conclusion_list.append(conclusion)
    except:
        df = df.drop(n)
        print(f'Skipping file {n} due to read error.')
print(f'Found {len(conclusion_list)} conclusions.')

dataset = pd.DataFrame({'Conclusion': conclusion_list, 'Summary': df['Summary']})
# remove rows with no conclusion
dataset = dataset.dropna()

# remove new line characters
dataset['Conclusion'] = dataset['Conclusion'].str.replace('\n', ' ')
dataset['Summary'] = dataset['Summary'].str.replace('\n', ' ')

print(dataset.head())
print(len(dataset))

dialogs = []
for i in range(len(dataset)):
    # dialogs.append([{"role": "user", "content": f"What are the key take away from the text enclosed in double angled brackets? <<{dataset.Summary.iloc[i]} {dataset.Conclusion.iloc[i]}>>"}])
    dialogs.append([{"role": "user", "content": f"What are the three scientific questions that are answered in the text enclosed in double angled brackets? Respond with both, the question directly followed by the answer. Before every question, put the Phrase QUESTION: and before every answer, put the phrase ANSWER:. Respond only with the questions and the answers and do not include introductions or confirmations <<{dataset.Summary.iloc[i]} {dataset.Conclusion.iloc[i]}>>"}])

generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

llama_output_list = []
for i in range(0, len(dialogs)//max_batch_size + 1):
    try:
        if (i+1)*max_batch_size > len(dialogs):
            llama_output = call_llama(dialogs[i*max_batch_size:], generator, temperature=0.6, top_p=0.9)
        else:
            llama_output = call_llama(dialogs[i*max_batch_size:(i+1)*max_batch_size], generator, temperature=0.6, top_p=0.9)
        llama_output_list.extend(llama_output)
        print(i*max_batch_size, (i+1)*max_batch_size)
    except AssertionError:
        continue

dataset = create_dataset_from_llama_output(llama_output_list, dialogs, llama_generate_questions=True)
if os.path.exists('data/extracted_data'):
    pass
else:
    os.mkdir('data/extracted_data')
dataset.to_csv('data/extracted_data/question_answer_dataset_from_arxiv.csv')
