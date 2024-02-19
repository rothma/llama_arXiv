import tarfile

from datasets import Dataset
from typing import Optional

def get_file_content(file_name):
    """
    Extracts the content of a .tar.gz file.

    Args:
        file_name (str): The path to the .tar.gz file.

    Returns:
        str: The joined content of the .tex files as a single string.
    """
    tex_content = []
    with tarfile.open(file_name, 'r:gz') as tar:
        # Iterate over each member
        for member in tar.getmembers():
            # Check if the member is a .tex file
            if member.name.endswith('.tex'):
                # Open the file (or similar object) for reading
                f = tar.extractfile(member)
                if f is not None:
                    content = f.read()
                    # Convert the file to a string
                    content = content.decode("utf-8")
                    tex_content.append(content)
    # Join the content of the .tex files as a single string
    joined_content = '\n'.join(tex_content)
    return joined_content

def extract_conclusion_from_content(content):
    """
    Extracts the conclusion section from the given content.

    Args:
        content (str): The content to extract the conclusion from.

    Returns:
        str: The extracted conclusion section.
        None: If no conclusion is found.
    """
    list_of_conclusion_synonyms = ['\section{conclusion}', '\section{summary}', '\section{outlook}',
                                   '\chapter{conclusion}', '\chapter{summary}', '\chapter{outlook}',
                                   '\subsection{conclusion}', '\subsection{summary}', '\subsection{outlook}']
    for conclusion_string in list_of_conclusion_synonyms:
        if conclusion_string in content.lower():
            # remove everything before the conclusion
            content = content[content.lower().find(conclusion_string):]
            # remove conclusion string
            content = content[content.lower().find('}') + 1:]
            # remove everything after the conclusion
            potential_closing_string = [content.lower().find(s) for s in ['\section', '\appendix', '\bibliograph', '\begin',
                                                                          '\\section', '\\appendix', '\\bibliograph', '\\begin']]
            filtered_list = [x for x in potential_closing_string if x != -1]
            if filtered_list:
                ending_index = min(filtered_list)
            else:
                ending_index = -1
            content = content[:ending_index]
            return content
    print('No conclusion found')
    return None

def call_llama(dialogs,
    generator,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_gen_len: Optional[int] = None,
    local_rank=None
):
    """
    Entry point of the program for generating text using a pretrained model.

    Args:
        dialogs (list): List of dialogues.
        generator: The generator object for text generation.
        temperature (float, optional): The temperature value for controlling randomness in generation.
            Defaults to 0.6.
        top_p (float, optional): The top-p sampling parameter for controlling diversity in generation.
            Defaults to 0.9.
        max_gen_len (int, optional): The maximum length of generated sequences. If None, it will be
            set to the model's max sequence length. Defaults to None.
        local_rank: The rank of the local process. Defaults to None.

    Returns:
        dict: The generated results.
    """
    results = generator.chat_completion(
        dialogs,  # type: ignore
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )

    return results

def create_dataset_from_llama_output(llama_output, dialogs, llama_generate_questions=False):
    """
    Creates a dataset from the output of the LLAMA model.

    Args:
        llama_output (list): List of outputs from the LLAMA model.
        dialogs (list): List of dialogues.
        llama_generate_questions (bool, optional): Whether to generate questions from the output.
            Defaults to False.

    Returns:
        Dataset: The created dataset.
    """
    data_list = []
    for dialog, result in zip(dialogs, llama_output):
        if llama_generate_questions:
            content = result['generation']['content'].strip()
            question_answer_pairs = content.split('QUESTION')
            for question_answer_pair in question_answer_pairs:
                if question_answer_pair == '':
                    continue
                try:
                    question, answer = question_answer_pair.split('ANSWER')
                    data_list.append(f'<s>[INST] {question.strip()} [/INST]' + answer.strip() + ' </s>')
                except ValueError:
                    print(question_answer_pair)
                    continue
        else:
            instruction = dialog[0]['content'].split(' <<')[0]
            content = result['generation']['content']
            data_list.append(f'<s>[INST] {instruction} [/INST]' + content + ' </s>')
    data = {'text': data_list}
    return Dataset.from_dict(data)
