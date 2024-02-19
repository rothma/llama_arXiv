import arxiv
import os
import pandas as pd

def get_papers(topic: str, max_results: int, file_name: str) -> pd.DataFrame:
    """
    Retrieves a specified number of papers from arXiv based on a given topic,
    saves the data to a CSV file, and returns a pandas DataFrame containing the extracted information.

    Args:
        topic (str): The topic to search for on arXiv.
        max_results (int): The maximum number of papers to retrieve.
        file_name (str): The name of the CSV file to save the data to.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the extracted information from the retrieved papers.
    """
    search = arxiv.Search(
        query = topic,
        max_results = max_results,
        sort_by = arxiv.SortCriterion.SubmittedDate,
        sort_order = arxiv.SortOrder.Descending)

    all_data = []
    for result in search.results():
        temp = ["","","","",""]
        temp[0] = result.title
        temp[1] = result.published
        temp[2] = result.entry_id
        temp[3] = result.summary
        temp[4] = result.pdf_url
        all_data.append(temp)
    
    column_names = ['Title','Date','Id','Summary','URL']
    df = pd.DataFrame(all_data, columns=column_names)
    
    print("Number of papers extracted : ", df.shape[0])
    df.head()

    path_name = 'data/'
    if os.path.exists(path_name):
        pass
    else:
        os.mkdir(path_name)

    for n, paper in enumerate(arxiv.Client().results(search)):
        try:
            print(paper.title)
            # Download the archive to a specified directory with a custom filename.
            paper.download_source(dirpath=path_name, filename=f"paper_{n}.tar.gz")
        except:
            pass
    
    df.to_csv(file_name)
    return df

if __name__ == "__main__":
    topic = ("Quantum Machine Learning")
    max_results = 5
    file_name = 'data/arxiv_data.csv'
    get_papers(topic, max_results, file_name)
