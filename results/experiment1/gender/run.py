import pandas as pd
from llm.utils import get_llms
from llm.experiment1 import experiment1, process_llms_and_df
from config.settings import TEMPERATURE

def run_experiment():
    llms = get_llms()
    df = pd.read_csv('path/to/your/data.csv')
    results = process_llms_and_df(llms, df)
    results.to_csv('path/to/output/results.csv', index=False)

if __name__ == "__main__":
    run_experiment()