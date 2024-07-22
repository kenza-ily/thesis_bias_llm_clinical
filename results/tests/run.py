import pandas as pd
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from llm.utils import get_llms
from llm.experiment1.ipynb import process_llms_and_df
from config.settings import TEMPERATURE

def run_experiment():
    llms = get_llms()
    df = pd.read_csv(project_root / 'data' / 'your_data_file.csv')
    results = process_llms_and_df(llms, df)
    results.to_csv(project_root / 'results' / 'experiment1_gender' / 'results.csv', index=False)

if __name__ == "__main__":
    run_experiment()