import os
import pandas as pd

def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    import json
    json.dump(data, open(path, "w"), indent=2)

def append_csv(row, path="results/logs.csv"):
    import pandas as pd
    df = pd.DataFrame([row])
    write_header = not os.path.exists(path)
    df.to_csv(path, mode="a", header=write_header, index=False)
