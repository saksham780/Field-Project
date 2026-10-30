import os
import urllib.request

url = "https://raw.githubusercontent.com/W-ayang/Early-stage-diabetes-risk-prediction/main/diabetes_data_upload.csv"
os.makedirs("data", exist_ok=True)
try:
    req = urllib.request.Request(url, headers={"User-Agent": "DiaPredict Student Project"})
    with urllib.request.urlopen(req, timeout=30) as response:
        content = response.read()
    if len(content) < 1000 or b"Age" not in content[:500]:
        raise ValueError("Unexpected file downloaded")
    with open("data/diabetes_data_upload.csv", "wb") as f:
        f.write(content)
    print("Dataset saved to data/diabetes_data_upload.csv")
except Exception as e:
    print("Download failed:", e)
    print("Download manually:", url)
    print("Save as data/diabetes_data_upload.csv")
