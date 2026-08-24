# New York Hospital Charges Analysis

Exploratory data analysis and regression modeling for hospital charges in New
York discharge records from 2009. The project includes a Jupyter notebook and
a Streamlit interface for exploring the data and model outputs.

## Scope

- Clean and explore hospital discharge records.
- Engineer features related to length of stay, urgency, and age group.
- Compare regression models using R-squared, RMSE, and MAE.
- Provide an interactive interface for inspecting model predictions.

## Repository contents

```text
NY_Hospital_Charges_Analysis.ipynb  Analysis and model experiments
gui_app.py                           Streamlit interface
NY Hospital Admissions - Dataset.csv Dataset tracked with Git LFS
```

## Run locally

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
streamlit run gui_app.py
```

On macOS or Linux, activate the environment with `source .venv/bin/activate`.

## Notes

The dataset is historical and the application is intended for exploratory
analysis only. It is not a clinical or financial decision-support tool.

Before reusing the data, review its source and license requirements.
