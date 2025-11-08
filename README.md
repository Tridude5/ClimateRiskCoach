# ClimateRiskCoach

ClimateRiskCoach is a project that looks at how climate and environmental data can be used to understand potential risks from climate change.
The main idea is to use data and Dynamic Bayesian Decision Networks to find trends, predict risk levels, and show results in a way that’s easy to read and interpret.

Python 3.11.9 required. 3.12 will lead to version mismatches with pgmpy. 


| Component / File                  | Description                                                                                   | Purpose                                            |
| --------------------------------- | --------------------------------------------------------------------------------------------- | -------------------------------------------------- |
| **data_sources/**                 | Contains input data. Includes `raw/` for original datasets and `processed/` for cleaned data. | Source of all model inputs.                        |
| **utils/get_data.py**             | Loads and manages data files.                                                                 | Fetches climate data for analysis.                 |
| **utils/cleaning.py**             | Cleans and formats the data.                                                                  | Ensures datasets are consistent and usable.        |
| **utils/plotting.py**             | Generates graphs and figures.                                                                 | Visualizes key patterns and results.               |
| **model/training_BN.py**          | Trains the Bayesian Network model.                                                            | Learns relationships between climate variables.    |
| **model/dynamic_bayesian_net.py** | Core logic of the dynamic Bayesian model.                                                     | Simulates evolving climate dependencies over time. |
| **model/discretize.py**           | Converts continuous data into categories.                                                     | Prepares data for Bayesian modeling.               |
| **model/regime.py**               | Detects shifts or “regimes” in climate patterns.                                              | Identifies structural changes or anomalies.        |
| **model/backtest.py**             | Tests model performance on historical data.                                                   | Evaluates model accuracy and reliability.          |
| **model/metrics.py**              | Calculates model evaluation metrics.                                                          | Measures predictive strength and error.            |
| **model/visualize_network.py**    | Draws the Bayesian network structure.                                                         | Helps interpret model connections visually.        |
| **artifacts/**                    | Stores results, saved models, and generated visual outputs.                                   | Keeps project outputs for later review.            |








| File / Folder               | Description                                                           | Purpose                                                      |
| --------------------------- | --------------------------------------------------------------------- | ------------------------------------------------------------ |
| **data_sources/raw/**       | Contains original, unprocessed climate and environmental data.        | Serves as the starting point for all analysis.               |
| **data_sources/processed/** | Stores cleaned and transformed datasets ready for modeling.           | Provides reliable, standardized data for the model.          |
| **utils/get_data.py**       | Loads data from files or online sources.                              | Automates data retrieval for the project.                    |
| **utils/cleaning.py**       | Cleans, filters, and formats raw data.                                | Ensures data quality and consistency.                        |
| **utils/plotting.py**       | Creates charts, graphs, and summary visuals.                          | Helps interpret model outputs and data trends.               |
| **Climate.ipynb**           | Jupyter Notebook for exploration and visual presentation.             | Lets users test, analyze, and explain results interactively. |
| **run_model.py**            | Main Python script that runs the entire pipeline.                     | Connects all parts (data → model → output) automatically.    |
| **artifacts/**              | Stores model outputs, generated figures, and saved results.           | Keeps reproducible project artifacts for review.             |
| **.gitignore**              | Excludes unnecessary files (like data or cache) from version control. | Keeps the project repository clean and efficient.            |
