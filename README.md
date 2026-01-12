# Minimalist example of model integration with CHAP (uv version)

This document demonstrates a minimalist example of how to write a CHAP-compatible forecasting model using modern Python tooling. The example uses [uv](https://docs.astral.sh/uv/) for dependency management and [cyclopts](https://cyclopts.readthedocs.io/) for command-line argument parsing.

The model simply learns a linear regression from rainfall and temperature to disease cases in the same month, without considering any previous disease or climate data. It also assumes and works only with a single region. The model is not meant to accurately capture any interesting relations - the purpose is just to show how CHAP integration works in a simplest possible setting.

## Requirements

Before running this example, you need to have [uv](https://docs.astral.sh/uv/) installed on your system. If you don't have it, install it with:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

No other setup is needed - `uv run` will automatically create the virtual environment and install dependencies on first use.

## Running the model without CHAP integration

Before getting a new model to work as part of CHAP, it can be useful to develop and debug it while running it directly on a small dataset from file.

### Training data

Create a minimalist input file containing rainfall, temperature and disease cases for a single region and two time points (`input/trainData.csv`):

```csv
time_period,rainfall,mean_temperature,disease_cases,location
2023-05,10,30,200,loc1
2023-06,2,30,100,loc1
```

### Training the model

The file `main.py` contains the code to train a model. Run it with:

```bash
uv run python main.py train input/trainData.csv output/model.pkl
```

The train command reads training data from a CSV file into a Pandas dataframe. It learns a linear regression from `rainfall` and `mean_temperature` (X) to `disease_cases` (Y). The trained model is stored to file using the joblib library:

```python
@app.command()
def train(train_data: str, model: str):
    df = pd.read_csv(train_data)
    features = df[["rainfall", "mean_temperature"]].fillna(0)
    target = df["disease_cases"].fillna(0)

    reg = LinearRegression()
    reg.fit(features, target)
    joblib.dump(reg, model)
```

### Future climate data

Create a minimalist future (predicted) climate data file (`input/futureClimateData.csv`). This file contains climate data for what is considered to be future periods (weather forecasts). It naturally contains no disease data:

```csv
time_period,rainfall,mean_temperature,location
2023-07,20,20,loc1
2023-08,30,20,loc1
2023-09,30,30,loc1
```

### Generating forecasts

Run the predict command to forecast disease cases based on future climate data and the previously trained model:

```bash
uv run python main.py predict output/model.pkl input/trainData.csv input/futureClimateData.csv output/predictions.csv
```

The predict command loads the trained model and applies it to future climate data, outputting disease forecasts as a CSV file:

```python
@app.command()
def predict(model: str, historic_data: str, future_data: str, out_file: str):
    reg = joblib.load(model)
    future_df = pd.read_csv(future_data)
    features = future_df[["rainfall", "mean_temperature"]].fillna(0)

    predictions = reg.predict(features)
    output_df = future_df[["time_period", "location"]].copy()
    output_df["sample_0"] = predictions
    output_df.to_csv(out_file, index=False)
```

## Running the minimalist model as part of CHAP

To run the minimalist model in CHAP, we define the model interface in a YAML specification file called `MLproject`. This file specifies that the model uses uv for environment management (`uv_env: pyproject.toml`) and defines the train and predict entry points:

```yaml
name: minimalist_example_uv

uv_env: pyproject.toml

entry_points:
  train:
    parameters:
      train_data: str
      model: str
    command: "python main.py train {train_data} {model}"
  predict:
    parameters:
      historic_data: str
      future_data: str
      model: str
      out_file: str
    command: "python main.py predict {model} {historic_data} {future_data} {out_file}"
```

After you have installed chap-core ([installation instructions](https://dhis2-chap.github.io/chap-core/chap-cli/chap-core-cli-setup.html)), you can run this minimalist model through CHAP as follows:

```bash
chap evaluate --model-name /path/to/minimalist_example_uv --dataset-name ISIMIP_dengue_harmonized --dataset-country brazil --report-filename report.pdf
```

Or if you have a local CSV dataset:

```bash
chap evaluate --model-name /path/to/minimalist_example_uv --dataset-csv your_data.csv --report-filename report.pdf
```

## Creating your own model

You can use this example as a starting point for your own model. The key files are:

- `MLproject` - Defines how CHAP interacts with your model
- `pyproject.toml` - Lists your Python dependencies
- `main.py` - Contains your model's train and predict logic

To create a new model from scratch, you can use the `chap init` command:

```bash
chap init my_new_model
cd my_new_model
python isolated_run.py  # Test the model
```
