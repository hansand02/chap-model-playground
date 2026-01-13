# Minimalist example of model integration with CHAP (uv version)

This document demonstrates a minimalist example of how to write a CHAP-compatible forecasting model using modern Python tooling. The example uses [uv](https://docs.astral.sh/uv/) for dependency management.

The model simply learns a linear regression from rainfall and temperature to disease cases in the same month, without considering any previous disease or climate data. It also assumes and works only with a single region. The model is not meant to accurately capture any interesting relations - the purpose is just to show how CHAP integration works in a simplest possible setting.

## Requirements

Before running this example, you need to have [uv](https://docs.astral.sh/uv/) installed on your system. If you don't have it, install it with:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```


## Repository structure

```
.
├── MLproject           # CHAP integration configuration
├── train.py            # Training logic
├── predict.py          # Prediction logic
├── pyproject.toml      # Python dependencies
├── isolated_run.py     # Script for testing without CHAP
├── input/              # Sample training and forecast data
└── output/             # Generated models and predictions
```

### Key files

- **MLproject**: Defines how CHAP interacts with your model (entry points, parameters)
- **train.py**: Contains the training logic
- **predict.py**: Contains the prediction logic
- **pyproject.toml**: Lists your Python dependencies - uv uses this to create the virtual environment
- **isolated_run.py**: Allows testing your model standalone, without CHAP

No other setup is needed - `uv run` will automatically create the virtual environment and install dependencies on first use.

## Running the model without CHAP integration

Before getting a new model to work as part of CHAP, it can be useful to develop and debug it while running it directly on a small dataset from file. Sample data files are included in the `input/` directory.

To quickly test everything works, run:

```bash
uv run python isolated_run.py
```

This will train the model and generate predictions using the sample data. After running, check the `output/` directory. You should see:

- `output/model.pkl` - the trained model
- `output/predictions.csv` - predicted disease cases

Open `predictions.csv` to see the forecasted values for each time period and location. 
The predictions will be in a column named `sample_0`. Note that a model may give multiple samples, 
in which case there will be additional columns named `sample_1`, `sample_2`, etc.


### Training the model

The train command is in the file `train.py` and reads training data from a CSV file into a Pandas dataframe. It learns a linear regression from `rainfall` and `mean_temperature` (X) to `disease_cases` (Y). The trained model is stored to file using the joblib library:

```python
def train(train_data_path, model_path):
    df = pd.read_csv(train_data_path)
    features = df[["rainfall", "mean_temperature"]].fillna(0)
    target = df["disease_cases"].fillna(0)

    model = LinearRegression()
    model.fit(features, target)
    joblib.dump(model, model_path)
```

### Generating forecasts

The predict function is in the file `predict.py` and generates forecasts of disease cases based on future climate data and the previously trained model:

```python
def predict(model_path, historic_data_path, future_data_path, out_file_path):
    model = joblib.load(model_path)
    future_df = pd.read_csv(future_data_path)
    features = future_df[["rainfall", "mean_temperature"]].fillna(0)

    predictions = model.predict(features)
    output_df = future_df[["time_period", "location"]].copy()
    output_df["sample_0"] = predictions
    output_df.to_csv(out_file_path, index=False)
```

## Making model alterations

We currently use a simple Linear Regression model from sklearn. Sklearn also supports other model types like Random Forest.
You can try make changes to train.py to experiment with using the random forest model. The class you will need to import is 
`RandomForestRegressor` from `sklearn.ensemble`.


### Add or remove features

You can also experiment with changing which features are used for training and prediction.

For instance to only use rainfall as a feature, modify the following lines in both `train.py` and `predict.py`:

```python
# Original uses rainfall and mean_temperature
features = df[["rainfall", "mean_temperature"]]

# Try using only rainfall
features = df[["rainfall"]]
```

### Test your changes

After making changes, always remember to run the isolated test to verify everything works.


Check that:
- The script runs without errors
- Output files are generated in the `output/` directory


## Running the minimalist model as part of CHAP

Running your model through CHAP gives you several benefits:

- You can easily run your model against standard evaluation datasets
- You can share your model with others in a standard way
- You can make your model accessible through the DHIS2 Modeling app

To run the minimalist model in CHAP, we define the model interface in a YAML specification file called `MLproject`. This file specifies that the model uses uv for environment management (`uv_env: pyproject.toml`) and defines the train and predict entry points:

```yaml
name: minimalist_example_uv

uv_env: pyproject.toml

entry_points:
  train:
    parameters:
      train_data: str
      model: str
    command: "python train.py {train_data} {model}"
  predict:
    parameters:
      historic_data: str
      future_data: str
      model: str
      out_file: str
    command: "python predict.py {model} {historic_data} {future_data} {out_file}"
```

After you have installed chap-core ([installation instructions](https://dhis2-chap.github.io/chap-core/chap-cli/chap-core-cli-setup.html)), you can run this minimalist model through CHAP as follows:

```bash
chap evaluate --model-name /path/to/minimalist_example_uv --dataset-name ISIMIP_dengue_harmonized --dataset-country brazil --report-filename report.pdf
```

**Parameters:**

- `--model-name`: Path to your local model directory (where MLproject is located)
- `--dataset-name`: The evaluation dataset to use
- `--dataset-country`: Country filter for the dataset
- `--report-filename`: Output PDF report

Or if you have a local CSV dataset:

```bash
chap evaluate --model-name /path/to/minimalist_example_uv --dataset-csv your_data.csv --report-filename report.pdf
```
