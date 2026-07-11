import argparse
import os
import pandas as pd

# DEFAULT_INPUT = "modellens_data/record_train.csv"
DEFAULT_INPUT = "modellens_data/question_answering_top100_datasets_models_over40.csv"
DEFAULT_OUTPUT_SUFFIX = "unique_datasets.csv"
PREFERRED_DATASET_COLUMNS = [
    "dataset",
    "dataset_name",
    "finetuned_dataset",
    "dataset_path",
]


def resolve_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(os.path.dirname(__file__), path))


def find_dataset_column(df: pd.DataFrame, explicit_column: str | None = None) -> str:
    if explicit_column:
        if explicit_column in df.columns:
            return explicit_column
        raise KeyError(f"Specified dataset column '{explicit_column}' not found in input CSV.")

    for column in PREFERRED_DATASET_COLUMNS:
        if column in df.columns:
            return column

    raise KeyError(
        "No dataset column found in input CSV. "
        f"Expected one of: {', '.join(PREFERRED_DATASET_COLUMNS)}. "
        f"Found columns: {', '.join(df.columns)}"
    )


def build_output_path(input_path: str, output_path: str | None) -> str:
    if output_path:
        return resolve_path(output_path)
    directory = os.path.dirname(input_path)
    return os.path.join(directory, DEFAULT_OUTPUT_SUFFIX)


def find_dataset_desp_column(df: pd.DataFrame) -> str | None:
    for column in ["dataset_desp", "dataset_desc", "description", "dataset_description"]:
        if column in df.columns:
            return column
    return None


def build_dataset_description_lookup(
    df: pd.DataFrame, dataset_column: str, dataset_desp_column: str | None
) -> dict[str, str]:
    if dataset_desp_column is None:
        return {}

    description_lookup: dict[str, str] = {}
    dataset_series = df[dataset_column].astype(str).fillna("").str.strip()
    description_series = df[dataset_desp_column].astype(str).fillna("").str.strip()

    for dataset_value, description_value in zip(dataset_series, description_series):
        if dataset_value == "":
            continue
        if dataset_value not in description_lookup or description_lookup[dataset_value] == "":
            description_lookup[dataset_value] = description_value

    return description_lookup


def load_unique_datasets(
    df: pd.DataFrame, dataset_column: str, dataset_desp_column: str | None
) -> pd.DataFrame:
    series = df[dataset_column].astype(str).fillna("").str.strip()
    series = series[series != ""]
    unique_values = sorted(series.unique(), key=lambda x: x.lower())
    description_lookup = build_dataset_description_lookup(df, dataset_column, dataset_desp_column)
    descriptions = [description_lookup.get(value, "") for value in unique_values]
    return pd.DataFrame({"dataset": unique_values, "dataset_desp": descriptions})


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract unique dataset names from a CSV file and save them to a new CSV."
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help="Input CSV path containing dataset records. Defaults to modellens_data/record_train.csv.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path for unique datasets. Defaults to '<input_dir>/unique_datasets.csv'.",
    )
    parser.add_argument(
        "--column",
        default=None,
        help="Dataset column name to extract. If omitted, tries common dataset column names.",
    )
    parser.add_argument(
        "--include-count",
        action="store_true",
        help="Include a count column showing how many times each dataset appears in the input file.",
    )
    args = parser.parse_args()

    input_path = resolve_path(args.input)
    output_path = build_output_path(input_path, args.output)

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)
    dataset_column = find_dataset_column(df, args.column)
    dataset_desp_column = find_dataset_desp_column(df)

    if args.include_count:
        series = df[dataset_column].astype(str).fillna("").str.strip()
        series = series[series != ""]
        result_df = (
            series.value_counts(dropna=True)
            .rename_axis("dataset")
            .reset_index(name="count")
            .sort_values(by=["count", "dataset"], ascending=[False, True])
            .reset_index(drop=True)
        )
        description_lookup = build_dataset_description_lookup(df, dataset_column, dataset_desp_column)
        result_df["dataset_desp"] = result_df["dataset"].map(description_lookup).fillna("")
    else:
        result_df = load_unique_datasets(df, dataset_column, dataset_desp_column)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result_df.to_csv(output_path, index=False)

    print(f"Input file:  {input_path}")
    print(f"Output file: {output_path}")
    print(f"Dataset column: {dataset_column}")
    print(f"Unique datasets: {len(result_df)}")


if __name__ == "__main__":
    main()
