import click
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score

import joblib


from src.data.make_dataset import make_dataset
from src.features.make_features import make_features
from src.model.random_forest_classifier import make_model


@click.group()
def cli():
    pass


@click.command()
@click.option("--task", help="Can be is_comic_video, is_name or find_comic_name")
@click.option("--input_filename", default="data/raw/train.csv", help="File training data")
@click.option("--model_dump_filename", default="models/dump.json", help="File to dump model")
def train(task, input_filename, model_dump_filename):
    df = make_dataset(input_filename)
    X, y = make_features(df, task)

    model = make_model()
    model.fit(X, y)

    # Dumping the model to the specified filename
    joblib.dump(model, model_dump_filename)
    return "Model saved to {}".format(model_dump_filename)



@click.command()
@click.option("--task", help="Can be is_comic_video, is_name or find_comic_name")
@click.option("--input_filename", default="data/raw/train.csv", help="File training data")
@click.option("--model_dump_filename", default="models/dump.json", help="File to dump model")
@click.option("--output_filename", default="data/processed/prediction.csv", help="Output file for predictions")
def predict(task, input_filename, model_dump_filename, output_filename):
    # 1. Load the trained model
    model = joblib.load(model_dump_filename)
    df = make_dataset(input_filename)
    X, _ = make_features(df, task)

    # 3. Use the model to make predictions
    predictions = model.predict(X)

    # Save predictions to the output file
    output_df = pd.DataFrame({"prediction": predictions})
    output_df.to_csv(output_filename, index=False)

    print(f"Predictions saved to {output_filename}")

@click.command()
@click.option("--task", help="Can be is_comic_video, is_name or find_comic_name")
@click.option("--input_filename", default="data/raw/train.csv", help="File training data")
def evaluate(task, input_filename):
    # Read CSV
    df = make_dataset(input_filename)

    # Make features (tokenization, lowercase, stopwords, stemming...)
    X, y = make_features(df, task)

    # Object with .fit, .predict methods
    model = make_model()

    # Run k-fold cross validation. Print results
    return evaluate_model(model, X, y)


def evaluate_model(model, X, y):
    # Scikit learn has function for cross validation
    scores = cross_val_score(model, X, y, scoring="accuracy")

    print(f"Got accuracy {100 * np.mean(scores)}%")

    return scores


cli.add_command(train)
cli.add_command(predict)
cli.add_command(evaluate)

if __name__ == "__main__":
    cli()
