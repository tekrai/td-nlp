import click
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score

import joblib
from sklearn.pipeline import Pipeline

from src.data.make_dataset import make_dataset
from src.features.make_features import make_features
from src.model import choose_model
from src.model.random_forest_classifier import make_model


@click.group()
def cli():
    pass


@click.command()
@click.option("--task", help="Can be is_comic_video, is_name or find_comic_name")
@click.option("--input_filename", default="data/raw/train1.csv", help="File training data")
@click.option("--model_dump_filename", default="models/dump.json", help="File to dump model")
def train(task, input_filename, model_dump_filename):
    df = make_dataset(input_filename)  # this never changes
    train_x, train_y = make_features(df, task)

    model = choose_model(task)

    model.fit(train_x, train_y)

    # Dumping the model to the specified filename
    joblib.dump(model, input_filename)
    return "Model saved to {}".format(model_dump_filename)


@click.command()
@click.option("--task", help="Can be is_comic_video, is_name or find_comic_name")
@click.option("--input_filename", default="data/raw/test.csv", help="File training data")
@click.option("--model_dump_filename", default="models/dump.json", help="File to dump model")
@click.option("--output_filename", default="data/output/prediction.csv", help="Output file for predictions")
def predict(task, input_filename, model_dump_filename, output_filename):
    # 1. Load the trained model
    model = joblib.load(model_dump_filename)

    df = make_dataset(input_filename)

    x, y = make_features(df, task)

    # 3. Use the model to make predictions
    predictions = model.predict(x)

    # Save predictions to the output file
    output_df = pd.DataFrame({"prediction": predictions, "result": y})
    output_df.to_csv(output_filename, index=False)

    print(f"Predictions saved to {output_filename}")


@click.command()
@click.option("--task", help="Can be is_comic_video, is_name or find_comic_name")
@click.option("--input_filename", default="data/raw/train1.csv", help="File training data")
def evaluate(task, input_filename):
    # Read CSV
    df = make_dataset(input_filename)

    # Make features (tokenization, lowercase, stopwords, stemming...)
    train_x, train_y = make_features(df, task)

    # Object with .fit, .predict methods
    model = choose_model(task)

    # Run k-fold cross validation. Print results
    return evaluate_model(model, train_x, train_y)


def evaluate_model(pipeline: Pipeline, train_x, train_y):
    # Scikit learn has function for cross validation
    scores = cross_val_score(pipeline, train_x, train_y, cv=5, scoring='accuracy', verbose=2)

    print(f"Got accuracy {100 * np.mean(scores)}%")

    return scores


cli.add_command(train)
cli.add_command(predict)
cli.add_command(evaluate)

if __name__ == "__main__":
    cli()
