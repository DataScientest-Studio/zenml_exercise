import click
from pipelines import feature_engineering, training, inference

@click.command()
@click.option(
    "--feature-pipeline",
    is_flag=True,
    default=False,
    help="Whether to run the pipeline that creates the dataset.",
)
@click.option(
    "--training-pipeline",
    is_flag=True,
    default=False,
    help="Whether to run the pipeline that trains the model.",
)
@click.option(
    "--inference-pipeline",
    is_flag=True,
    default=False,
    help="Whether to run the pipeline that performs inference.",
)

def main(
    feature_pipeline: bool = False,
    training_pipeline: bool = False,
    inference_pipeline: bool = False,
    ):

    if feature_pipeline:
        print("Starting the feature engineering pipeline...")
        feature_engineering()

    if training_pipeline:
        print("Starting the training pipeline...")
        training()

    if inference_pipeline:
        print("Starting the inference pipeline...")
        inference()


if __name__ == "__main__":
    main()

