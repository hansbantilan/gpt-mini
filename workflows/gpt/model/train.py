"""
Train a GPT model on a training dataset.
"""
import argparse
from datetime import date

from gpt_mini.modeling.gpt_factory import GptFactory
from gpt_mini.utility import logger, well_known_paths

log = logger.init("train")

curr_date = date.today().strftime("%Y%m%d")


#############################################
# training
#############################################
def main(
    model_type: str,
    model_version: str,
    model_config: str,
    data_source: str,
    disable_gpu: bool,
) -> None:
    f"{well_known_paths['PARAMS_DIR']}/{model_type}/{model_config}.yaml"

    log.info("Initialising gpt...")
    gpt = GptFactory(
        model_type=model_type,
        model_version=model_version,
        model_config=model_config,
        data_source=data_source,
        disable_gpu=disable_gpu,
    )
    gpt.train()
    log.info(f"Successfully trained a {model_type} GPT model!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a GPT model")
    parser.add_argument(
        "--modelType",
        default="tensorflow_char",
        action="store",
        dest="model_type",
        choices=["tensorflow_char"],
        help="one of {tensorflow_char,...}",
    )
    parser.add_argument(
        "--modelVersion",
        action="store",
        default=f"{curr_date}",
        dest="model_version",
        help="model version prepended to the path",
    )
    parser.add_argument(
        "--modelConfig",
        default="default",
        action="store",
        dest="model_config",
        help="filename of model params (e.g. default for default.yaml).",
    )
    parser.add_argument(
        "--dataSource",
        default="local",
        action="store",
        dest="data_source",
        choices=["shakespeare", "local"],
        help="one of {shakespeare, local}",
    )
    parser.add_argument(
        "--disableGPU",
        dest="disable_gpu",
        action="store_true",
        default=False,
        help="if flag is passed in, will not use gpu",
    )

    args = parser.parse_args()
    print(vars(args))

    for k, v in vars(args).items():
        exec(f"{k} = '{v}'")
    main(**vars(args))
