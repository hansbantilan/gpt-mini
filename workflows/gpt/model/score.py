"""
Run inference on a scoring dataset with a GPT model.
"""

import argparse
from datetime import date

from gpt_mini.modeling.gpt_factory import GptFactory
from gpt_mini.utility import logger

log = logger.init("score")

curr_date = date.today().strftime("%Y%m%d")


#############################################
# scoring
#############################################
def main(
    data_source: str,
    model_type: str,
    model_version: str,
    model_config: str,
    disable_gpu: bool,
) -> None:
    gpt = GptFactory(
        data_source=data_source,
        model_type=model_type,
        model_version=model_version,
        model_config=model_config,
        disable_gpu=disable_gpu,
    )
    gpt.score()
    log.info(f"Successfully scored dataset using the {model_type} GPT model!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference using a GPT model")
    parser.add_argument(
        "--dataSource",
        default="local",
        action="store",
        dest="data_source",
        choices=["shakespeare", "local"],
        help="one of {shakespeare, local}",
    )
    parser.add_argument(
        "--modelType",
        default="tensorflow_char",
        action="store",
        dest="model_type",
        choices=["tensorflow_char"],
        help="one of {tensorflow_char, ...}",
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
