import os

from gpt_mini.utility import logger

log = logger.init("utility_init_logger")

curr_dir = os.path.dirname(os.path.realpath(__file__))
_ROOT = os.path.join(curr_dir, "..", "..")
_MODULE_ROOT = os.path.join(curr_dir, "..")

WELL_KNOWN_PATHS = {
    "ROOT": _ROOT,
    "WORKFLOW_ROOT": os.path.join(_ROOT, "workflows"),
    "DATASETS_DIR": os.path.join(_ROOT, "datasets/"),
    "PARAMS_DIR": os.path.join(_ROOT, "gpt_mini/modeling/params/"),
}
