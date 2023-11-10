import os
import argparse
from yacs.config import CfgNode

from demo import Demo
from utils import utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config_file", type=str, default="config/config.yaml", help="Path to config file"
    )
    parser.add_argument(
        "-o", "--output_file", type=str, default="messages.json", help="Path to output file"
    )
    parser.add_argument(
        "-l", "--log_file", type=str, default="log.log", help="Path to log file"
    )
    parser.add_argument(
        "-n", "--log_name", type=str, default=str(os.getpid()), help="Name of logger"
    )
    parser.add_argument(
        "-p",
        "--play_role",
        type=bool,
        default=False,
        help="Add a user controllable role",
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    logger = utils.set_logger(args.log_file, args.log_name)
    logger.info(f"os.getpid()={os.getpid()}")

    # create config
    config = CfgNode(new_allowed=True)
    config = utils.add_variable_to_config(config, "log_name", args.log_name)
    config = utils.add_variable_to_config(config, "play_role", args.play_role)
    config.merge_from_file(args.config_file)
    config["execution_mode"] = "serial"
    logger.info(f"\n{config}")
    os.environ["OPENAI_API_KEY"] = config["api_keys"][0]
    demo = Demo(config, logger)
    demo.launch_demo()
