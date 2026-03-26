import argparse
import copy
import json
import logging
from datetime import datetime
from pathlib import Path

from src.config import vTrainConfig
from src.predictor import vTrain


DEFAULT_PARALLELISMS = [
    (8, 8, 35),
    (8, 10, 35),
    (8, 12, 35),
    (8, 12, 21),
    (8, 16, 21),
    (8, 20, 21),
]


logger = logging.getLogger("parallelism_experiment")
logging.basicConfig(
    format="[%(asctime)s] (%(levelname)s) %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


def parse_parallelism(spec):
    parts = [part.strip() for part in spec.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"parallelism must be formatted as T,D,P, got: {spec}"
        )

    try:
        triple = tuple(int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"parallelism must contain integers only, got: {spec}"
        ) from exc

    return triple


def build_config_dict(base_config, parallelism):
    tensor_parallel_size, data_parallel_size, pipeline_parallel_size = parallelism

    config_dict = copy.deepcopy(base_config)
    config_dict["tensor_parallel_size"] = tensor_parallel_size
    config_dict["data_parallel_size"] = data_parallel_size
    config_dict["pipeline_parallel_size"] = pipeline_parallel_size
    config_dict["num_gpus"] = (
        tensor_parallel_size * data_parallel_size * pipeline_parallel_size
    )

    return config_dict


def write_report(output_path, base_config_path, experiments):
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "base_config_path": str(base_config_path),
        "experiments": experiments,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(report, f, indent=2)


def run_experiment(base_config_path, output_path, parallelisms):
    with open(base_config_path, "r") as f:
        base_config = json.load(f)

    experiments = []

    for index, parallelism in enumerate(parallelisms, start=1):
        tensor_parallel_size, data_parallel_size, pipeline_parallel_size = parallelism
        logger.info(
            "experiment %d/%d with parallelism (T, D, P) = (%d, %d, %d)",
            index,
            len(parallelisms),
            tensor_parallel_size,
            data_parallel_size,
            pipeline_parallel_size,
        )

        config_dict = build_config_dict(base_config, parallelism)
        experiment = {
            "parallelism": {
                "tensor_parallel_size": tensor_parallel_size,
                "data_parallel_size": data_parallel_size,
                "pipeline_parallel_size": pipeline_parallel_size,
                "num_gpus": config_dict["num_gpus"],
            },
            "input_config": config_dict,
        }

        try:
            config = vTrainConfig(**config_dict)
            sim = vTrain(config)
            result, _ = sim()
            predicted_iteration_ms = max(result.values()) / 1000 / 1000

            experiment["predicted_iteration_ms"] = round(predicted_iteration_ms, 3)
            logger.info(
                "predicted iteration time: %.3f ms", predicted_iteration_ms
            )
        except Exception as exc:
            experiment["error"] = str(exc)
            logger.exception(
                "experiment failed for parallelism (T, D, P) = (%d, %d, %d)",
                tensor_parallel_size,
                data_parallel_size,
                pipeline_parallel_size,
            )

        experiments.append(experiment)
        write_report(output_path, base_config_path, experiments)

    logger.info("saved report to %s", output_path)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run vTrain for multiple parallelism settings and save the predicted "
            "iteration time to a JSON report."
        )
    )
    parser.add_argument(
        "-c",
        "--config",
        required=True,
        help="Path to the base JSON configuration file.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help=(
            "Path to the output JSON report. "
            "Default: results/<config-stem>_parallelism_experiment.json"
        ),
    )
    parser.add_argument(
        "-p",
        "--parallelism",
        action="append",
        type=parse_parallelism,
        dest="parallelisms",
        help=(
            "Parallelism tuple formatted as T,D,P. "
            "Repeat this option to provide multiple tuples."
        ),
    )
    args = parser.parse_args()

    base_config_path = Path(args.config)
    if args.output is None:
        output_path = Path("results") / (
            f"{base_config_path.stem}_parallelism_experiment.json"
        )
    else:
        output_path = Path(args.output)

    parallelisms = args.parallelisms or DEFAULT_PARALLELISMS
    run_experiment(base_config_path, output_path, parallelisms)


if __name__ == "__main__":
    main()
