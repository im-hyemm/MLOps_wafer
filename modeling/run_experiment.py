"""Notebook과 CLI가 공유하는 11회 단계별 실험 실행기."""

import argparse
import hashlib
import json
import logging
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from .config import (
    DEFAULT_ARTIFACT_ROOT,
    DEFAULT_DATA_PATH,
    ExperimentConfig,
    code_version,
    experiment_identity,
    read_json,
    write_json,
)
from .data import create_or_load_splits, load_dataset
from .evaluation import (
    evaluate_checkpoint,
    load_experiment_results,
    select_winner,
)
from .training import configure_logging, train_experiment

STAGES = ("models", "recipes", "preprocessing", "final")


def _winner(configs, splits, root):
    identifiers = [
        experiment_identity(config, splits.split_id)[0] for config in configs
    ]
    results = load_experiment_results(root)
    if results.empty or not set(identifiers).issubset(
        set(results.experiment_id)
    ):
        raise ValueError("앞 단계의 실험을 먼저 완료해야 합니다.")
    row = select_winner(results[results.experiment_id.isin(identifiers)])
    return configs[identifiers.index(row.experiment_id)]


def stage_configs(stage, splits, artifact_root, base_config=None):
    """완료된 이전 단계의 우승 결과로 현재 단계 후보를 구성한다.

    Args:
        stage: models, recipes, preprocessing 또는 final.
        splits: 공통 고정 데이터 분할.
        artifact_root: 완료 실험의 저장 루트.
        base_config: epoch·batch size 등 공통 설정.

    Returns:
        현재 단계에서 실행할 ExperimentConfig 목록.

    Raises:
        ValueError: 이전 단계가 완료되지 않았거나 지원하지 않는 단계.
    """
    if stage not in STAGES:
        raise ValueError(f"지원하지 않는 단계: {stage}")
    base = replace(
        base_config or ExperimentConfig(),
        seed=42,
        preprocessing="fixed_resize",
        loss="weighted_ce",
        augmentation=False,
    )
    models = [
        replace(base, model=name)
        for name in ("small_cnn", "spatial_cnn", "residual_cnn", "hybrid_cnn")
    ]
    if stage == "models":
        return models
    model_winner = _winner(models, splits, artifact_root)
    recipes = [
        replace(model_winner, loss=loss, augmentation=augmentation)
        for loss in ("ce", "weighted_ce")
        for augmentation in (False, True)
    ]
    if stage == "recipes":
        return recipes
    recipe_winner = _winner(recipes, splits, artifact_root)
    preprocessing = [
        replace(recipe_winner, preprocessing=mode)
        for mode in ("fixed_resize", "resize_pad", "resize_pad_mask")
    ]
    if stage == "preprocessing":
        return preprocessing
    final = _winner(preprocessing, splits, artifact_root)
    return [replace(final, seed=seed) for seed in (42, 43, 44)]


def run_stage(
    stage,
    splits,
    artifact_root=DEFAULT_ARTIFACT_ROOT,
    base_config=None,
    device=None,
):
    """Notebook에서 한 단계 또는 전체 실험을 실행한다.

    Args:
        stage: models, recipes, preprocessing, final 또는 all.
        splits: 고정 80/10/10 데이터 분할.
        artifact_root: 결과를 저장할 디렉터리.
        base_config: 기본 30 epoch를 포함하는 공통 학습 설정.
        device: 선택적 CPU/CUDA 장치.

    Returns:
        마지막으로 실행한 단계의 실험 디렉터리 목록.
    """
    root = Path(artifact_root)
    base = base_config or ExperimentConfig()
    signature = {
        "base": asdict(base),
        "split_id": splits.split_id,
        "code_version": code_version(),
    }
    suite_id = hashlib.sha256(
        json.dumps(signature, sort_keys=True).encode()
    ).hexdigest()[:12]
    suite_dir = root / "suites" / suite_id
    suite_path = suite_dir / "status.json"
    suite = (
        read_json(suite_path)
        if suite_path.exists()
        else {**signature, "completed_ids": [], "total": 11}
    )
    stages = STAGES if stage == "all" else (stage,)
    paths = []
    for current in stages:
        configs = stage_configs(current, splits, root, base)
        paths = []
        suite.update(stage=current, state="running")
        write_json(suite_path, suite)
        if current == "final":
            write_json(
                suite_dir / "final_selection.json",
                {
                    "config": asdict(configs[0]),
                    "selection_metric": "validation_macro_f1",
                    "split_id": splits.split_id,
                },
            )
        try:
            for index, config in enumerate(configs, 1):
                identifier = experiment_identity(config, splits.split_id)[0]
                logging.getLogger("modeling").info(
                    "단계=%s 후보=%d/%d 고유 학습 완료=%d/11 현재=%s",
                    current,
                    index,
                    len(configs),
                    len(suite["completed_ids"]),
                    identifier,
                )
                path = train_experiment(
                    config, splits, root, stage=current, device=device
                )
                paths.append(path)
                if identifier not in suite["completed_ids"]:
                    suite["completed_ids"].append(identifier)
                suite.update(
                    current_experiment=identifier,
                    completed=len(suite["completed_ids"]),
                )
                write_json(suite_path, suite)
                logging.getLogger("modeling").info(
                    "고유 학습 완료=%d/11", suite["completed"]
                )
            if current == "final":
                # test를 보기 전에 배포 후보 seed를 validation만으로 확정한다.
                winner = _winner(configs, splits, root)
                deploy_id = experiment_identity(winner, splits.split_id)[0]
                write_json(
                    suite_dir / "deployment_selection.json",
                    {"experiment_id": deploy_id, "seed": winner.seed},
                )
                scores = []
                for config, path in zip(configs, paths):
                    logging.getLogger("modeling").info(
                        "최종 test 평가 시작 seed=%d", config.seed
                    )
                    metrics = evaluate_checkpoint(
                        path, splits, split="test", device=device
                    )
                    scores.append(
                        {
                            "seed": config.seed,
                            "experiment_id": path.name,
                            **metrics,
                        }
                    )
                values = [row["macro_f1"] for row in scores]
                summary = {
                    "runs": scores,
                    "macro_f1_mean": float(np.mean(values)),
                    "macro_f1_std": float(np.std(values, ddof=1)),
                    "std_ddof": 1,
                    "deployment_experiment_id": deploy_id,
                }
                write_json(suite_dir / "final_summary.json", summary)
                logging.getLogger("modeling").info(
                    "최종 test Macro-F1 %.6f ± %.6f",
                    summary["macro_f1_mean"],
                    summary["macro_f1_std"],
                )
            suite.update(
                state="completed" if current == "final" else "stage_completed"
            )
            write_json(suite_path, suite)
        except (Exception, KeyboardInterrupt) as error:
            suite.update(
                state="interrupted"
                if isinstance(error, KeyboardInterrupt)
                else "failed",
                error=str(error),
            )
            write_json(suite_path, suite)
            raise
    return paths


def main():
    """CLI 인자를 읽어 단일 실험 또는 단계를 실행한다."""
    parser = argparse.ArgumentParser(description="웨이퍼 모델링 실험")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument(
        "--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT
    )
    parser.add_argument(
        "--stage", choices=("single", "all", *STAGES), default="single"
    )
    parser.add_argument("--model", default="small_cnn")
    parser.add_argument("--preprocessing", default="fixed_resize")
    parser.add_argument(
        "--loss", choices=("ce", "weighted_ce"), default="weighted_ce"
    )
    parser.add_argument("--augmentation", action="store_true")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--force", action="store_true", help="단일 실험을 처음부터 재학습"
    )
    args = parser.parse_args()
    if args.force and args.stage != "single":
        parser.error(
            "--force는 중복 학습 방지를 위해 single에서만 지원합니다."
        )
    configure_logging()
    splits = create_or_load_splits(
        load_dataset(args.data_path), args.artifact_root
    )
    config = ExperimentConfig(
        model=args.model,
        preprocessing=args.preprocessing,
        loss=args.loss,
        augmentation=args.augmentation,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    if args.stage == "single":
        train_experiment(
            config,
            splits,
            args.artifact_root,
            force=args.force,
            device=args.device,
        )
    else:
        run_stage(
            args.stage, splits, args.artifact_root, config, device=args.device
        )


if __name__ == "__main__":
    main()
