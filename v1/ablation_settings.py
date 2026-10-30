# 文件功能：定义并应用候选、奖励、特征和 DQN 等消融实验配置。
# 主要内容：AblationVariant、variant_names、apply_ablation_variant。
# 模块上游（导入关系）：v1.evaluate_v1、v1.train_v1、tests.v1.test_ablation_settings_v1。
# 模块下游（项目依赖）：shared.config。
# 调用说明：函数上游仅列可静态定位的直接调用，下游保留源码调用表达式。
# 动态策略、回调及工厂注入需结合入口与运行时装配代码阅读。

"""Small, ordered ablation registry shared by training and evaluation."""

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Mapping

from shared import config


@dataclass(frozen=True)
class AblationVariant:
    group: str
    overrides: Mapping[str, object]
    requires_retraining: bool = True


# Keep groups in the requested execution order.  The reference configuration
# remains in shared.config; variants only declare the field being ablated.
ABLATION_VARIANTS = {
    "candidate.complete": AblationVariant(
        "candidate", {"V1_CANDIDATE_MODE": "complete"}
    ),
    "candidate.layered_pool": AblationVariant(
        "candidate", {"V1_CANDIDATE_MODE": "layered_pool"}
    ),
    "objective.cost_only": AblationVariant(
        "objective",
        {
            "V1_OBJECTIVE_COST_WEIGHT": 1.0,
            "V1_OBJECTIVE_GREEN_WEIGHT": 0.0,
            "V1_OBJECTIVE_BALANCE_WEIGHT": 0.0,
        },
    ),
    "objective.green_only": AblationVariant(
        "objective",
        {
            "V1_OBJECTIVE_COST_WEIGHT": 0.0,
            "V1_OBJECTIVE_GREEN_WEIGHT": 1.0,
            "V1_OBJECTIVE_BALANCE_WEIGHT": 0.0,
        },
    ),
    "objective.equal_no_balance": AblationVariant(
        "objective", {"V1_OBJECTIVE_BALANCE_WEIGHT": 0.0}
    ),
    "objective.full": AblationVariant("objective", {}),
    "wait.off": AblationVariant(
        "wait", {"V1_ACTIVE_WAIT_ENABLED": False}
    ),
    "wait.on": AblationVariant("wait", {}),
    "reward.estimate_only": AblationVariant(
        "reward",
        {
            "V1_REWARD_REALIZATION_CORRECTION_ENABLED": False,
            "V1_REWARD_TERMINAL_PENALTIES_ENABLED": False,
        },
    ),
    "reward.no_correction": AblationVariant(
        "reward", {"V1_REWARD_REALIZATION_CORRECTION_ENABLED": False}
    ),
    "reward.no_terminal_penalty": AblationVariant(
        "reward", {"V1_REWARD_TERMINAL_PENALTIES_ENABLED": False}
    ),
    "discount.none": AblationVariant(
        "reward", {"V1_DISCOUNT_MODE": "none"}
    ),
    "discount.decision_step": AblationVariant(
        "reward", {"V1_DISCOUNT_MODE": "decision_step"}
    ),
    "discount.physical_time": AblationVariant("reward", {}),
    "feature.no_cost": AblationVariant(
        "feature", {"V1_DISABLED_CANDIDATE_FEATURE_GROUPS": ("cost",)}
    ),
    "feature.no_green": AblationVariant(
        "feature", {"V1_DISABLED_CANDIDATE_FEATURE_GROUPS": ("green",)}
    ),
    "feature.no_sla": AblationVariant(
        "feature", {"V1_DISABLED_CANDIDATE_FEATURE_GROUPS": ("sla",)}
    ),
    "feature.no_load": AblationVariant(
        "feature", {"V1_DISABLED_CANDIDATE_FEATURE_GROUPS": ("load",)}
    ),
    "dqn.candidate_only": AblationVariant(
        "feature", {"V1_DQN_USE_GLOBAL_STATE": False}
    ),
    "dqn.vanilla": AblationVariant(
        "feature", {"V1_DQN_DOUBLE_DQN": False}
    ),
    "dqn.no_target_lag": AblationVariant(
        "feature", {"V1_TARGET_UPDATE_INTERVAL": 1}
    ),
    "price.fixed": AblationVariant(
        "price", {"V1_TARIFF_MODE": "fixed"}
    ),
    "price.tou_uniform": AblationVariant(
        "price", {"V1_TARIFF_MODE": "tou_uniform"}
    ),
    "price.tou_region": AblationVariant(
        "price", {"V1_TARIFF_MODE": "tou_region"}
    ),
    "price.green_subsidy": AblationVariant(
        "price", {"V1_TARIFF_MODE": "green_subsidy"}
    ),
    "price.carbon_tax": AblationVariant(
        "price", {"V1_TARIFF_MODE": "carbon_tax"}
    ),
    "price.full": AblationVariant(
        "price", {"V1_TARIFF_MODE": "full"}
    ),
}


# 作用：返回支持的消融变体名称。
# 上游（直接调用）：v1.evaluate_v1.main、v1.train_v1.main。
def variant_names():
    return tuple(ABLATION_VARIANTS)


# 作用：应用消融配置并返回变更信息。
# 上游（直接调用）：v1.evaluate_v1.run_evaluation、v1.train_v1.main。
@contextmanager
def apply_ablation_variant(name):
    if name is None:
        yield None
        return
    try:
        variant = ABLATION_VARIANTS[name]
    except KeyError as error:
        raise ValueError(f"unknown ablation variant: {name}") from error
    overrides = dict(variant.overrides)
    overrides["V1_ABLATION_VARIANT"] = name
    if "V1_CANDIDATE_MODE" in overrides:
        overrides["CANDIDATE_MODE"] = overrides["V1_CANDIDATE_MODE"]
    original = {key: getattr(config, key) for key in overrides}
    try:
        for key, value in overrides.items():
            setattr(config, key, value)
        yield variant
    finally:
        for key, value in original.items():
            setattr(config, key, value)
