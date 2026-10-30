# 文件功能：声明 v1.learning 包，汇集对外导出的类型与组件。
# 主要内容：导出或引用 CandidateDQNPolicy、CandidateDQNTrainer、CandidateDqnMetadata、CandidateSelectionTrace、SharedCandidateQNetwork、double_dqn_target、mask_q_values 等。
# 使用入口：包导入或外部调用方使用。
# 模块下游（项目依赖）：v1.learning.candidate_dqn、v1.learning.features、v1.learning.replay、v1.learning.reward。
# 调用说明：函数上游仅列可静态定位的直接调用，下游保留源码调用表达式。
# 动态策略、回调及工厂注入需结合入口与运行时装配代码阅读。

"""v1.0 variable-candidate learning and physical-time reward contracts."""

from .candidate_dqn import (
    CandidateDQNPolicy,
    CandidateDQNTrainer,
    CandidateDqnMetadata,
    CandidateSelectionTrace,
    SharedCandidateQNetwork,
    double_dqn_target,
    mask_q_values,
    select_candidate_id,
    validate_checkpoint_metadata,
)
from .features import (
    CANDIDATE_FEATURE_NAMES,
    CandidateFeatureConfig,
    CandidateFeatureEncoder,
)
from .replay import CandidateReplayBuffer, ReplayTransition, TimestampedReward
from .reward import DecisionRecord, EventRewardBuffer, GammaClock, RewardAssembler

__all__ = [
    "CANDIDATE_FEATURE_NAMES",
    "CandidateDQNPolicy",
    "CandidateDQNTrainer",
    "CandidateDqnMetadata",
    "CandidateSelectionTrace",
    "CandidateFeatureConfig",
    "CandidateFeatureEncoder",
    "CandidateReplayBuffer",
    "DecisionRecord",
    "EventRewardBuffer",
    "GammaClock",
    "ReplayTransition",
    "RewardAssembler",
    "SharedCandidateQNetwork",
    "TimestampedReward",
    "double_dqn_target",
    "mask_q_values",
    "select_candidate_id",
    "validate_checkpoint_metadata",
]
