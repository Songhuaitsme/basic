# 文件功能：定义评估报告结构，统一转换为基础类型及规范 JSON。
# 主要内容：FormalSchemaError、to_primitive、to_canonical_json。
# 模块上游（导入关系）：v1.evaluation_v1、tests.v1.test_v1_acceptance。
# 调用说明：函数上游仅列可静态定位的直接调用，下游保留源码调用表达式。
# 动态策略、回调及工厂注入需结合入口与运行时装配代码阅读。

"""Finite-value validation and canonical formal-result serialization."""

from dataclasses import asdict, is_dataclass
from enum import Enum
import json
import math


class FormalSchemaError(ValueError):
    pass


# 作用：递归把报告对象转换为 JSON 基础类型。
# 上游（直接调用）：v1.evaluation_v1.schema.to_canonical_json。
# 下游调用：to_primitive、FormalSchemaError。
def to_primitive(value, path="root"):
    if is_dataclass(value):
        return to_primitive(asdict(value), path)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {
            str(key.value if isinstance(key, Enum) else key): to_primitive(
                item, f"{path}.{key}"
            )
            for key, item in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [to_primitive(item, f"{path}[{index}]") for index, item in enumerate(value)]
    if isinstance(value, float) and not math.isfinite(value):
        raise FormalSchemaError(f"{path}: formal numeric value must be finite")
    return value


# 作用：按规范字段顺序输出 JSON。
# 上游（直接调用）：tests.v1.test_v1_acceptance.V1AcceptanceContractTest.test_not_applicable_wait_relative_change_and_finite_json。
# 下游调用：to_primitive。
def to_canonical_json(value) -> str:
    return json.dumps(
        to_primitive(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
