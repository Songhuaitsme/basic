# 文件功能：复用 V1 冻结策略评估，提供 V2 版本及输出默认值。
# 主要内容：main。
# 使用入口：命令行执行，或由其他脚本导入后调用。
# 模块下游（项目依赖）：v1.evaluate_v1。
# 调用说明：函数上游仅列可静态定位的直接调用，下游保留源码调用表达式。
# 动态策略、回调及工厂注入需结合入口与运行时装配代码阅读。

"""V2 evaluation entry point with periodic audits and compact reports."""

from v1.evaluate_v1 import main as _evaluation_main


# 作用：解析或转发命令行参数，组织本文件执行流程。
# 调用约定：本文件命令行入口或上层版本入口调用。
# 下游调用：_evaluation_main。
def main():
    _evaluation_main(
        system_version="2.0",
        default_output="artifacts/v2/evaluation/report.json",
        default_audit_mode="periodic",
        default_report_mode="compact",
        default_candidate_chunk_size=65536,
    )


if __name__ == "__main__":
    main()
