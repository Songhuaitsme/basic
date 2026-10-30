# 文件功能：记录已完成预留，待任务全部结清后生成能源和成本报告。
# 主要内容：MetricsLedger。
# 模块上游（导入关系）：v1.accounting、v1.v1_runtime、v3.v3_runtime、v6.v6_runtime、tests.v1.test_accounting_v1 等。
# 模块下游（项目依赖）：v1.domain.reservations、v1.accounting.energy。
# 调用说明：函数上游仅列可静态定位的直接调用，下游保留源码调用表达式。
# 动态策略、回调及工厂注入需结合入口与运行时装配代码阅读。

"""Append-only, idempotent realized metrics ledger."""

from typing import Dict

from v1.domain.reservations import Reservation

from .energy import AccountingReport, ExogenousEnergyAccounting


class MetricsLedger:
    # 作用：保存依赖与配置，建立实例初始状态。
    # 上游（直接调用）：v1.v1_runtime.create_v1_runtime、v3.v3_runtime.create_v3_runtime、v6.v6_runtime.create_v6_runtime。
    # 调用约定：构造实例时由 Python 调用。
    def __init__(self, accounting: ExogenousEnergyAccounting):
        self.accounting = accounting
        self._completed: Dict[str, Reservation] = {}
        self._finalized_report = None

    # 作用：把完成预留的实际核算记录追加到账本。
    # 调用上下文：调度周期遇到 TASK_COMPLETED 后调用；最终由账本汇总实际核算。
    def record_completed_reservation(self, reservation: Reservation) -> bool:
        if self._finalized_report is not None:
            raise RuntimeError("cannot append after ledger finalization")
        existing = self._completed.get(reservation.reservation_id)
        if existing is not None:
            if existing != reservation:
                raise RuntimeError("reservation id collision in metrics ledger")
            return False
        self._completed[reservation.reservation_id] = reservation
        return True

    # 作用：汇总完成记录，生成指定核算窗口的最终报告。
    # 调用约定：供所属对象、导入方或回调流程使用；未静态定位到直接调用。
    # 下游调用：self.accounting.realize。
    def finalize_after_full_settlement(self, accounting_interval=None) -> AccountingReport:
        if self._finalized_report is None:
            self._finalized_report = self.accounting.realize(
                self._completed.values(),
                accounting_interval=accounting_interval,
            )
        return self._finalized_report
