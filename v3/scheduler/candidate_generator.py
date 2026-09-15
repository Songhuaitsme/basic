"""V3 candidate generator using the frozen V1 candidate semantics."""

from v1.scheduler.candidate_generator import CandidateGenerator


class V3CandidateGenerator(CandidateGenerator):
    """Marker implementation for the lossless V3 candidate pipeline.

    Feasibility remains ordered as CPU, Path, SLA, then exact accounting.  The
    expensive accounting implementation is supplied by the V3 runtime, so no
    V1 candidate-generation source needs to be edited or copied.
    """

    optimization_version = "sweep-prefix-full-sla-v1"
