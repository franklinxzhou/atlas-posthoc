from atlas_posthoc.diagnostics.total_variation import (
    PairwiseTVResult,
    TVConvergenceResult,
    TVSnapshotResult,
    TotalVariationError,
    compute_rankwise_pairwise_tv,
    compute_tv,
    get_tv_convergence,
    make_compactness_bins,
    make_histogram_bins,
    make_vote_share_bins,
    max_average_pairwise_tv,
)

from atlas_posthoc.diagnostics.rhat import (
    RHatError,
    RHatRankRow,
    RHatSummary,
    classic_r_hat,
    gelman_rubin_by_rank,
    rank_interpretation,
    rank_normalized_split_r_hat,
    summarize_gelman_rubin_by_rank,
)

from atlas_posthoc.diagnostics.partition_metrics import (
    PairwiseDistanceMatrix,
    PartitionMetricError,
    PartitionMetricName,
    PartitionOverlap,
    TransferDistanceResult,
    get_partition_metric,
    normalized_transfer_distance,
    optimal_transport_distance,
    pairwise_partition_distance_matrix,
    partition_overlap,
    population_transfer_distance,
    transfer_distance_result,
    unconstrained_transfer_distance,
    unit_transfer_distance,
)

__all__ = [
    # total_variation
    "PairwiseTVResult",
    "TVConvergenceResult",
    "TVSnapshotResult",
    "TotalVariationError",
    "compute_rankwise_pairwise_tv",
    "compute_tv",
    "get_tv_convergence",
    "make_compactness_bins",
    "make_histogram_bins",
    "make_vote_share_bins",
    "max_average_pairwise_tv",

    # rhat
    "RHatError",
    "RHatRankRow",
    "RHatSummary",
    "classic_r_hat",
    "gelman_rubin_by_rank",
    "rank_interpretation",
    "rank_normalized_split_r_hat",
    "summarize_gelman_rubin_by_rank",

    # partition_metrics
    "PairwiseDistanceMatrix",
    "PartitionMetricError",
    "PartitionMetricName",
    "PartitionOverlap",
    "TransferDistanceResult",
    "get_partition_metric",
    "normalized_transfer_distance",
    "optimal_transport_distance",
    "pairwise_partition_distance_matrix",
    "partition_overlap",
    "population_transfer_distance",
    "transfer_distance_result",
    "unconstrained_transfer_distance",
    "unit_transfer_distance",
]