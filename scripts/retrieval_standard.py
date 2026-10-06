"""Compatibility imports for the installed standard runtime and historical clients."""
from zero_index.standard import (
    STANDARD_POLICY, MEASURED_JJJ_CONFIG, RetrievalAdapter, RetrievalConfig,
    select_standard_evidence, retrieve_standard,
)
from scripts.jev_scoped_client import ScopedEvidenceJev
from scripts.jev_evidence_comparison import EvidenceComparisonJev
from scripts.jev_joint_evidence import SharedSetJev


def make_standard_clients(cache,budget):
    """Historical cached/accounted Jev clients used for the measured archive."""
    clients=(ScopedEvidenceJev(cache,budget),EvidenceComparisonJev(cache,budget),SharedSetJev(cache,budget))
    for client in clients:client.max_state_chars=250000
    return clients
