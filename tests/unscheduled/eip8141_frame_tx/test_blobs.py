"""Blob-related tests for EIP-8141 frame transactions."""

import pytest
from execution_testing import (
    Account,
    Alloc,
    Blob,
    Environment,
    Fork,
    Op,
    StateTestFiller,
    TransactionException,
)

from .helpers import (
    approve_bytecode,
    build_frame,
    frame_tx_gas_summary,
    make_frame_tx,
    max_cost,
)
from .spec import Spec, ref_spec_8141

REFERENCE_SPEC_GIT_PATH = ref_spec_8141.git_path
REFERENCE_SPEC_VERSION = ref_spec_8141.version

BLOB_COUNT_SLOT = 0x01
MAX_COST_SLOT = 0x02


def test_frame_tx_with_blobs(
    state_test: StateTestFiller,
    pre: Alloc,
    fork: Fork,
) -> None:
    """Valid frame tx with blobs and TXPARAM blob fields."""
    sender = pre.fund_eoa(amount=10**18)
    pre.deploy_contract(
        code=approve_bytecode(Spec.APPROVE_BOTH), address=sender
    )

    blob = Blob.from_fork(fork)
    blob_hashes = [blob.versioned_hash]

    txparam_target = pre.deploy_contract(
        code=(
            Op.SSTORE(BLOB_COUNT_SLOT, Op.TXPARAMLOAD(0x07, 0, 0))
            + Op.SSTORE(MAX_COST_SLOT, Op.TXPARAMLOAD(0x06, 0, 0))
            + Op.STOP
        )
    )

    frames = [
        build_frame(
            mode=Spec.MODE_VERIFY,
            target=sender,
            gas_limit=40_000,
            data=b"",
        ),
        build_frame(
            mode=Spec.MODE_SENDER,
            target=txparam_target,
            gas_limit=60_000,
            data=b"",
        ),
    ]

    tx = make_frame_tx(
        sender=sender,
        frames=frames,
        max_fee_per_gas=1,
        max_priority_fee_per_gas=0,
        max_fee_per_blob_gas=fork.min_base_fee_per_blob_gas(),
        blob_versioned_hashes=blob_hashes,
    )

    gas_summary = frame_tx_gas_summary(frames)
    expected_max_cost = max_cost(
        base_fee=1,
        tx_gas_limit=gas_summary.intrinsic_gas,
        max_fee=tx.max_fee_per_gas,
        max_priority=tx.max_priority_fee_per_gas,
        blob_count=1,
        blob_base_fee=fork.min_base_fee_per_blob_gas(),
        gas_per_blob=fork.blob_gas_per_blob(),
    )

    state_test(
        env=Environment(base_fee_per_gas=1, excess_blob_gas=0),
        pre=pre,
        tx=tx,
        post={
            sender: Account(
                code=approve_bytecode(Spec.APPROVE_BOTH),
                nonce=1,
                balance=10**18,
            ),
            txparam_target: Account(
                storage={
                    BLOB_COUNT_SLOT: 1,
                    MAX_COST_SLOT: expected_max_cost,
                },
            ),
        },
    )


def test_frame_tx_with_multiple_blobs(
    state_test: StateTestFiller,
    pre: Alloc,
    fork: Fork,
) -> None:
    """Frame tx with multiple blobs: count and max cost correct."""
    sender = pre.fund_eoa(amount=10**18)
    pre.deploy_contract(
        code=approve_bytecode(Spec.APPROVE_BOTH),
        address=sender,
    )

    blob_a = Blob.from_fork(fork)
    blob_b = Blob.from_fork(fork)
    blob_hashes = [blob_a.versioned_hash, blob_b.versioned_hash]

    txparam_target = pre.deploy_contract(
        code=(
            Op.SSTORE(BLOB_COUNT_SLOT, Op.TXPARAMLOAD(0x07, 0, 0))
            + Op.SSTORE(MAX_COST_SLOT, Op.TXPARAMLOAD(0x06, 0, 0))
            + Op.STOP
        ),
    )

    frames = [
        build_frame(
            mode=Spec.MODE_VERIFY,
            target=sender,
            gas_limit=40_000,
            data=b"",
        ),
        build_frame(
            mode=Spec.MODE_SENDER,
            target=txparam_target,
            gas_limit=60_000,
            data=b"",
        ),
    ]

    min_blob_fee = fork.min_base_fee_per_blob_gas()
    tx = make_frame_tx(
        sender=sender,
        frames=frames,
        max_fee_per_gas=1,
        max_priority_fee_per_gas=0,
        max_fee_per_blob_gas=min_blob_fee,
        blob_versioned_hashes=blob_hashes,
    )

    gas_summary = frame_tx_gas_summary(frames)
    expected_max_cost = max_cost(
        base_fee=1,
        tx_gas_limit=gas_summary.intrinsic_gas,
        max_fee=tx.max_fee_per_gas,
        max_priority=tx.max_priority_fee_per_gas,
        blob_count=2,
        blob_base_fee=min_blob_fee,
        gas_per_blob=fork.blob_gas_per_blob(),
    )

    state_test(
        env=Environment(base_fee_per_gas=1, excess_blob_gas=0),
        pre=pre,
        tx=tx,
        post={
            sender: Account(
                code=approve_bytecode(Spec.APPROVE_BOTH),
                nonce=1,
                balance=10**18,
            ),
            txparam_target: Account(
                storage={
                    BLOB_COUNT_SLOT: 2,
                    MAX_COST_SLOT: expected_max_cost,
                },
            ),
        },
    )


def test_blob_fee_below_base_fee_invalid(
    state_test: StateTestFiller,
    pre: Alloc,
    fork: Fork,
) -> None:
    """Reject tx when max_fee_per_blob_gas < blob base fee."""
    sender = pre.fund_eoa(amount=10**18)
    pre.deploy_contract(
        code=approve_bytecode(Spec.APPROVE_BOTH),
        address=sender,
    )

    blob = Blob.from_fork(fork)

    frames = [
        build_frame(
            mode=Spec.MODE_VERIFY,
            target=sender,
            gas_limit=40_000,
            data=b"",
        ),
    ]

    # Set max_fee_per_blob_gas to 1 below the minimum.
    min_blob_fee = fork.min_base_fee_per_blob_gas()
    too_low = max(min_blob_fee - 1, 0)

    # Only test when min_blob_fee > 0 (otherwise can't go below).
    if too_low == 0 and min_blob_fee == 0:
        pytest.skip("min blob base fee is 0; cannot test below")

    tx = make_frame_tx(
        sender=sender,
        frames=frames,
        max_fee_per_gas=10,
        max_priority_fee_per_gas=0,
        max_fee_per_blob_gas=too_low,
        blob_versioned_hashes=[blob.versioned_hash],
    )
    tx.error = TransactionException.INSUFFICIENT_MAX_FEE_PER_BLOB_GAS

    state_test(
        env=Environment(
            base_fee_per_gas=1,
            excess_blob_gas=0,
        ),
        pre=pre,
        tx=tx,
        post={},
    )
