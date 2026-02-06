"""Helper types and functions for EIP-8141 frame transaction tests."""

from dataclasses import dataclass
from typing import Any, Iterable, List, Sequence

import ethereum_rlp as eth_rlp
from execution_testing import (
    Address,
    Bytecode,
    Bytes,
    Frame,
    Hash,
    Op,
    Transaction,
)

from .spec import Spec


@dataclass(frozen=True)
class FrameGasSummary:
    """Computed gas summary for a frame transaction."""

    intrinsic_gas: int
    calldata_gas: int
    frame_gas_sum: int


def calldata_cost(
    data: bytes,
    *,
    zero_cost: int = 4,
    non_zero_cost: int = 16,
) -> int:
    """Return calldata cost for the given bytes."""
    return sum(zero_cost if byte == 0 else non_zero_cost for byte in data)


def rlp_frames(frames: Sequence[Frame]) -> Bytes:
    """Return RLP encoding for a list of frames."""
    return Bytes(
        eth_rlp.encode([frame.to_list(signing=False) for frame in frames])
    )


def frame_tx_gas_summary(
    frames: Sequence[Frame],
) -> FrameGasSummary:
    """Return intrinsic and calldata gas for a frame tx."""
    encoded_frames = rlp_frames(frames)
    cd_gas = calldata_cost(encoded_frames)
    fg_sum = sum(int(frame.gas_limit) for frame in frames)
    intrinsic = Spec.FRAME_TX_INTRINSIC_COST + cd_gas + fg_sum
    return FrameGasSummary(
        intrinsic_gas=intrinsic,
        calldata_gas=cd_gas,
        frame_gas_sum=fg_sum,
    )


def build_frame(
    *,
    mode: int,
    target: Address | None,
    gas_limit: int,
    data: Bytes | bytes = b"",
) -> Frame:
    """Construct a frame with the given parameters."""
    return Frame(
        mode=mode,
        target=target,
        gas_limit=gas_limit,
        data=Bytes(data),
    )


def make_frame_tx(
    *,
    sender: Any,
    frames: Sequence[Frame],
    chain_id: int = 1,
    nonce: int = 0,
    max_fee_per_gas: int = 1,
    max_priority_fee_per_gas: int = 0,
    max_fee_per_blob_gas: int = 0,
    blob_versioned_hashes: Sequence[Hash] | None = None,
    gas_limit: int | None = None,
) -> Transaction:
    """Build a frame transaction with sensible defaults."""
    if blob_versioned_hashes is None:
        blob_versioned_hashes = []
    if gas_limit is None:
        gas_limit = frame_tx_gas_summary(frames).intrinsic_gas
    return Transaction(
        ty=Spec.FRAME_TX_TYPE,
        sender=sender,
        chain_id=chain_id,
        nonce=nonce,
        frames=list(frames),
        gas_limit=gas_limit,
        max_fee_per_gas=max_fee_per_gas,
        max_priority_fee_per_gas=max_priority_fee_per_gas,
        max_fee_per_blob_gas=max_fee_per_blob_gas,
        blob_versioned_hashes=blob_versioned_hashes,
    )


def approve_bytecode(
    scope: int,
    return_data: bytes | Bytes = b"",
) -> Bytecode:
    """Return bytecode that approves with optional return data."""
    return_data = Bytes(return_data)
    if len(return_data) == 0:
        return Op.APPROVE(scope, 0, 0)
    return Op.MSTORE(0, return_data) + Op.APPROVE(scope, 0, len(return_data))


def compute_sig_hash(tx: Transaction) -> Bytes:
    """Compute signature hash with VERIFY frame data elided."""
    frames: List[Frame] = []
    for frame in tx.frames or []:
        if int(frame.mode) == Spec.MODE_VERIFY:
            frames.append(frame.copy(data=Bytes(b"")))
        else:
            frames.append(frame)
    sanitized = tx.copy(frames=frames)
    return sanitized.rlp().keccak256()


def effective_gas_price(
    *,
    base_fee: int,
    max_fee: int,
    max_priority: int,
) -> int:
    """Return EIP-1559 effective gas price for a transaction."""
    return min(max_fee, base_fee + max_priority)


def max_cost(
    *,
    base_fee: int,
    tx_gas_limit: int,
    max_fee: int,
    max_priority: int,
    blob_count: int = 0,
    blob_base_fee: int = 0,
    gas_per_blob: int = 0,
) -> int:
    """Compute max cost for TXPARAMLOAD(0x06)."""
    price = effective_gas_price(
        base_fee=base_fee,
        max_fee=max_fee,
        max_priority=max_priority,
    )
    blob_fees = blob_count * gas_per_blob * blob_base_fee
    return tx_gas_limit * price + blob_fees


def frame_status_list(
    statuses: Iterable[int],
) -> List[int]:
    """Return a list of frame statuses as ints."""
    return list(statuses)
