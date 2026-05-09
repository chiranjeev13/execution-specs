"""Helper types and functions for EIP-8141 frame transaction tests."""

from dataclasses import dataclass
from typing import Any, Iterable, List, Sequence

import ethereum_rlp as eth_rlp
from execution_testing import (
    Address,
    Bytecode,
    Bytes,
    Conditional,
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
    per_frame = len(frames) * Spec.FRAME_TX_PER_FRAME_COST
    intrinsic = (
        Spec.FRAME_TX_INTRINSIC_COST + per_frame + cd_gas + fg_sum
    )
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
    flags: int | None = None,
    value: int = 0,
) -> Frame:
    """Construct a frame with the given parameters."""
    if flags is None:
        # VERIFY must advertise a non-zero allowed APPROVE scope (EIP-8141).
        flags = (
            Spec.APPROVE_PAYMENT | Spec.APPROVE_EXECUTION
            if mode == Spec.MODE_VERIFY
            else 0
        )
    return Frame(
        mode=mode,
        flags=flags,
        target=target,
        gas_limit=gas_limit,
        value=value,
        data=Bytes(data),
    )


def make_frame_tx(
    *,
    sender: Any,
    frames: Sequence[Frame],
    chain_id: int = 1,
    nonce: int = 1,
    max_fee_per_gas: int = 7,
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
    """Return bytecode that executes APPROVE and optionally returns data."""
    return_data = Bytes(return_data)

    if len(return_data) == 0:
        direct_approve = Op.APPROVE(0, 0, scope)
    else:
        # MSTORE stores a 32-byte big-endian value, so the data lands
        # at offset (32 - len(return_data)) within the 32-byte word.
        data_offset = 32 - len(return_data)
        direct_approve = Op.MSTORE(0, return_data) + Op.APPROVE(
            data_offset,
            len(return_data),
            scope,
        )

    # frame_target = tx.frames[current_frame_index].target, with null target
    # resolved to tx.sender.
    current_frame_index = Op.TXPARAM(0x0A)
    frame_target_raw = Op.FRAMEPARAM(0x00, current_frame_index)
    frame_target_resolved = Op.OR(
        frame_target_raw,
        Op.MUL(
            Op.ISZERO(frame_target_raw),
            Op.TXPARAM(0x02),
        ),
    )

    # ``ADDRESS == resolved_target`` (EIP-8141); use self-CALL if needed so the
    # inner execution has ``ADDRESS`` equal to the frame target contract.
    return Conditional(
        condition=Op.EQ(Op.ADDRESS, frame_target_resolved),
        if_true=direct_approve,
        if_false=(
            Op.CALL(
                gas=100_000,
                address=Op.ADDRESS,
                value=0,
                args_offset=0,
                args_size=0,
                ret_offset=0,
                ret_size=0,
            )
            + Op.RETURNDATACOPY(0, 0, Op.RETURNDATASIZE)
            + Op.RETURN(0, Op.RETURNDATASIZE)
        ),
    )


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
    """Compute max cost for ``TXPARAM(0x06)``."""
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
