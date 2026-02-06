"""
Tests invalid frame transactions from EIP-8141.

Covers static validation and RLP encoding edge cases for frame transactions.
"""

from enum import Enum, auto
from typing import List, Sequence, Type

import pytest
from execution_testing import (
    Address,
    Alloc,
    Bytes,
    Frame,
    Transaction,
    TransactionException,
    TransactionTestFiller,
)
from execution_testing.base_types import FixedSizeBytes, HexNumber

from .helpers import build_frame, make_frame_tx
from .spec import Spec, ref_spec_8141

REFERENCE_SPEC_GIT_PATH = ref_spec_8141.git_path
REFERENCE_SPEC_VERSION = ref_spec_8141.version

pytestmark = pytest.mark.exception_test


class OversizedAddress(FixedSizeBytes[21]):  # type: ignore[misc]
    """Oversized Address Type."""


class UndersizedAddress(FixedSizeBytes[19]):  # type: ignore[misc]
    """Undersized Address Type."""


class OversizedInt(FixedSizeBytes[33]):  # type: ignore[misc]
    """Oversized int encoding for invalid RLP."""


class InvalidRLPMode(Enum):
    """Enum for invalid RLP modes."""

    TRUNCATED_RLP = auto()
    EXTRA_BYTES = auto()


def _dummy_frames(count: int) -> List[Frame]:
    return [
        build_frame(
            mode=Spec.MODE_VERIFY,
            target=Address(0x01),
            gas_limit=0,
            data=b"",
        )
        for _ in range(count)
    ]


@pytest.mark.parametrize(
    "frame_count,tx_error",
    [
        pytest.param(
            0, TransactionException.TYPE_6_INVALID_FRAME_FORMAT, id="zero"
        ),
        pytest.param(
            Spec.MAX_FRAMES + 1,
            TransactionException.TYPE_6_INVALID_FRAME_FORMAT,
            id="too_many",
        ),
    ],
)
def test_invalid_frame_count(
    transaction_test: TransactionTestFiller,
    pre: Alloc,
    frame_count: int,
    tx_error: TransactionException,
) -> None:
    """Reject frame transactions with invalid frame counts."""
    sender = pre.fund_eoa()
    tx = make_frame_tx(sender=sender, frames=_dummy_frames(frame_count))
    tx.error = tx_error
    transaction_test(pre=pre, tx=tx)


@pytest.mark.parametrize("mode", [3, 4])
def test_invalid_mode(
    transaction_test: TransactionTestFiller,
    pre: Alloc,
    mode: int,
) -> None:
    """Reject frame transactions with invalid modes."""
    sender = pre.fund_eoa()
    frame = build_frame(
        mode=mode,
        target=Address(0x01),
        gas_limit=0,
        data=b"",
    )
    tx = make_frame_tx(sender=sender, frames=[frame])
    tx.error = TransactionException.TYPE_6_INVALID_FRAME_FORMAT
    transaction_test(pre=pre, tx=tx)


@pytest.mark.parametrize(
    "address_type",
    [OversizedAddress, UndersizedAddress],
)
def test_invalid_target_length(
    transaction_test: TransactionTestFiller,
    pre: Alloc,
    address_type: Type[FixedSizeBytes],
) -> None:
    """Reject frames with invalid target length encodings."""

    class FrameWithInvalidTarget(Frame):
        target: address_type  # type: ignore[assignment]

    sender = pre.fund_eoa()
    frame = FrameWithInvalidTarget(
        mode=Spec.MODE_VERIFY,
        target=1,
        gas_limit=0,
        data=Bytes(b""),
    )
    tx = make_frame_tx(sender=sender, frames=[frame])
    tx.error = TransactionException.TYPE_6_INVALID_FRAME_FORMAT
    transaction_test(pre=pre, tx=tx)


@pytest.mark.parametrize(
    "sender_type",
    [OversizedAddress, UndersizedAddress],
)
def test_invalid_sender_length(
    transaction_test: TransactionTestFiller,
    pre: Alloc,
    sender_type: Type[FixedSizeBytes],
) -> None:
    """Reject transactions with invalid sender address length."""

    class TransactionWithInvalidSender(Transaction):
        sender: sender_type  # type: ignore[assignment]

    frame = build_frame(
        mode=Spec.MODE_VERIFY,
        target=Address(0x01),
        gas_limit=0,
        data=b"",
    )
    tx = TransactionWithInvalidSender(
        ty=Spec.FRAME_TX_TYPE,
        sender=1,
        frames=[frame],
        chain_id=1,
        nonce=0,
        max_fee_per_gas=1,
        max_priority_fee_per_gas=0,
        max_fee_per_blob_gas=0,
        blob_versioned_hashes=[],
    )
    tx.error = TransactionException.TYPE_6_INVALID_FRAME_FORMAT
    transaction_test(pre=pre, tx=tx)


@pytest.mark.parametrize(
    "chain_id,tx_error",
    [
        pytest.param(
            Spec.MAX_CHAIN_ID + 1,
            TransactionException.INVALID_CHAINID,
            id="2**256",
        ),
    ],
)
def test_invalid_chain_id(
    transaction_test: TransactionTestFiller,
    pre: Alloc,
    chain_id: int,
    tx_error: TransactionException,
) -> None:
    """Reject chain IDs above 2**256-1."""
    sender = pre.fund_eoa()
    frame = build_frame(
        mode=Spec.MODE_VERIFY,
        target=sender,
        gas_limit=0,
        data=b"",
    )
    tx = make_frame_tx(sender=sender, frames=[frame], chain_id=chain_id)
    tx.error = tx_error
    transaction_test(pre=pre, tx=tx)


@pytest.mark.parametrize(
    "nonce,tx_error",
    [
        pytest.param(
            Spec.MAX_NONCE + 1, TransactionException.NONCE_OVERFLOW, id="2**64"
        ),
    ],
)
def test_invalid_nonce(
    transaction_test: TransactionTestFiller,
    pre: Alloc,
    nonce: int,
    tx_error: TransactionException,
) -> None:
    """Reject nonces above 2**64-1."""
    sender = pre.fund_eoa()
    frame = build_frame(
        mode=Spec.MODE_VERIFY,
        target=sender,
        gas_limit=0,
        data=b"",
    )
    tx = make_frame_tx(sender=sender, frames=[frame], nonce=nonce)
    tx.error = tx_error
    transaction_test(pre=pre, tx=tx)


def test_invalid_blob_fields_empty_hashes_non_zero_fee(
    transaction_test: TransactionTestFiller,
    pre: Alloc,
) -> None:
    """Reject non-zero blob fee when no blob hashes are present."""
    sender = pre.fund_eoa()
    frame = build_frame(
        mode=Spec.MODE_VERIFY,
        target=sender,
        gas_limit=0,
        data=b"",
    )
    tx = make_frame_tx(
        sender=sender,
        frames=[frame],
        max_fee_per_blob_gas=1,
        blob_versioned_hashes=[],
    )
    tx.error = TransactionException.TYPE_6_INVALID_BLOB_FIELDS
    transaction_test(pre=pre, tx=tx)


def test_invalid_blob_fields_hashes_zero_fee(
    transaction_test: TransactionTestFiller,
    pre: Alloc,
) -> None:
    """Reject zero blob fee when blob hashes are present."""
    sender = pre.fund_eoa()
    frame = build_frame(
        mode=Spec.MODE_VERIFY,
        target=sender,
        gas_limit=0,
        data=b"",
    )
    tx = make_frame_tx(
        sender=sender,
        frames=[frame],
        max_fee_per_blob_gas=0,
        blob_versioned_hashes=[Bytes(b"\x11" * 32)],
    )
    tx.error = TransactionException.TYPE_6_INVALID_BLOB_FIELDS
    transaction_test(pre=pre, tx=tx)


@pytest.mark.parametrize(
    "invalid_rlp_mode",
    [
        pytest.param(InvalidRLPMode.TRUNCATED_RLP, id="truncated"),
        pytest.param(InvalidRLPMode.EXTRA_BYTES, id="extra"),
    ],
)
def test_invalid_rlp_encoding(
    transaction_test: TransactionTestFiller,
    pre: Alloc,
    invalid_rlp_mode: InvalidRLPMode,
) -> None:
    """Reject invalid RLP encodings for frame transactions."""
    sender = pre.fund_eoa()
    frame = build_frame(
        mode=Spec.MODE_VERIFY,
        target=sender,
        gas_limit=0,
        data=b"",
    )
    tx = make_frame_tx(sender=sender, frames=[frame])
    tx.error = TransactionException.TYPE_6_INVALID_FRAME_FORMAT
    if invalid_rlp_mode == InvalidRLPMode.TRUNCATED_RLP:
        tx.rlp_override = Bytes(tx.rlp()[:-1])
    else:
        tx.rlp_override = Bytes(tx.rlp() + b"\x00")
    transaction_test(pre=pre, tx=tx)


@pytest.mark.parametrize("missing_index", [0, 1])
def test_invalid_rlp_missing_field(
    transaction_test: TransactionTestFiller,
    pre: Alloc,
    missing_index: int,
) -> None:
    """Reject transactions with missing RLP fields."""

    class MissingFieldTransaction(Transaction):
        missing_index: int

        def get_rlp_fields(self) -> List[str]:
            fields = super().get_rlp_fields()
            fields.pop(self.missing_index)
            return fields

    sender = pre.fund_eoa()
    frame = build_frame(
        mode=Spec.MODE_VERIFY,
        target=sender,
        gas_limit=0,
        data=b"",
    )
    tx = MissingFieldTransaction(
        ty=Spec.FRAME_TX_TYPE,
        sender=sender,
        frames=[frame],
        chain_id=1,
        nonce=0,
        max_fee_per_gas=1,
        max_priority_fee_per_gas=0,
        max_fee_per_blob_gas=0,
        blob_versioned_hashes=[],
        missing_index=missing_index,
    )
    tx.error = TransactionException.TYPE_6_INVALID_FRAME_FORMAT
    transaction_test(pre=pre, tx=tx)


def test_invalid_frame_not_list(
    transaction_test: TransactionTestFiller,
    pre: Alloc,
) -> None:
    """Reject transactions where frames are encoded as bytes, not lists."""

    class TransactionWithBytesFrames(Transaction):
        frames: Sequence[Bytes]  # type: ignore[assignment]

    sender = pre.fund_eoa()
    frame = build_frame(
        mode=Spec.MODE_VERIFY,
        target=sender,
        gas_limit=0,
        data=b"",
    )
    tx = TransactionWithBytesFrames(
        ty=Spec.FRAME_TX_TYPE,
        sender=sender,
        frames=[frame.rlp()],
        chain_id=1,
        nonce=0,
        max_fee_per_gas=1,
        max_priority_fee_per_gas=0,
        max_fee_per_blob_gas=0,
        blob_versioned_hashes=[],
    )
    tx.error = TransactionException.TYPE_6_INVALID_FRAME_FORMAT
    transaction_test(pre=pre, tx=tx)


def test_invalid_frame_field_type(
    transaction_test: TransactionTestFiller,
    pre: Alloc,
) -> None:
    """Reject frames with invalid field types."""

    class FrameWithModeAsList(Frame):
        mode: List[HexNumber]  # type: ignore[assignment]

    sender = pre.fund_eoa()
    frame = FrameWithModeAsList(
        mode=[HexNumber(0)],
        target=sender,
        gas_limit=0,
        data=Bytes(b""),
    )
    tx = make_frame_tx(sender=sender, frames=[frame])
    tx.error = TransactionException.TYPE_6_INVALID_FRAME_FORMAT
    transaction_test(pre=pre, tx=tx)


def test_max_frames_valid(
    transaction_test: TransactionTestFiller,
    pre: Alloc,
) -> None:
    """Frame transactions with MAX_FRAMES should be accepted."""
    sender = pre.fund_eoa()
    tx = make_frame_tx(sender=sender, frames=_dummy_frames(Spec.MAX_FRAMES))
    transaction_test(pre=pre, tx=tx)


def test_valid_max_chain_id(
    transaction_test: TransactionTestFiller,
    pre: Alloc,
) -> None:
    """Accept chain_id == 2**256-1 (maximum valid value)."""
    sender = pre.fund_eoa()
    frame = build_frame(
        mode=Spec.MODE_VERIFY,
        target=sender,
        gas_limit=0,
        data=b"",
    )
    tx = make_frame_tx(
        sender=sender, frames=[frame], chain_id=Spec.MAX_CHAIN_ID
    )
    transaction_test(pre=pre, tx=tx)


def test_valid_max_nonce(
    transaction_test: TransactionTestFiller,
    pre: Alloc,
) -> None:
    """Accept nonce == 2**64-1 (maximum valid value)."""
    sender = pre.fund_eoa()
    frame = build_frame(
        mode=Spec.MODE_VERIFY,
        target=sender,
        gas_limit=0,
        data=b"",
    )
    tx = make_frame_tx(sender=sender, frames=[frame], nonce=Spec.MAX_NONCE)
    transaction_test(pre=pre, tx=tx)


def test_valid_minimal_frame_tx(
    transaction_test: TransactionTestFiller,
    pre: Alloc,
) -> None:
    """Accept a minimal valid frame tx: 1 VERIFY + 1 SENDER frame."""
    sender = pre.fund_eoa()
    frames = [
        build_frame(
            mode=Spec.MODE_VERIFY,
            target=sender,
            gas_limit=21_000,
            data=b"",
        ),
        build_frame(
            mode=Spec.MODE_SENDER,
            target=Address(0x01),
            gas_limit=21_000,
            data=b"",
        ),
    ]
    tx = make_frame_tx(sender=sender, frames=frames)
    transaction_test(pre=pre, tx=tx)


def test_null_target_valid(
    transaction_test: TransactionTestFiller,
    pre: Alloc,
) -> None:
    """Accept a frame with null target (interpreted as tx.sender)."""
    sender = pre.fund_eoa()
    frame = build_frame(
        mode=Spec.MODE_VERIFY,
        target=None,
        gas_limit=21_000,
        data=b"",
    )
    tx = make_frame_tx(sender=sender, frames=[frame])
    transaction_test(pre=pre, tx=tx)


def test_frame_gas_limit_zero(
    transaction_test: TransactionTestFiller,
    pre: Alloc,
) -> None:
    """Accept a frame with gas_limit=0 (allowed per spec, will OOG)."""
    sender = pre.fund_eoa()
    frame = build_frame(
        mode=Spec.MODE_VERIFY,
        target=sender,
        gas_limit=0,
        data=b"",
    )
    tx = make_frame_tx(sender=sender, frames=[frame])
    # gas_limit=0 is allowed but the frame will OOG, making the
    # VERIFY frame fail to APPROVE → invalid tx.
    tx.error = TransactionException.TYPE_6_INVALID_FRAME_EXECUTION
    transaction_test(pre=pre, tx=tx)


@pytest.mark.parametrize("mode", [0, 1, 2])
def test_valid_modes(
    transaction_test: TransactionTestFiller,
    pre: Alloc,
    mode: int,
) -> None:
    """Accept frames with each valid mode value (0, 1, 2)."""
    sender = pre.fund_eoa()
    frame = build_frame(
        mode=mode,
        target=sender,
        gas_limit=21_000,
        data=b"",
    )
    tx = make_frame_tx(sender=sender, frames=[frame])
    transaction_test(pre=pre, tx=tx)
