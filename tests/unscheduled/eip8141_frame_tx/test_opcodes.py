"""Tests opcode semantics for EIP-8141 frame transactions."""

from dataclasses import dataclass
from typing import Callable

import pytest
from execution_testing import (
    Account,
    Address,
    Alloc,
    Conditional,
    Environment,
    FrameReceipt,
    Op,
    StateTestFiller,
    TransactionException,
    TransactionReceipt,
)

from .helpers import (
    approve_bytecode,
    build_frame,
    compute_sig_hash,
    frame_tx_gas_summary,
    make_frame_tx,
    max_cost,
)
from .spec import Spec, ref_spec_8141

REFERENCE_SPEC_GIT_PATH = ref_spec_8141.git_path
REFERENCE_SPEC_VERSION = ref_spec_8141.version

CALLER_SLOT = 0x01
TXPARAM_SLOT = 0x02
SIZE_SLOT = 0x03
DATA_SLOT = 0x04
STATUS_SLOT = 0x05


def caller_origin_guard(expected: Address, success_code) -> bytes:
    """Return bytecode that asserts caller/origin and executes success_code."""
    return Conditional(
        condition=Op.AND(
            Op.EQ(Op.CALLER, expected),
            Op.EQ(Op.ORIGIN, expected),
        ),
        if_true=success_code,
        if_false=Op.REVERT(0, 0),
    )


@pytest.mark.parametrize(
    "mode,expected_caller",
    [
        pytest.param(Spec.MODE_DEFAULT, Spec.ENTRY_POINT, id="default"),
        pytest.param(Spec.MODE_SENDER, "sender", id="sender"),
    ],
)
def test_caller_origin_modes(
    state_test: StateTestFiller,
    pre: Alloc,
    mode: int,
    expected_caller: Address | str,
) -> None:
    """Caller/origin semantics for DEFAULT and SENDER modes."""
    sender = pre.fund_eoa(amount=10**18)
    pre.deploy_contract(
        code=approve_bytecode(Spec.APPROVE_BOTH), address=sender
    )

    expected_address = (
        sender if isinstance(expected_caller, str) else expected_caller
    )
    target_code = caller_origin_guard(
        expected_address,
        Op.SSTORE(CALLER_SLOT, 1) + Op.STOP,
    )
    target = pre.deploy_contract(code=target_code)

    frames = [
        build_frame(
            mode=Spec.MODE_VERIFY,
            target=sender,
            gas_limit=50_000,
            data=b"",
        ),
        build_frame(
            mode=mode,
            target=target,
            gas_limit=50_000,
            data=b"",
        ),
    ]

    tx = make_frame_tx(
        sender=sender,
        frames=frames,
        max_fee_per_gas=7,
        max_priority_fee_per_gas=0,
    )

    state_test(
        env=Environment(),
        pre=pre,
        tx=tx,
        post={
            sender: Account(
                code=approve_bytecode(Spec.APPROVE_BOTH),
                nonce=1,
            ),
            target: Account(storage={CALLER_SLOT: 1}),
        },
    )


def test_caller_origin_verify_mode(
    state_test: StateTestFiller,
    pre: Alloc,
) -> None:
    """Caller/origin semantics for VERIFY mode (ENTRY_POINT)."""
    sender = pre.fund_eoa(amount=10**18)
    sender_code = caller_origin_guard(
        Spec.ENTRY_POINT,
        approve_bytecode(Spec.APPROVE_BOTH),
    )
    pre.deploy_contract(code=sender_code, address=sender)

    frames = [
        build_frame(
            mode=Spec.MODE_VERIFY,
            target=sender,
            gas_limit=50_000,
            data=b"",
        )
    ]
    tx = make_frame_tx(
        sender=sender,
        frames=frames,
        max_fee_per_gas=7,
        max_priority_fee_per_gas=0,
    )

    tx.expected_receipt = TransactionReceipt(
        payer=sender,
        frame_receipts=[FrameReceipt(status=Spec.STATUS_APPROVED_BOTH)],
    )

    state_test(
        env=Environment(),
        pre=pre,
        tx=tx,
        post={sender: Account(code=sender_code, nonce=1)},
    )


@pytest.mark.parametrize(
    "mode",
    [Spec.MODE_DEFAULT, Spec.MODE_SENDER],
)
def test_null_target_calls_sender(
    state_test: StateTestFiller,
    pre: Alloc,
    mode: int,
) -> None:
    """Null target should call tx.sender in non-VERIFY modes."""
    sender = pre.fund_eoa(amount=10**18)
    sender_code = Conditional(
        condition=Op.ISZERO(Op.CALLDATALOAD(0)),
        if_true=approve_bytecode(Spec.APPROVE_BOTH),
        if_false=Op.SSTORE(CALLER_SLOT, 1) + Op.STOP,
    )
    pre.deploy_contract(code=sender_code, address=sender)

    frames = [
        build_frame(
            mode=Spec.MODE_VERIFY,
            target=sender,
            gas_limit=50_000,
            data=b"\x00",
        ),
        build_frame(
            mode=mode,
            target=None,
            gas_limit=50_000,
            data=b"\x01",
        ),
    ]
    tx = make_frame_tx(
        sender=sender,
        frames=frames,
        max_fee_per_gas=7,
        max_priority_fee_per_gas=0,
    )

    state_test(
        env=Environment(),
        pre=pre,
        tx=tx,
        post={
            sender: Account(
                code=sender_code,
                nonce=1,
                storage={CALLER_SLOT: 1},
            ),
        },
    )


@pytest.mark.exception_test
def test_invalid_approve_scope(
    state_test: StateTestFiller,
    pre: Alloc,
) -> None:
    """APPROVE scope >= 3 should exceptional halt."""
    sender = pre.fund_eoa(amount=10**18)
    pre.deploy_contract(code=Op.APPROVE(3, 0, 0), address=sender)

    frames = [
        build_frame(
            mode=Spec.MODE_VERIFY,
            target=sender,
            gas_limit=50_000,
            data=b"",
        )
    ]
    tx = make_frame_tx(sender=sender, frames=frames)
    tx.error = TransactionException.TYPE_6_INVALID_FRAME_EXECUTION

    state_test(
        env=Environment(),
        pre=pre,
        tx=tx,
        post={},
    )


@pytest.mark.exception_test
def test_invalid_approve_scope_zero_non_sender(
    state_test: StateTestFiller,
    pre: Alloc,
) -> None:
    """Scope=0 with non-sender target should invalidate the tx."""
    sender = pre.fund_eoa(amount=10**18)
    other = pre.deploy_contract(code=approve_bytecode(Spec.APPROVE_EXECUTION))

    frames = [
        build_frame(
            mode=Spec.MODE_VERIFY,
            target=other,
            gas_limit=50_000,
            data=b"",
        )
    ]
    tx = make_frame_tx(sender=sender, frames=frames)
    tx.error = TransactionException.TYPE_6_INVALID_APPROVAL

    state_test(env=Environment(), pre=pre, tx=tx, post={})


def test_approve_return_data_and_call_status(
    state_test: StateTestFiller,
    pre: Alloc,
) -> None:
    """APPROVE should return data and propagate 2-4 statuses in CALL."""
    sender = pre.fund_eoa(amount=10**18)
    pre.deploy_contract(
        code=approve_bytecode(Spec.APPROVE_BOTH), address=sender
    )

    return_data = b"\x12\x34"
    approve_target = pre.deploy_contract(
        code=approve_bytecode(Spec.APPROVE_BOTH, return_data)
    )

    caller_code = (
        Op.SSTORE(
            STATUS_SLOT,
            Op.CALL(
                gas=100_000,
                address=approve_target,
                value=0,
                args_offset=0,
                args_size=0,
                ret_offset=0,
                ret_size=32,
            ),
        )
        + Op.RETURNDATACOPY(0, 0, len(return_data))
        + Op.SSTORE(DATA_SLOT, Op.MLOAD(0))
        + Op.STOP
    )
    caller = pre.deploy_contract(code=caller_code)

    frames = [
        build_frame(
            mode=Spec.MODE_VERIFY,
            target=sender,
            gas_limit=100_000,
            data=b"",
        ),
        build_frame(
            mode=Spec.MODE_SENDER,
            target=caller,
            gas_limit=100_000,
            data=b"",
        ),
    ]
    tx = make_frame_tx(
        sender=sender,
        frames=frames,
        max_fee_per_gas=7,
        max_priority_fee_per_gas=0,
    )

    expected_word = int.from_bytes(
        return_data.ljust(32, b"\x00"), byteorder="big"
    )

    state_test(
        env=Environment(),
        pre=pre,
        tx=tx,
        post={
            sender: Account(
                code=approve_bytecode(Spec.APPROVE_BOTH),
                nonce=1,
            ),
            caller: Account(
                storage={
                    STATUS_SLOT: Spec.STATUS_APPROVED_BOTH,
                    DATA_SLOT: expected_word,
                }
            ),
        },
    )


@dataclass(frozen=True)
class TxParamCase:
    """Test case for TXPARAMLOAD selectors."""

    selector: int
    index: int
    expected: Callable
    id: str


@pytest.mark.parametrize(
    "case",
    [
        TxParamCase(0x00, 0, lambda _: Spec.FRAME_TX_TYPE, "tx_type"),
        TxParamCase(0x01, 0, lambda ctx: ctx.nonce, "nonce"),
        TxParamCase(0x02, 0, lambda ctx: ctx.sender_int, "sender"),
        TxParamCase(0x03, 0, lambda ctx: ctx.max_priority_fee, "max_priority"),
        TxParamCase(0x04, 0, lambda ctx: ctx.max_fee, "max_fee"),
        TxParamCase(0x05, 0, lambda ctx: ctx.max_blob_fee, "max_blob_fee"),
        TxParamCase(0x06, 0, lambda ctx: ctx.max_cost, "max_cost"),
        TxParamCase(0x07, 0, lambda _: 0, "blob_count"),
        TxParamCase(0x08, 0, lambda ctx: ctx.sig_hash, "sig_hash"),
        TxParamCase(0x09, 0, lambda ctx: ctx.frame_count, "frame_count"),
        TxParamCase(0x10, 0, lambda ctx: ctx.current_index, "current_index"),
        TxParamCase(0x11, 1, lambda ctx: ctx.frame_target_int, "frame_target"),
        TxParamCase(
            0x13, 1, lambda ctx: ctx.frame_gas_limit, "frame_gas_limit"
        ),
        TxParamCase(0x14, 1, lambda ctx: ctx.frame_mode, "frame_mode"),
    ],
    ids=lambda case: case.id,
)
def test_txparamload_fields(
    state_test: StateTestFiller,
    pre: Alloc,
    case: TxParamCase,
) -> None:
    """TXPARAMLOAD should expose static transaction fields."""
    sender = pre.fund_eoa(amount=10**18)
    pre.deploy_contract(
        code=approve_bytecode(Spec.APPROVE_BOTH), address=sender
    )

    txparam_target = pre.deploy_contract(
        code=Op.SSTORE(
            TXPARAM_SLOT, Op.TXPARAMLOAD(case.selector, case.index, 0)
        )
        + Op.STOP
    )

    frames = [
        build_frame(
            mode=Spec.MODE_VERIFY,
            target=sender,
            gas_limit=40_000,
            data=b"sig-data",
        ),
        build_frame(
            mode=Spec.MODE_SENDER,
            target=txparam_target,
            gas_limit=60_000,
            data=b"call-data",
        ),
    ]

    tx = make_frame_tx(
        sender=sender,
        frames=frames,
        max_fee_per_gas=10,
        max_priority_fee_per_gas=2,
    )

    gas_summary = frame_tx_gas_summary(frames)
    tx_gas_limit = gas_summary.intrinsic_gas
    _sig_hash = int.from_bytes(compute_sig_hash(tx), byteorder="big")

    class Ctx:
        sender_int = int.from_bytes(bytes(sender), byteorder="big")
        nonce = tx.nonce
        max_priority_fee = tx.max_priority_fee_per_gas
        max_fee = tx.max_fee_per_gas
        max_blob_fee = tx.max_fee_per_blob_gas
        frame_count = len(frames)
        current_index = 1
        frame_target_int = int.from_bytes(
            bytes(txparam_target), byteorder="big"
        )
        frame_gas_limit = frames[1].gas_limit
        frame_mode = frames[1].mode
        sig_hash = _sig_hash
        max_cost = max_cost(
            base_fee=7,
            tx_gas_limit=tx_gas_limit,
            max_fee=tx.max_fee_per_gas,
            max_priority=tx.max_priority_fee_per_gas,
        )

    expected_value = case.expected(Ctx)

    state_test(
        env=Environment(),
        pre=pre,
        tx=tx,
        post={
            sender: Account(
                code=approve_bytecode(Spec.APPROVE_BOTH),
                nonce=1,
            ),
            txparam_target: Account(storage={TXPARAM_SLOT: expected_value}),
        },
    )


@pytest.mark.parametrize(
    "frame_index,expected_size,expected_word",
    [
        pytest.param(
            1,
            9,
            int.from_bytes(b"call-data".ljust(32, b"\x00"), byteorder="big"),
            id="sender_frame",
        ),
        pytest.param(0, 0, 0, id="verify_elided"),
    ],
)
def test_txparam_data_access(
    state_test: StateTestFiller,
    pre: Alloc,
    frame_index: int,
    expected_size: int,
    expected_word: int,
) -> None:
    """TXPARAMSIZE/COPY should expose frame data (elided for VERIFY)."""
    sender = pre.fund_eoa(amount=10**18)
    pre.deploy_contract(
        code=approve_bytecode(Spec.APPROVE_BOTH), address=sender
    )

    code = (
        Op.SSTORE(SIZE_SLOT, Op.TXPARAMSIZE(0x12, frame_index))
        + Op.TXPARAMCOPY(0x12, frame_index, 0, 0, 32)
        + Op.SSTORE(DATA_SLOT, Op.MLOAD(0))
        + Op.STOP
    )
    target = pre.deploy_contract(code=code)

    frames = [
        build_frame(
            mode=Spec.MODE_VERIFY,
            target=sender,
            gas_limit=40_000,
            data=b"sig-data",
        ),
        build_frame(
            mode=Spec.MODE_SENDER,
            target=target,
            gas_limit=60_000,
            data=b"call-data",
        ),
    ]
    tx = make_frame_tx(
        sender=sender,
        frames=frames,
        max_fee_per_gas=7,
        max_priority_fee_per_gas=0,
    )

    state_test(
        env=Environment(),
        pre=pre,
        tx=tx,
        post={
            sender: Account(
                code=approve_bytecode(Spec.APPROVE_BOTH),
                nonce=1,
            ),
            target: Account(
                storage={SIZE_SLOT: expected_size, DATA_SLOT: expected_word}
            ),
        },
    )


def test_txparam_status_previous_frames(
    state_test: StateTestFiller,
    pre: Alloc,
) -> None:
    """TXPARAMLOAD(0x15) returns status for previous frames."""
    sender = pre.fund_eoa(amount=10**18)
    pre.deploy_contract(
        code=approve_bytecode(Spec.APPROVE_BOTH), address=sender
    )

    status_reader = pre.deploy_contract(
        code=Op.SSTORE(STATUS_SLOT, Op.TXPARAMLOAD(0x15, 0, 0)) + Op.STOP
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
            target=status_reader,
            gas_limit=60_000,
            data=b"",
        ),
    ]
    tx = make_frame_tx(
        sender=sender,
        frames=frames,
        max_fee_per_gas=7,
        max_priority_fee_per_gas=0,
    )

    state_test(
        env=Environment(),
        pre=pre,
        tx=tx,
        post={
            sender: Account(
                code=approve_bytecode(Spec.APPROVE_BOTH),
                nonce=1,
            ),
            status_reader: Account(
                storage={STATUS_SLOT: Spec.STATUS_APPROVED_BOTH}
            ),
        },
    )


def test_txparam_status_current_frame_invalid(
    state_test: StateTestFiller,
    pre: Alloc,
) -> None:
    """TXPARAMLOAD(0x15) on current frame should exceptional halt the frame."""
    sender = pre.fund_eoa(amount=10**18)
    pre.deploy_contract(
        code=approve_bytecode(Spec.APPROVE_BOTH), address=sender
    )

    reader = pre.deploy_contract(
        code=Op.SSTORE(STATUS_SLOT, Op.TXPARAMLOAD(0x15, 1, 0)) + Op.STOP
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
            target=reader,
            gas_limit=60_000,
            data=b"",
        ),
    ]
    tx = make_frame_tx(sender=sender, frames=frames)
    # The SENDER frame fails (ExceptionalHalt) but the tx still succeeds
    # since the sender approved BOTH in the VERIFY frame.

    state_test(
        env=Environment(),
        pre=pre,
        tx=tx,
        post={
            # The SENDER frame's SSTORE is reverted due to ExceptionalHalt,
            # so STATUS_SLOT remains 0.
            reader: Account(storage={STATUS_SLOT: 0}),
        },
    )


def test_txparam_invalid_selector(
    state_test: StateTestFiller,
    pre: Alloc,
) -> None:
    """Invalid TXPARAM selector should exceptional halt the frame."""
    sender = pre.fund_eoa(amount=10**18)
    pre.deploy_contract(
        code=approve_bytecode(Spec.APPROVE_BOTH), address=sender
    )

    reader = pre.deploy_contract(
        code=Op.SSTORE(TXPARAM_SLOT, Op.TXPARAMLOAD(0x16, 0, 0)) + Op.STOP
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
            target=reader,
            gas_limit=60_000,
            data=b"",
        ),
    ]
    tx = make_frame_tx(sender=sender, frames=frames)

    state_test(
        env=Environment(),
        pre=pre,
        tx=tx,
        post={reader: Account(storage={TXPARAM_SLOT: 0})},
    )


def test_txparam_out_of_bounds_frame_index(
    state_test: StateTestFiller,
    pre: Alloc,
) -> None:
    """Out-of-bounds frame index should exceptional halt the frame."""
    sender = pre.fund_eoa(amount=10**18)
    pre.deploy_contract(
        code=approve_bytecode(Spec.APPROVE_BOTH), address=sender
    )

    reader = pre.deploy_contract(
        code=Op.SSTORE(TXPARAM_SLOT, Op.TXPARAMLOAD(0x11, 5, 0)) + Op.STOP
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
            target=reader,
            gas_limit=60_000,
            data=b"",
        ),
    ]
    tx = make_frame_tx(sender=sender, frames=frames)

    state_test(
        env=Environment(),
        pre=pre,
        tx=tx,
        post={reader: Account(storage={TXPARAM_SLOT: 0})},
    )


def test_null_target_verify_mode(
    state_test: StateTestFiller,
    pre: Alloc,
) -> None:
    """Null target in VERIFY mode should call tx.sender."""
    sender = pre.fund_eoa(amount=10**18)
    pre.deploy_contract(
        code=approve_bytecode(Spec.APPROVE_BOTH), address=sender
    )

    frames = [
        build_frame(
            mode=Spec.MODE_VERIFY,
            target=None,
            gas_limit=50_000,
            data=b"",
        ),
        build_frame(
            mode=Spec.MODE_SENDER,
            target=sender,
            gas_limit=50_000,
            data=b"",
        ),
    ]
    tx = make_frame_tx(
        sender=sender,
        frames=frames,
        max_fee_per_gas=7,
        max_priority_fee_per_gas=0,
    )
    tx.expected_receipt = TransactionReceipt(
        payer=sender,
        frame_receipts=[
            FrameReceipt(status=Spec.STATUS_APPROVED_BOTH),
            FrameReceipt(status=Spec.STATUS_SUCCESS),
        ],
    )

    state_test(
        env=Environment(),
        pre=pre,
        tx=tx,
        post={
            sender: Account(
                code=approve_bytecode(Spec.APPROVE_BOTH),
                nonce=1,
            ),
        },
    )


def test_txparam_status_future_frame_invalid(
    state_test: StateTestFiller,
    pre: Alloc,
) -> None:
    """TXPARAMLOAD(0x15) on a future frame index halts the frame."""
    sender = pre.fund_eoa(amount=10**18)
    pre.deploy_contract(
        code=approve_bytecode(Spec.APPROVE_BOTH),
        address=sender,
    )

    # Frame index 2 is the last frame; reading status of
    # index 2 from index 1 is a future frame.
    reader = pre.deploy_contract(
        code=(Op.SSTORE(STATUS_SLOT, Op.TXPARAMLOAD(0x15, 2, 0)) + Op.STOP),
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
            target=reader,
            gas_limit=60_000,
            data=b"",
        ),
        build_frame(
            mode=Spec.MODE_SENDER,
            target=sender,
            gas_limit=40_000,
            data=b"",
        ),
    ]
    tx = make_frame_tx(sender=sender, frames=frames)

    state_test(
        env=Environment(),
        pre=pre,
        tx=tx,
        post={reader: Account(storage={STATUS_SLOT: 0})},
    )


def test_approve_in_subcall_propagation(
    state_test: StateTestFiller,
    pre: Alloc,
) -> None:
    """APPROVE in nested CALL propagates status 2-4."""
    sender = pre.fund_eoa(amount=10**18)
    pre.deploy_contract(
        code=approve_bytecode(Spec.APPROVE_BOTH),
        address=sender,
    )

    inner = pre.deploy_contract(
        code=approve_bytecode(Spec.APPROVE_BOTH),
    )

    outer_code = (
        Op.SSTORE(
            STATUS_SLOT,
            Op.CALL(
                gas=100_000,
                address=inner,
                value=0,
                args_offset=0,
                args_size=0,
                ret_offset=0,
                ret_size=0,
            ),
        )
        + Op.STOP
    )
    outer = pre.deploy_contract(code=outer_code)

    frames = [
        build_frame(
            mode=Spec.MODE_VERIFY,
            target=sender,
            gas_limit=100_000,
            data=b"",
        ),
        build_frame(
            mode=Spec.MODE_SENDER,
            target=outer,
            gas_limit=200_000,
            data=b"",
        ),
    ]
    tx = make_frame_tx(
        sender=sender,
        frames=frames,
        max_fee_per_gas=7,
        max_priority_fee_per_gas=0,
    )

    state_test(
        env=Environment(),
        pre=pre,
        tx=tx,
        post={
            outer: Account(
                storage={
                    STATUS_SLOT: Spec.STATUS_APPROVED_BOTH,
                },
            ),
        },
    )


def test_txparamcopy_out_of_bounds_offset(
    state_test: StateTestFiller,
    pre: Alloc,
) -> None:
    """TXPARAMCOPY with offset beyond data returns zero-padded."""
    sender = pre.fund_eoa(amount=10**18)
    pre.deploy_contract(
        code=approve_bytecode(Spec.APPROVE_BOTH),
        address=sender,
    )

    frame_data = b"hello"
    code = (
        Op.TXPARAMCOPY(0x12, 1, 0, 100, 32)
        + Op.SSTORE(DATA_SLOT, Op.MLOAD(0))
        + Op.STOP
    )
    target = pre.deploy_contract(code=code)

    frames = [
        build_frame(
            mode=Spec.MODE_VERIFY,
            target=sender,
            gas_limit=40_000,
            data=b"",
        ),
        build_frame(
            mode=Spec.MODE_SENDER,
            target=target,
            gas_limit=60_000,
            data=frame_data,
        ),
    ]
    tx = make_frame_tx(
        sender=sender,
        frames=frames,
        max_fee_per_gas=7,
        max_priority_fee_per_gas=0,
    )

    state_test(
        env=Environment(),
        pre=pre,
        tx=tx,
        post={
            target: Account(storage={DATA_SLOT: 0}),
        },
    )


def test_verify_data_elision_from_other_frame(
    state_test: StateTestFiller,
    pre: Alloc,
) -> None:
    """SENDER frame reading VERIFY frame data via TXPARAM sees 0."""
    sender = pre.fund_eoa(amount=10**18)
    pre.deploy_contract(
        code=approve_bytecode(Spec.APPROVE_BOTH),
        address=sender,
    )

    code = (
        Op.SSTORE(SIZE_SLOT, Op.TXPARAMSIZE(0x12, 0))
        + Op.TXPARAMCOPY(0x12, 0, 0, 0, 32)
        + Op.SSTORE(DATA_SLOT, Op.MLOAD(0))
        + Op.STOP
    )
    target = pre.deploy_contract(code=code)

    frames = [
        build_frame(
            mode=Spec.MODE_VERIFY,
            target=sender,
            gas_limit=40_000,
            data=b"verify-secret",
        ),
        build_frame(
            mode=Spec.MODE_SENDER,
            target=target,
            gas_limit=60_000,
            data=b"sender-data",
        ),
    ]
    tx = make_frame_tx(
        sender=sender,
        frames=frames,
        max_fee_per_gas=7,
        max_priority_fee_per_gas=0,
    )

    state_test(
        env=Environment(),
        pre=pre,
        tx=tx,
        post={
            target: Account(
                storage={SIZE_SLOT: 0, DATA_SLOT: 0},
            ),
        },
    )
