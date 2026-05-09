"""Tests opcode semantics for EIP-8141 frame transactions."""

from dataclasses import dataclass
from typing import Any, Callable

import pytest
from execution_testing import (
    Account,
    Address,
    Alloc,
    Bytecode,
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

pytestmark = pytest.mark.valid_from("Bogota")

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
        condition=Op.OR(
            Op.AND(
                Op.EQ(Op.CALLER, expected),
                Op.EQ(Op.ORIGIN, expected),
            ),
            # Allow self-calls used by APPROVE call-trampoline patterns.
            Op.EQ(Op.CALLER, Op.ADDRESS),
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
    sender = pre.deploy_contract(
        code=approve_bytecode(Spec.APPROVE_BOTH),
        balance=10**18,
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
                nonce=2,
            ),
            target: Account(storage={CALLER_SLOT: 1}),
        },
    )


def test_caller_origin_verify_mode(
    state_test: StateTestFiller,
    pre: Alloc,
) -> None:
    """Caller/origin semantics for VERIFY mode (ENTRY_POINT)."""
    sender_code = caller_origin_guard(
        Spec.ENTRY_POINT,
        approve_bytecode(Spec.APPROVE_BOTH),
    )
    sender = pre.deploy_contract(code=sender_code, balance=10**18)

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
        frame_receipts=[FrameReceipt(status=Spec.STATUS_SUCCESS)],
    )

    state_test(
        env=Environment(),
        pre=pre,
        tx=tx,
        post={sender: Account(code=sender_code, nonce=2)},
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
    sender_code = Conditional(
        condition=Op.ISZERO(Op.CALLDATALOAD(0)),
        if_true=approve_bytecode(Spec.APPROVE_BOTH),
        if_false=Op.SSTORE(CALLER_SLOT, 1) + Op.STOP,
    )
    sender = pre.deploy_contract(code=sender_code, balance=10**18)

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
                nonce=2,
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
    sender = pre.deploy_contract(code=Op.APPROVE(0, 0, 4), balance=10**18)

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
    sender = pre.deploy_contract(code=Op.STOP, balance=10**18)
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
    tx.error = TransactionException.TYPE_6_INVALID_FRAME_EXECUTION

    state_test(env=Environment(), pre=pre, tx=tx, post={})


def test_approve_return_data_and_call_status(
    state_test: StateTestFiller,
    pre: Alloc,
) -> None:
    """
    EIP-8141 requires ``ADDRESS == resolved_target`` for every ``APPROVE``, and
    ``APPROVE_PAYMENT`` collects fees from ``resolved_target``. Payment approval
    therefore runs only on a ``VERIFY`` frame whose target is the paymaster
    (see EIP example “Canonical Paymaster”), not inside a nested ``CALL`` from
    another ``SENDER`` target.

    Here the paymaster frame uses ``APPROVE_PAYMENT`` with output data; a later
    ``SENDER`` frame uses ordinary ``CALL`` + ``RETURN`` to assert binary call
    status and ``RETURNDATA`` handling.
    """
    sender = pre.deploy_contract(
        code=approve_bytecode(Spec.APPROVE_EXECUTION),
        balance=10**18,
    )

    return_data = b"\x12\x34"
    paymaster = pre.deploy_contract(
        code=approve_bytecode(Spec.APPROVE_PAYMENT, return_data),
        balance=10**18,
    )
    data_offset = 32 - len(return_data)
    callee_code = Op.MSTORE(0, return_data) + Op.RETURN(
        data_offset, len(return_data)
    )
    callee = pre.deploy_contract(code=callee_code)

    caller_code = (
        Op.SSTORE(
            STATUS_SLOT,
            Op.CALL(
                gas=100_000,
                address=callee,
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
    caller = pre.deploy_contract(code=caller_code, balance=10**18)

    frames = [
        build_frame(
            mode=Spec.MODE_VERIFY,
            target=sender,
            gas_limit=100_000,
            data=b"",
            flags=Spec.APPROVE_EXECUTION,
        ),
        build_frame(
            mode=Spec.MODE_VERIFY,
            target=paymaster,
            gas_limit=100_000,
            data=b"",
            flags=Spec.APPROVE_PAYMENT,
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
                code=approve_bytecode(Spec.APPROVE_EXECUTION),
                nonce=2,
            ),
            caller: Account(
                storage={
                    STATUS_SLOT: Spec.STATUS_SUCCESS,
                    DATA_SLOT: expected_word,
                }
            ),
        },
    )


@dataclass(frozen=True)
class TxParamCase:
    """Test case for ``TXPARAM`` or ``FRAMEPARAM``."""

    expected: Callable[[Any], int]
    id: str
    txparam: int | None = None
    frameparam: tuple[int, int] | None = None


def txparam_case_opcode(case: TxParamCase) -> Bytecode:
    """Bytecode that pushes the word under test onto the stack."""
    if case.txparam is not None and case.frameparam is not None:
        raise ValueError("TxParamCase must set only one of txparam or frameparam")
    if case.txparam is not None:
        return Op.TXPARAM(case.txparam)
    if case.frameparam is not None:
        fp, fi = case.frameparam
        return Op.FRAMEPARAM(param=fp, frame_index=fi)
    raise ValueError("TxParamCase requires txparam or frameparam")


@pytest.mark.parametrize(
    "case",
    [
        TxParamCase(txparam=0x00, expected=lambda _: Spec.FRAME_TX_TYPE, id="tx_type"),
        TxParamCase(txparam=0x01, expected=lambda ctx: ctx.nonce, id="nonce"),
        TxParamCase(txparam=0x02, expected=lambda ctx: ctx.sender_int, id="sender"),
        TxParamCase(
            txparam=0x03, expected=lambda ctx: ctx.max_priority_fee, id="max_priority"
        ),
        TxParamCase(txparam=0x04, expected=lambda ctx: ctx.max_fee, id="max_fee"),
        TxParamCase(
            txparam=0x05, expected=lambda ctx: ctx.max_blob_fee, id="max_blob_fee"
        ),
        TxParamCase(txparam=0x06, expected=lambda ctx: ctx.max_cost, id="max_cost"),
        TxParamCase(txparam=0x07, expected=lambda _: 0, id="blob_count"),
        TxParamCase(txparam=0x08, expected=lambda ctx: ctx.sig_hash, id="sig_hash"),
        TxParamCase(
            txparam=0x09, expected=lambda ctx: ctx.frame_count, id="frame_count"
        ),
        TxParamCase(
            txparam=0x0A, expected=lambda ctx: ctx.current_index, id="current_index"
        ),
        TxParamCase(
            frameparam=(0x00, 1),
            expected=lambda ctx: ctx.frame_target_int,
            id="frame_target",
        ),
        TxParamCase(
            frameparam=(0x01, 1),
            expected=lambda ctx: ctx.frame_gas_limit,
            id="frame_gas_limit",
        ),
        TxParamCase(
            frameparam=(0x02, 1),
            expected=lambda ctx: ctx.frame_mode,
            id="frame_mode",
        ),
    ],
    ids=lambda case: case.id,
)
def test_txparam_fields(
    state_test: StateTestFiller,
    pre: Alloc,
    case: TxParamCase,
) -> None:
    """``TXPARAM`` / ``FRAMEPARAM`` expose static transaction and frame fields."""
    sender = pre.deploy_contract(
        code=approve_bytecode(Spec.APPROVE_BOTH),
        balance=10**18,
    )

    txparam_target = pre.deploy_contract(
        code=Op.SSTORE(TXPARAM_SLOT, txparam_case_opcode(case)) + Op.STOP
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
                nonce=2,
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
    """``FRAMEPARAM`` data length and ``FRAMEDATACOPY`` expose frame data."""
    sender = pre.deploy_contract(
        code=approve_bytecode(Spec.APPROVE_BOTH),
        balance=10**18,
    )

    code = (
        Op.SSTORE(SIZE_SLOT, Op.FRAMEPARAM(param=0x04, frame_index=frame_index))
        + Op.FRAMEDATACOPY(
            mem_offset=0, data_offset=0, length=32, frame_index=frame_index
        )
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
                nonce=2,
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
    """``FRAMEPARAM(0x05)`` returns status for previous frames."""
    sender = pre.deploy_contract(
        code=approve_bytecode(Spec.APPROVE_BOTH),
        balance=10**18,
    )

    status_reader = pre.deploy_contract(
        code=Op.SSTORE(
            STATUS_SLOT, Op.FRAMEPARAM(param=0x05, frame_index=0)
        )
        + Op.STOP
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
                nonce=2,
            ),
            status_reader: Account(storage={STATUS_SLOT: Spec.STATUS_SUCCESS}),
        },
    )


def test_txparam_status_current_frame_invalid(
    state_test: StateTestFiller,
    pre: Alloc,
) -> None:
    """``FRAMEPARAM(0x05)`` on the current frame exceptional-halts."""
    sender = pre.deploy_contract(
        code=approve_bytecode(Spec.APPROVE_BOTH),
        balance=10**18,
    )

    reader = pre.deploy_contract(
        code=Op.SSTORE(
            STATUS_SLOT, Op.FRAMEPARAM(param=0x05, frame_index=1)
        )
        + Op.STOP
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
    sender = pre.deploy_contract(
        code=approve_bytecode(Spec.APPROVE_BOTH),
        balance=10**18,
    )

    reader = pre.deploy_contract(
        code=Op.SSTORE(TXPARAM_SLOT, Op.TXPARAM(0x0B)) + Op.STOP
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
    sender = pre.deploy_contract(
        code=approve_bytecode(Spec.APPROVE_BOTH),
        balance=10**18,
    )

    reader = pre.deploy_contract(
        code=Op.SSTORE(
            TXPARAM_SLOT, Op.FRAMEPARAM(param=0x00, frame_index=5)
        )
        + Op.STOP
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
    sender = pre.deploy_contract(
        code=approve_bytecode(Spec.APPROVE_BOTH),
        balance=10**18,
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
            FrameReceipt(status=Spec.STATUS_SUCCESS),
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
                nonce=2,
            ),
        },
    )


def test_txparam_status_future_frame_invalid(
    state_test: StateTestFiller,
    pre: Alloc,
) -> None:
    """``FRAMEPARAM(0x05)`` on a future frame index halts the frame."""
    sender = pre.deploy_contract(
        code=approve_bytecode(Spec.APPROVE_BOTH),
        balance=10**18,
    )

    # Frame index 2 is the last frame; reading status of
    # index 2 from index 1 is a future frame.
    reader = pre.deploy_contract(
        code=(
            Op.SSTORE(
                STATUS_SLOT, Op.FRAMEPARAM(param=0x05, frame_index=2)
            )
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
    """
    ``APPROVE_PAYMENT`` cannot succeed inside a nested ``CALL`` when the active
    frame's resolved target differs from ``ADDRESS`` (EIP-8141). Sponsorship uses
    a dedicated ``VERIFY`` paymaster frame; the ``SENDER`` frame then performs an
    ordinary nested ``CALL``, which must still report success (status 1).
    """
    sender = pre.deploy_contract(
        code=approve_bytecode(Spec.APPROVE_EXECUTION),
        balance=10**18,
    )

    paymaster = pre.deploy_contract(
        code=approve_bytecode(Spec.APPROVE_PAYMENT),
        balance=10**18,
    )
    call_target = pre.deploy_contract(code=Op.STOP)

    outer_code = (
        Op.SSTORE(
            STATUS_SLOT,
            Op.CALL(
                gas=100_000,
                address=call_target,
                value=0,
                args_offset=0,
                args_size=0,
                ret_offset=0,
                ret_size=0,
            ),
        )
        + Op.STOP
    )
    outer = pre.deploy_contract(code=outer_code, balance=10**18)

    frames = [
        build_frame(
            mode=Spec.MODE_VERIFY,
            target=sender,
            gas_limit=100_000,
            data=b"",
            flags=Spec.APPROVE_EXECUTION,
        ),
        build_frame(
            mode=Spec.MODE_VERIFY,
            target=paymaster,
            gas_limit=100_000,
            data=b"",
            flags=Spec.APPROVE_PAYMENT,
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
                    STATUS_SLOT: Spec.STATUS_SUCCESS,
                },
            ),
        },
    )


def test_txparamcopy_out_of_bounds_offset(
    state_test: StateTestFiller,
    pre: Alloc,
) -> None:
    """``FRAMEDATACOPY`` with offset beyond data returns zero-padded."""
    sender = pre.deploy_contract(
        code=approve_bytecode(Spec.APPROVE_BOTH),
        balance=10**18,
    )

    frame_data = b"hello"
    code = (
        Op.FRAMEDATACOPY(
            mem_offset=0, data_offset=100, length=32, frame_index=1
        )
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
    """SENDER frame reading VERIFY frame data sees length 0 and zero bytes."""
    sender = pre.deploy_contract(
        code=approve_bytecode(Spec.APPROVE_BOTH),
        balance=10**18,
    )

    code = (
        Op.SSTORE(SIZE_SLOT, Op.FRAMEPARAM(param=0x04, frame_index=0))
        + Op.FRAMEDATACOPY(
            mem_offset=0, data_offset=0, length=32, frame_index=0
        )
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
