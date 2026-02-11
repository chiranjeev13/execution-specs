"""Test frame execution ordering and approval flows for EIP-8141."""

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

from .helpers import approve_bytecode, build_frame, make_frame_tx
from .spec import Spec, ref_spec_8141

pytestmark = pytest.mark.valid_from("Bogota")

REFERENCE_SPEC_GIT_PATH = ref_spec_8141.git_path
REFERENCE_SPEC_VERSION = ref_spec_8141.version

SLOT_EXECUTED = 0x01


@pytest.mark.parametrize(
    "approval_scope",
    [Spec.APPROVE_BOTH],
)
def test_happy_path_self_paid(
    state_test: StateTestFiller,
    pre: Alloc,
    approval_scope: int,
) -> None:
    """Happy-path: sender approves execution/payment then executes a frame."""
    sender = pre.fund_eoa(amount=10**18)
    sender_code = approve_bytecode(approval_scope)
    pre.deploy_contract(code=sender_code, address=sender)

    execution_target = pre.deploy_contract(
        code=Op.SSTORE(SLOT_EXECUTED, 1) + Op.STOP
    )

    frames = [
        build_frame(
            mode=Spec.MODE_VERIFY,
            target=sender,
            gas_limit=50_000,
            data=b"",
        ),
        build_frame(
            mode=Spec.MODE_SENDER,
            target=execution_target,
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
            sender: Account(code=sender_code, nonce=1),
            execution_target: Account(storage={SLOT_EXECUTED: 1}),
        },
    )


def test_sponsored_transaction(
    state_test: StateTestFiller,
    pre: Alloc,
) -> None:
    """Sponsored: sender approves exec, sponsor approves payment."""
    sender = pre.fund_eoa(amount=10**18)
    sponsor = pre.fund_eoa(amount=10**18)

    sender_code = approve_bytecode(Spec.APPROVE_EXECUTION)
    sponsor_code = approve_bytecode(Spec.APPROVE_PAYMENT)

    pre.deploy_contract(code=sender_code, address=sender)
    pre.deploy_contract(code=sponsor_code, address=sponsor)

    execution_target = pre.deploy_contract(
        code=Op.SSTORE(SLOT_EXECUTED, 1) + Op.STOP
    )

    frames = [
        build_frame(
            mode=Spec.MODE_VERIFY,
            target=sender,
            gas_limit=50_000,
            data=b"",
        ),
        build_frame(
            mode=Spec.MODE_VERIFY,
            target=sponsor,
            gas_limit=50_000,
            data=b"",
        ),
        build_frame(
            mode=Spec.MODE_SENDER,
            target=execution_target,
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
        payer=sponsor,
        frame_receipts=[
            FrameReceipt(status=Spec.STATUS_SUCCESS),
            FrameReceipt(status=Spec.STATUS_SUCCESS),
            FrameReceipt(status=Spec.STATUS_SUCCESS),
        ],
    )

    state_test(
        env=Environment(),
        pre=pre,
        tx=tx,
        post={
            sender: Account(code=sender_code, nonce=1, balance=10**18),
            sponsor: Account(code=sponsor_code),
            execution_target: Account(storage={SLOT_EXECUTED: 1}),
        },
    )


@pytest.mark.exception_test
def test_invalid_payment_before_execution(
    state_test: StateTestFiller,
    pre: Alloc,
) -> None:
    """Payment approval before execution approval invalidates the tx."""
    sender = pre.fund_eoa(amount=10**18)
    sponsor = pre.fund_eoa(amount=10**18)
    pre.deploy_contract(
        code=approve_bytecode(Spec.APPROVE_EXECUTION), address=sender
    )
    pre.deploy_contract(
        code=approve_bytecode(Spec.APPROVE_PAYMENT), address=sponsor
    )

    frames = [
        build_frame(
            mode=Spec.MODE_VERIFY,
            target=sponsor,
            gas_limit=50_000,
            data=b"",
        ),
        build_frame(
            mode=Spec.MODE_VERIFY,
            target=sender,
            gas_limit=50_000,
            data=b"",
        ),
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
def test_invalid_sender_frame_before_approval(
    state_test: StateTestFiller,
    pre: Alloc,
) -> None:
    """SENDER frame before approval invalidates the tx."""
    sender = pre.fund_eoa(amount=10**18)
    target = pre.deploy_contract(code=Op.STOP)
    pre.deploy_contract(
        code=approve_bytecode(Spec.APPROVE_EXECUTION), address=sender
    )

    frames = [
        build_frame(
            mode=Spec.MODE_SENDER,
            target=target,
            gas_limit=50_000,
            data=b"",
        ),
        build_frame(
            mode=Spec.MODE_VERIFY,
            target=sender,
            gas_limit=50_000,
            data=b"",
        ),
    ]
    tx = make_frame_tx(sender=sender, frames=frames)
    tx.error = TransactionException.TYPE_6_INVALID_APPROVAL

    state_test(
        env=Environment(),
        pre=pre,
        tx=tx,
        post={},
    )


@pytest.mark.exception_test
def test_duplicate_execution_approval_invalid(
    state_test: StateTestFiller,
    pre: Alloc,
) -> None:
    """Duplicate execution approvals in VERIFY frames revert the tx."""
    sender = pre.fund_eoa(amount=10**18)
    pre.deploy_contract(
        code=approve_bytecode(Spec.APPROVE_EXECUTION), address=sender
    )

    frames = [
        build_frame(
            mode=Spec.MODE_VERIFY,
            target=sender,
            gas_limit=50_000,
            data=b"",
        ),
        build_frame(
            mode=Spec.MODE_VERIFY,
            target=sender,
            gas_limit=50_000,
            data=b"",
        ),
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
def test_duplicate_payment_approval_invalid(
    state_test: StateTestFiller,
    pre: Alloc,
) -> None:
    """Duplicate payment approvals revert the tx."""
    sender = pre.fund_eoa(amount=10**18)
    sponsor = pre.fund_eoa(amount=10**18)
    pre.deploy_contract(
        code=approve_bytecode(Spec.APPROVE_EXECUTION), address=sender
    )
    pre.deploy_contract(
        code=approve_bytecode(Spec.APPROVE_PAYMENT), address=sponsor
    )

    frames = [
        build_frame(
            mode=Spec.MODE_VERIFY,
            target=sender,
            gas_limit=50_000,
            data=b"",
        ),
        build_frame(
            mode=Spec.MODE_VERIFY,
            target=sponsor,
            gas_limit=50_000,
            data=b"",
        ),
        build_frame(
            mode=Spec.MODE_VERIFY,
            target=sponsor,
            gas_limit=50_000,
            data=b"",
        ),
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
def test_status_both_after_sender_approved_invalid(
    state_test: StateTestFiller,
    pre: Alloc,
) -> None:
    """APPROVE_BOTH after sender approval should revert the frame."""
    sender = pre.fund_eoa(amount=10**18)
    sender_code = Conditional(
        condition=Op.ISZERO(Op.CALLDATALOAD(0)),
        if_true=approve_bytecode(Spec.APPROVE_EXECUTION),
        if_false=approve_bytecode(Spec.APPROVE_BOTH),
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
            mode=Spec.MODE_VERIFY,
            target=sender,
            gas_limit=50_000,
            data=b"\x01",
        ),
    ]
    tx = make_frame_tx(sender=sender, frames=frames)
    tx.error = TransactionException.TYPE_6_INVALID_FRAME_EXECUTION

    state_test(
        env=Environment(),
        pre=pre,
        tx=tx,
        post={},
    )


def test_status_both_when_neither_approved(
    state_test: StateTestFiller,
    pre: Alloc,
) -> None:
    """APPROVE_BOTH when neither approval is set should succeed."""
    sender = pre.fund_eoa(amount=10**18)
    pre.deploy_contract(
        code=approve_bytecode(Spec.APPROVE_BOTH), address=sender
    )

    execution_target = pre.deploy_contract(
        code=Op.SSTORE(SLOT_EXECUTED, 1) + Op.STOP
    )

    frames = [
        build_frame(
            mode=Spec.MODE_VERIFY,
            target=sender,
            gas_limit=50_000,
            data=b"",
        ),
        build_frame(
            mode=Spec.MODE_SENDER,
            target=execution_target,
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
                nonce=1,
            ),
            execution_target: Account(storage={SLOT_EXECUTED: 1}),
        },
    )


@pytest.mark.parametrize(
    "verify_code,expected_error",
    [
        pytest.param(
            Op.STOP,
            TransactionException.TYPE_6_INVALID_FRAME_EXECUTION,
            id="stop",
        ),
        pytest.param(
            Op.RETURN(0, 0),
            TransactionException.TYPE_6_INVALID_FRAME_EXECUTION,
            id="return",
        ),
        pytest.param(
            Op.REVERT(0, 0),
            TransactionException.TYPE_6_INVALID_FRAME_EXECUTION,
            id="revert",
        ),
        pytest.param(
            Op.INVALID,
            TransactionException.TYPE_6_INVALID_FRAME_EXECUTION,
            id="invalid",
        ),
        pytest.param(
            Op.SSTORE(0, 1),
            TransactionException.TYPE_6_INVALID_FRAME_EXECUTION,
            id="sstore",
        ),
    ],
)
@pytest.mark.exception_test
def test_verify_frame_must_approve(
    state_test: StateTestFiller,
    pre: Alloc,
    verify_code: Bytecode,
    expected_error: TransactionException,
) -> None:
    """VERIFY frames that do not APPROVE or that modify state are invalid."""
    sender = pre.fund_eoa(amount=10**18)
    pre.deploy_contract(code=verify_code, address=sender)

    frames = [
        build_frame(
            mode=Spec.MODE_VERIFY,
            target=sender,
            gas_limit=50_000,
            data=b"",
        )
    ]
    tx = make_frame_tx(sender=sender, frames=frames)
    tx.error = expected_error

    state_test(
        env=Environment(),
        pre=pre,
        tx=tx,
        post={},
    )


@pytest.mark.exception_test
def test_verify_frame_oog_invalid(
    state_test: StateTestFiller,
    pre: Alloc,
) -> None:
    """VERIFY frame that runs out of gas (no APPROVE) invalidates the tx."""
    sender = pre.fund_eoa(amount=10**18)
    # Deploy code that burns gas in a loop until OOG, never reaching APPROVE
    oog_code = Op.JUMPDEST + Op.JUMP(0)
    pre.deploy_contract(code=oog_code, address=sender)

    frames = [
        build_frame(
            mode=Spec.MODE_VERIFY,
            target=sender,
            gas_limit=10_000,
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


def test_sponsored_balance_and_nonce(
    state_test: StateTestFiller,
    pre: Alloc,
) -> None:
    """Sponsor balance is deducted; payer nonce is NOT modified."""
    initial = 10**18
    sender = pre.fund_eoa(amount=initial)
    sponsor = pre.fund_eoa(amount=initial)

    pre.deploy_contract(
        code=approve_bytecode(Spec.APPROVE_EXECUTION),
        address=sender,
    )
    pre.deploy_contract(
        code=approve_bytecode(Spec.APPROVE_PAYMENT),
        address=sponsor,
    )

    target = pre.deploy_contract(code=Op.STOP)

    base_fee = 10
    frames = [
        build_frame(
            mode=Spec.MODE_VERIFY,
            target=sender,
            gas_limit=50_000,
            data=b"",
        ),
        build_frame(
            mode=Spec.MODE_VERIFY,
            target=sponsor,
            gas_limit=50_000,
            data=b"",
        ),
        build_frame(
            mode=Spec.MODE_SENDER,
            target=target,
            gas_limit=50_000,
            data=b"",
        ),
    ]
    tx = make_frame_tx(
        sender=sender,
        frames=frames,
        max_fee_per_gas=base_fee,
        max_priority_fee_per_gas=0,
    )

    state_test(
        env=Environment(base_fee_per_gas=base_fee),
        pre=pre,
        tx=tx,
        post={
            # Sender nonce increments once.
            sender: Account(
                nonce=1,
                balance=initial,
            ),
            # Sponsor pays gas but nonce does NOT change.
            sponsor: Account(
                nonce=0,
            ),
        },
    )


@pytest.mark.exception_test
def test_verify_frame_value_transfer_invalid(
    state_test: StateTestFiller,
    pre: Alloc,
) -> None:
    """VERIFY frame that performs a value transfer halts the tx."""
    sender = pre.fund_eoa(amount=10**18)
    # Code that sends 1 wei then APPROVEs — the CALL with
    # value should trigger static-mode exceptional halt.
    receiver = Address(0xBEEF)
    verify_code = Op.CALL(
        gas=10_000,
        address=receiver,
        value=1,
        args_offset=0,
        args_size=0,
        ret_offset=0,
        ret_size=0,
    ) + approve_bytecode(Spec.APPROVE_BOTH)
    pre.deploy_contract(code=verify_code, address=sender)

    frames = [
        build_frame(
            mode=Spec.MODE_VERIFY,
            target=sender,
            gas_limit=100_000,
            data=b"",
        ),
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
def test_verify_frame_create_invalid(
    state_test: StateTestFiller,
    pre: Alloc,
) -> None:
    """VERIFY frame that executes CREATE halts the tx."""
    sender = pre.fund_eoa(amount=10**18)
    # CREATE in static context → exceptional halt
    verify_code = Op.CREATE(0, 0, 0) + approve_bytecode(Spec.APPROVE_BOTH)
    pre.deploy_contract(code=verify_code, address=sender)

    frames = [
        build_frame(
            mode=Spec.MODE_VERIFY,
            target=sender,
            gas_limit=100_000,
            data=b"",
        ),
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
def test_payment_revert_atomicity(
    state_test: StateTestFiller,
    pre: Alloc,
) -> None:
    """
    Payment not processed if frame reverts.

    Scenario: sender approves execution, sponsor frame REVERTs
    (so payment approval fails). A second sponsor then approves
    payment successfully. Transaction should succeed with the
    second sponsor as payer.
    """
    initial = 10**18
    sender = pre.fund_eoa(amount=initial)
    bad_sponsor = pre.fund_eoa(amount=initial)
    good_sponsor = pre.fund_eoa(amount=initial)

    pre.deploy_contract(
        code=approve_bytecode(Spec.APPROVE_EXECUTION),
        address=sender,
    )
    # bad_sponsor REVERTs instead of approving
    pre.deploy_contract(
        code=Op.REVERT(0, 0),
        address=bad_sponsor,
    )
    pre.deploy_contract(
        code=approve_bytecode(Spec.APPROVE_PAYMENT),
        address=good_sponsor,
    )

    target = pre.deploy_contract(
        code=Op.SSTORE(SLOT_EXECUTED, 1) + Op.STOP,
    )

    frames = [
        build_frame(
            mode=Spec.MODE_VERIFY,
            target=sender,
            gas_limit=50_000,
            data=b"",
        ),
        # bad_sponsor VERIFY frame reverts — payment not processed
        build_frame(
            mode=Spec.MODE_VERIFY,
            target=bad_sponsor,
            gas_limit=50_000,
            data=b"",
        ),
        # good_sponsor approves payment
        build_frame(
            mode=Spec.MODE_VERIFY,
            target=good_sponsor,
            gas_limit=50_000,
            data=b"",
        ),
        build_frame(
            mode=Spec.MODE_SENDER,
            target=target,
            gas_limit=50_000,
            data=b"",
        ),
    ]

    # bad_sponsor's VERIFY frame reverts, so the tx should be
    # invalid because the VERIFY frame didn't APPROVE.
    tx = make_frame_tx(sender=sender, frames=frames)
    tx.error = TransactionException.TYPE_6_INVALID_FRAME_EXECUTION

    state_test(
        env=Environment(),
        pre=pre,
        tx=tx,
        post={},
    )
