"""
Ethereum Virtual Machine (EVM) Interpreter.

.. contents:: Table of Contents
    :backlinks: none
    :local:

Introduction
------------

A straightforward interpreter that executes EVM code.
"""

from dataclasses import dataclass
from typing import List, Optional, Set, Tuple

from ethereum_types.bytes import Bytes, Bytes0
from ethereum_types.numeric import U64, U256, Uint, ulen

from ethereum.exceptions import EthereumException
from ethereum.trace import (
    EvmStop,
    OpEnd,
    OpException,
    OpStart,
    PrecompileEnd,
    PrecompileStart,
    TransactionEnd,
    evm_trace,
)

from ..blocks import Log
from ..exceptions import (
    FrameTransactionInvalidApprovalError,
    FrameTransactionInvalidFrameExecutionError,
)
from ..fork_types import Address
from ..state import (
    TransientStorage,
    account_has_code_or_nonce,
    account_has_storage,
    begin_transaction,
    commit_transaction,
    destroy_storage,
    get_account,
    increment_nonce,
    mark_account_created,
    move_ether,
    rollback_transaction,
    set_code,
)
from ..state_tracker import (
    capture_pre_balance,
    capture_pre_code,
    create_child_frame,
    merge_on_failure,
    merge_on_success,
    track_address,
    track_balance_change,
    track_code_change,
    track_nonce_change,
)
from ..transactions import ENTRY_POINT
from ..vm import Message
from ..vm.eoa_delegation import get_delegated_code_address, set_delegation
from ..vm.gas import GAS_CODE_DEPOSIT, charge_gas
from ..vm.precompiled_contracts.mapping import PRE_COMPILED_CONTRACTS
from . import Evm
from .exceptions import (
    AddressCollision,
    ExceptionalHalt,
    InvalidContractPrefix,
    InvalidOpcode,
    OutOfGasError,
    Revert,
    StackDepthLimitError,
)
from .instructions import Ops, op_implementation
from .runtime import get_valid_jump_destinations

STACK_DEPTH_LIMIT = Uint(1024)
MAX_CODE_SIZE = 0x6000
MAX_INIT_CODE_SIZE = 2 * MAX_CODE_SIZE


@dataclass
class MessageCallOutput:
    """
    Output of a particular message call.

    Contains the following:

          1. `gas_left`: remaining gas after execution.
          2. `refund_counter`: gas to refund after execution.
          3. `logs`: list of `Log` generated during execution.
          4. `accounts_to_delete`: Contracts which have self-destructed.
          5. `error`: The error from the execution if any.
          6. `return_data`: The output of the execution.
          7. `payer`: The payer selected by frame-transaction approvals.
          8. `frame_logs`: Per-frame logs for frame transactions.
    """

    gas_left: Uint
    refund_counter: U256
    logs: Tuple[Log, ...]
    accounts_to_delete: Set[Address]
    error: Optional[EthereumException]
    return_data: Bytes
    payer: Optional[Address] = None
    frame_logs: Tuple[Tuple[Log, ...], ...] = ()


def process_message_call(message: Message) -> MessageCallOutput:
    """
    If `message.target` is empty then it creates a smart contract
    else it executes a call from the `message.caller` to the `message.target`.

    Parameters
    ----------
    message :
        Transaction specific items.

    Returns
    -------
    output : `MessageCallOutput`
        Output of the message call

    """
    block_env = message.block_env
    refund_counter = U256(0)
    if message.target == Bytes0(b""):
        is_collision = account_has_code_or_nonce(
            block_env.state, message.current_target
        ) or account_has_storage(block_env.state, message.current_target)
        track_address(message.tx_env.state_changes, message.current_target)
        if is_collision:
            return MessageCallOutput(
                Uint(0),
                U256(0),
                tuple(),
                set(),
                AddressCollision(),
                Bytes(b""),
            )
        else:
            evm = process_create_message(message)
    else:
        if message.tx_env.authorizations != ():
            refund_counter += set_delegation(message)

        delegated_address = get_delegated_code_address(message.code)
        if delegated_address is not None:
            message.disable_precompiles = True
            message.tx_env.accessed_addresses.add(delegated_address)
            message.code = get_account(block_env.state, delegated_address).code
            message.code_address = delegated_address
            track_address(message.block_env.state_changes, delegated_address)

        evm = process_message(message)

    if evm.error:
        logs: Tuple[Log, ...] = ()
        accounts_to_delete = set()
    else:
        logs = evm.logs
        accounts_to_delete = evm.accounts_to_delete
        refund_counter += U256(evm.refund_counter)

    tx_end = TransactionEnd(
        int(message.gas) - int(evm.gas_left), evm.output, evm.error
    )
    evm_trace(evm, tx_end)

    return MessageCallOutput(
        gas_left=evm.gas_left,
        refund_counter=refund_counter,
        logs=logs,
        accounts_to_delete=accounts_to_delete,
        error=evm.error,
        return_data=evm.output,
    )


def process_abstract_call(message: Message) -> MessageCallOutput:
    """
    Execute a frame transaction using a single top-level message.

    The top-level frame message carries the frame payload in
    ``message.frames``. Individual frames are then executed as regular
    message calls.
    """
    frames = message.frames
    if frames is None:
        raise AssertionError("frame payload is required")

    frame_tx_sender = message.caller

    tx_env = message.tx_env
    tx_approval = tx_env.frame_tx_approval
    frame_statuses = tx_env.frame_statuses

    if tx_approval is None or frame_statuses is None:
        raise AssertionError("frame transaction context is required")

    tx_env.accessed_addresses.add(message.block_env.coinbase)
    tx_env.accessed_addresses.update(PRE_COMPILED_CONTRACTS.keys())

    total_gas_used = Uint(0)
    frame_logs: List[Tuple[Log, ...]] = []
    accounts_to_delete: Set[Address] = set()

    try:
        for frame_index, frame in enumerate(frames):
            if isinstance(frame.target, Bytes0) or frame.target == Bytes0(b""):
                target = frame_tx_sender
            else:
                target = Address(frame.target)

            if frame.mode == Uint(2):
                if not tx_approval.sender_approved:
                    raise FrameTransactionInvalidApprovalError(
                        "SENDER mode before execution approval"
                    )
                caller = frame_tx_sender
            else:
                caller = ENTRY_POINT

            tx_env.origin = caller
            tx_env.gas = frame.gas_limit
            tx_env.transient_storage = TransientStorage()
            tx_env.current_frame_index = frame_index

            tx_env.accessed_addresses.add(target)
            tx_env.accessed_addresses.add(caller)
            tx_env.accessed_addresses.add(frame_tx_sender)

            code = get_account(message.block_env.state, target).code

            call_frame = create_child_frame(tx_env.state_changes)
            frame_message = Message(
                block_env=message.block_env,
                tx_env=tx_env,
                caller=caller,
                target=target,
                current_target=target,
                gas=frame.gas_limit,
                value=U256(0),
                data=frame.data,
                code_address=target,
                code=code,
                depth=Uint(0),
                should_transfer_value=False,
                is_static=frame.mode == Uint(1),
                disable_precompiles=False,
                parent_evm=None,
                is_create=False,
                state_changes=call_frame,
            )

            sender_approved_before = tx_approval.sender_approved
            payer_approved_before = tx_approval.payer_approved
            payer_address_before = tx_approval.payer_address

            tx_approval.approve_called_in_frame = False
            frame_output = process_message_call(frame_message)
            accounts_to_delete.update(frame_output.accounts_to_delete)

            if frame_output.error is not None:
                tx_approval.sender_approved = sender_approved_before
                tx_approval.payer_approved = payer_approved_before
                tx_approval.payer_address = payer_address_before
                tx_approval.approve_called_in_frame = False

            frame_gas_used = frame.gas_limit - frame_output.gas_left
            total_gas_used += frame_gas_used

            frame_status = 1 if frame_output.error is None else 0

            if (
                frame.mode == Uint(1)
                and not tx_approval.approve_called_in_frame
            ):
                raise FrameTransactionInvalidFrameExecutionError(
                    "VERIFY frame must successfully call APPROVE"
                )

            frame_statuses.append(frame_status)
            frame_logs.append(frame_output.logs if frame_status == 1 else ())

        if not tx_approval.payer_approved:
            raise FrameTransactionInvalidFrameExecutionError(
                "payer_approved must be true after all frames"
            )
    finally:
        tx_env.current_frame_index = None

    frame_gas_sum = Uint(0)
    for frame in frames:
        frame_gas_sum += frame.gas_limit

    gas_left = frame_gas_sum - total_gas_used

    all_logs: Tuple[Log, ...] = ()
    for logs in frame_logs:
        all_logs += logs

    return MessageCallOutput(
        gas_left=gas_left,
        refund_counter=U256(0),
        logs=all_logs,
        accounts_to_delete=accounts_to_delete,
        error=None,
        return_data=Bytes(b""),
        payer=tx_approval.payer_address,
        frame_logs=tuple(frame_logs),
    )


def process_create_message(message: Message) -> Evm:
    """
    Executes a call to create a smart contract.

    Parameters
    ----------
    message :
        Transaction specific items.

    Returns
    -------
    evm: :py:class:`~ethereum.forks.bogota.vm.Evm`
        Items containing execution specific objects.

    """
    state = message.block_env.state
    transient_storage = message.tx_env.transient_storage
    # take snapshot of state before processing the message
    begin_transaction(state, transient_storage)

    # If the address where the account is being created has storage, it is
    # destroyed. This can only happen in the following highly unlikely
    # circumstances:
    # * The address created by a `CREATE` call collides with a subsequent
    #   `CREATE` or `CREATE2` call.
    # * The first `CREATE` happened before Spurious Dragon and left empty
    #   code.
    destroy_storage(state, message.current_target)

    # In the previously mentioned edge case the preexisting storage is ignored
    # for gas refund purposes. In order to do this we must track created
    # accounts. This tracking is also needed to respect the constraints
    # added to SELFDESTRUCT by EIP-6780.
    mark_account_created(state, message.current_target)

    increment_nonce(state, message.current_target)
    nonce_after = get_account(state, message.current_target).nonce
    track_nonce_change(
        message.state_changes,
        message.current_target,
        U64(nonce_after),
    )

    capture_pre_code(message.tx_env.state_changes, message.current_target, b"")

    evm = process_message(message)
    if not evm.error:
        contract_code = evm.output
        contract_code_gas = Uint(len(contract_code)) * GAS_CODE_DEPOSIT
        try:
            if len(contract_code) > 0:
                if contract_code[0] == 0xEF:
                    raise InvalidContractPrefix
            charge_gas(evm, contract_code_gas)
            if len(contract_code) > MAX_CODE_SIZE:
                raise OutOfGasError
        except ExceptionalHalt as error:
            rollback_transaction(state, transient_storage)
            merge_on_failure(message.state_changes)
            evm.gas_left = Uint(0)
            evm.output = b""
            evm.error = error
        else:
            # Note: No need to capture pre code since it's always b"" here
            set_code(state, message.current_target, contract_code)
            if contract_code != b"":
                track_code_change(
                    message.state_changes,
                    message.current_target,
                    contract_code,
                )
            commit_transaction(state, transient_storage)
            merge_on_success(message.state_changes)
    else:
        rollback_transaction(state, transient_storage)
        merge_on_failure(message.state_changes)
    return evm


def process_message(message: Message) -> Evm:
    """
    Move ether and execute the relevant code.

    Parameters
    ----------
    message :
        Transaction specific items.

    Returns
    -------
    evm: :py:class:`~ethereum.forks.bogota.vm.Evm`
        Items containing execution specific objects

    """
    state = message.block_env.state
    transient_storage = message.tx_env.transient_storage
    if message.depth > STACK_DEPTH_LIMIT:
        raise StackDepthLimitError("Stack depth limit reached")

    code = message.code
    valid_jump_destinations = get_valid_jump_destinations(code)
    evm = Evm(
        pc=Uint(0),
        stack=[],
        memory=bytearray(),
        code=code,
        gas_left=message.gas,
        valid_jump_destinations=valid_jump_destinations,
        logs=(),
        refund_counter=0,
        running=True,
        message=message,
        output=b"",
        accounts_to_delete=set(),
        return_data=b"",
        error=None,
        accessed_addresses=message.tx_env.accessed_addresses,
        accessed_storage_keys=message.tx_env.accessed_storage_keys,
        state_changes=message.state_changes,
    )

    # take snapshot of state before processing the message
    begin_transaction(state, transient_storage)

    track_address(message.state_changes, message.current_target)

    if message.should_transfer_value and message.value != 0:
        # Track value transfer
        sender_balance = get_account(state, message.caller).balance
        recipient_balance = get_account(state, message.current_target).balance

        track_address(message.state_changes, message.caller)
        capture_pre_balance(
            message.tx_env.state_changes, message.caller, sender_balance
        )
        capture_pre_balance(
            message.tx_env.state_changes,
            message.current_target,
            recipient_balance,
        )

        move_ether(
            state, message.caller, message.current_target, message.value
        )

        sender_new_balance = get_account(state, message.caller).balance
        recipient_new_balance = get_account(
            state, message.current_target
        ).balance

        track_balance_change(
            message.state_changes,
            message.caller,
            U256(sender_new_balance),
        )
        track_balance_change(
            message.state_changes,
            message.current_target,
            U256(recipient_new_balance),
        )

    try:
        if evm.message.code_address in PRE_COMPILED_CONTRACTS:
            if not message.disable_precompiles:
                evm_trace(evm, PrecompileStart(evm.message.code_address))
                PRE_COMPILED_CONTRACTS[evm.message.code_address](evm)
                evm_trace(evm, PrecompileEnd())
        else:
            while evm.running and evm.pc < ulen(evm.code):
                try:
                    op = Ops(evm.code[evm.pc])
                except ValueError as e:
                    raise InvalidOpcode(evm.code[evm.pc]) from e

                evm_trace(evm, OpStart(op))
                op_implementation[op](evm)
                evm_trace(evm, OpEnd())

            evm_trace(evm, EvmStop(Ops.STOP))

    except ExceptionalHalt as error:
        evm_trace(evm, OpException(error))
        evm.gas_left = Uint(0)
        evm.output = b""
        evm.error = error
    except Revert as error:
        evm_trace(evm, OpException(error))
        evm.error = error

    if evm.error:
        rollback_transaction(state, transient_storage)
        if not message.is_create:
            merge_on_failure(evm.state_changes)
    else:
        commit_transaction(state, transient_storage)
        if not message.is_create:
            merge_on_success(evm.state_changes)
    return evm
