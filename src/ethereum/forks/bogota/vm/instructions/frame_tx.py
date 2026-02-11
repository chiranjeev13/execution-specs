"""
Ethereum Virtual Machine (EVM) Frame Transaction Instructions.

.. contents:: Table of Contents
    :backlinks: none
    :local:

Introduction
------------

Implementations of the EVM frame transaction instructions defined in
[EIP-8141].

[EIP-8141]: https://eips.ethereum.org/EIPS/eip-8141
"""

from ethereum_types.bytes import Bytes, Bytes0
from ethereum_types.numeric import U64, U256, Uint

from ...state import get_account, increment_nonce, set_account_balance
from ...state_tracker import (
    capture_pre_balance,
    track_address,
    track_balance_change,
    track_nonce_change,
)
from ...transactions import (
    FrameTransaction,
    signing_hash_8141,
)
from .. import Evm, FrameTxApprovalContext
from ..exceptions import ExceptionalHalt, Revert
from ..gas import (
    GAS_BASE,
    GAS_VERY_LOW,
    GAS_ZERO,
    calculate_gas_extend_memory,
    charge_gas,
)
from ..memory import memory_write
from ..stack import pop, push


class FrameTxNotActiveError(ExceptionalHalt):
    """Raised when frame tx opcodes are used outside a frame transaction."""


class InvalidApproveScope(ExceptionalHalt):
    """Raised when APPROVE is called with an invalid scope."""


class InvalidTxParamSelector(ExceptionalHalt):
    """Raised when TXPARAM* is called with an invalid selector."""


class TxParamOutOfBounds(ExceptionalHalt):
    """Raised when TXPARAM* frame index is out of bounds."""


def _get_frame_tx(evm: Evm) -> FrameTransaction:
    """Get the active frame transaction or raise."""
    tx = evm.message.tx_env.frame_tx
    if tx is None or not isinstance(tx, FrameTransaction):
        raise FrameTxNotActiveError
    return tx


def _get_frame_tx_approval(evm: Evm) -> FrameTxApprovalContext:
    """Get the active frame transaction approval context or raise."""
    frame_tx_approval = evm.message.tx_env.frame_tx_approval
    if frame_tx_approval is None:
        raise FrameTxNotActiveError
    return frame_tx_approval


def _get_current_frame_target(evm: Evm, tx: FrameTransaction):
    """Resolve the target address of the currently executing frame."""
    frame_idx = evm.message.tx_env.current_frame_index
    if frame_idx is None:
        raise FrameTxNotActiveError
    if frame_idx >= len(tx.frames):
        raise FrameTxNotActiveError

    frame = tx.frames[frame_idx]
    if isinstance(frame.target, Bytes0) or frame.target == Bytes0(b""):
        return tx.sender
    return frame.target


def _revert_frame(evm: Evm) -> None:
    """Revert the current frame with empty output."""
    evm.output = Bytes(b"")
    raise Revert


def _get_txparam_value(
    evm: Evm,
    selector: int,
    index: int,
) -> tuple:
    """
    Get the value and size for a TXPARAM selector.

    Returns (value_bytes, size) where value_bytes is the raw bytes of the
    parameter value.
    """
    tx = _get_frame_tx(evm)
    tx_env = evm.message.tx_env
    block_env = evm.message.block_env

    if selector == 0x00:
        # Transaction type
        return (U256(0x06).to_be_bytes32(), Uint(32))
    elif selector == 0x01:
        # Nonce
        return (U256(tx.nonce).to_be_bytes32(), Uint(32))
    elif selector == 0x02:
        # Sender (left-padded to 32 bytes)
        padded = b"\x00" * 12 + bytes(tx.sender)
        return (padded, Uint(32))
    elif selector == 0x03:
        # max_priority_fee_per_gas
        return (U256(tx.max_priority_fee_per_gas).to_be_bytes32(), Uint(32))
    elif selector == 0x04:
        # max_fee_per_gas
        return (U256(tx.max_fee_per_gas).to_be_bytes32(), Uint(32))
    elif selector == 0x05:
        # max_fee_per_blob_gas
        return (U256(tx.max_fee_per_blob_gas).to_be_bytes32(), Uint(32))
    elif selector == 0x06:
        # max cost
        from ...transactions import calculate_frame_tx_intrinsic_cost
        from ..gas import (
            GAS_PER_BLOB,
            calculate_blob_gas_price,
        )

        tx_gas_limit = calculate_frame_tx_intrinsic_cost(tx)
        effective_gas_price = tx_env.gas_price
        blob_count = len(tx.blob_versioned_hashes)
        blob_gas_price = calculate_blob_gas_price(block_env.excess_blob_gas)
        blob_fees = (
            Uint(blob_count) * Uint(GAS_PER_BLOB) * Uint(blob_gas_price)
        )
        max_cost_val = (
            Uint(tx_gas_limit) * Uint(effective_gas_price) + blob_fees
        )
        return (U256(max_cost_val).to_be_bytes32(), Uint(32))
    elif selector == 0x07:
        # len(blob_versioned_hashes)
        return (
            U256(len(tx.blob_versioned_hashes)).to_be_bytes32(),
            Uint(32),
        )
    elif selector == 0x08:
        # compute_sig_hash(tx)
        sig_hash = signing_hash_8141(tx)
        return (bytes(sig_hash), Uint(32))
    elif selector == 0x09:
        # len(frames)
        return (U256(len(tx.frames)).to_be_bytes32(), Uint(32))
    elif selector == 0x10:
        # current frame index
        frame_idx = tx_env.current_frame_index
        if frame_idx is None:
            raise FrameTxNotActiveError
        return (U256(frame_idx).to_be_bytes32(), Uint(32))
    elif selector == 0x11:
        # frame[index].target
        if index >= len(tx.frames):
            raise TxParamOutOfBounds
        frame = tx.frames[index]

        if isinstance(frame.target, Bytes0) or frame.target == Bytes0(b""):
            padded = b"\x00" * 32
        else:
            padded = b"\x00" * 12 + bytes(frame.target)
        return (padded, Uint(32))
    elif selector == 0x12:
        # frame[index].data (elided for VERIFY)
        if index >= len(tx.frames):
            raise TxParamOutOfBounds
        frame = tx.frames[index]
        if frame.mode == Uint(1):  # VERIFY
            return (b"", Uint(0))
        data = bytes(frame.data)
        return (data, Uint(len(data)))
    elif selector == 0x13:
        # frame[index].gas_limit
        if index >= len(tx.frames):
            raise TxParamOutOfBounds
        return (
            U256(tx.frames[index].gas_limit).to_be_bytes32(),
            Uint(32),
        )
    elif selector == 0x14:
        # frame[index].mode
        if index >= len(tx.frames):
            raise TxParamOutOfBounds
        return (U256(tx.frames[index].mode).to_be_bytes32(), Uint(32))
    elif selector == 0x15:
        # frame[index].status (0=failure, 1=success; only for past frames)
        if index >= len(tx.frames):
            raise TxParamOutOfBounds
        frame_idx = tx_env.current_frame_index
        if frame_idx is None:
            raise FrameTxNotActiveError
        if index >= frame_idx:
            raise TxParamOutOfBounds  # current/future → exceptional halt
        statuses = tx_env.frame_statuses
        if statuses is None or index >= len(statuses):
            raise TxParamOutOfBounds
        return (U256(statuses[index]).to_be_bytes32(), Uint(32))
    else:
        raise InvalidTxParamSelector


def approve(evm: Evm) -> None:
    """
    ``APPROVE`` opcode (``0xAA``).

    Like ``RETURN`` but with a scope operand that updates transaction-scoped
    approval state.

    Stack (top first): ``offset``, ``length``, ``scope``.

    Parameters
    ----------
    evm :
        The current EVM frame.

    """
    # STACK
    offset = pop(evm.stack)
    length = pop(evm.stack)
    scope = pop(evm.stack)

    # GAS
    extend_memory = calculate_gas_extend_memory(evm.memory, [(offset, length)])
    charge_gas(evm, GAS_ZERO + extend_memory.cost)

    # OPERATION
    tx = _get_frame_tx(evm)
    tx_approval = _get_frame_tx_approval(evm)

    if scope > U256(2):
        raise InvalidApproveScope

    frame_target = _get_current_frame_target(evm, tx)
    if evm.message.caller != frame_target:
        _revert_frame(evm)

    state = evm.message.block_env.state
    tx_state_changes = evm.message.tx_env.state_changes
    frame_state_changes = evm.state_changes

    if scope == U256(0):
        if tx_approval.sender_approved:
            _revert_frame(evm)
        if evm.message.caller != tx.sender:
            _revert_frame(evm)
        tx_approval.sender_approved = True

    elif scope == U256(1):
        if tx_approval.payer_approved:
            _revert_frame(evm)
        if not tx_approval.sender_approved:
            _revert_frame(evm)

        payer_balance = get_account(state, frame_target).balance
        if Uint(payer_balance) < tx_approval.tx_fee:
            _revert_frame(evm)

        track_address(frame_state_changes, tx.sender)
        increment_nonce(state, tx.sender)
        sender_nonce_after = get_account(state, tx.sender).nonce
        track_nonce_change(
            frame_state_changes,
            tx.sender,
            U64(sender_nonce_after),
        )

        track_address(frame_state_changes, frame_target)
        capture_pre_balance(tx_state_changes, frame_target, payer_balance)
        payer_balance_after = U256(Uint(payer_balance) - tx_approval.tx_fee)
        set_account_balance(state, frame_target, payer_balance_after)
        track_balance_change(
            frame_state_changes,
            frame_target,
            payer_balance_after,
        )

        tx_approval.payer_approved = True
        tx_approval.payer_address = frame_target

    else:  # scope == 2
        if tx_approval.sender_approved or tx_approval.payer_approved:
            _revert_frame(evm)
        if evm.message.caller != tx.sender:
            _revert_frame(evm)

        payer_balance = get_account(state, frame_target).balance
        if Uint(payer_balance) < tx_approval.tx_fee:
            _revert_frame(evm)

        tx_approval.sender_approved = True

        track_address(frame_state_changes, tx.sender)
        increment_nonce(state, tx.sender)
        sender_nonce_after = get_account(state, tx.sender).nonce
        track_nonce_change(
            frame_state_changes,
            tx.sender,
            U64(sender_nonce_after),
        )

        track_address(frame_state_changes, frame_target)
        capture_pre_balance(tx_state_changes, frame_target, payer_balance)
        payer_balance_after = U256(Uint(payer_balance) - tx_approval.tx_fee)
        set_account_balance(state, frame_target, payer_balance_after)
        track_balance_change(
            frame_state_changes,
            frame_target,
            payer_balance_after,
        )

        tx_approval.payer_approved = True
        tx_approval.payer_address = frame_target

    evm.memory += b"\x00" * extend_memory.expand_by
    evm.output = Bytes(bytes(evm.memory[offset : offset + length]))
    tx_approval.approve_called_in_frame = True
    evm.running = False

    # PROGRAM COUNTER
    pass


def txparamload(evm: Evm) -> None:
    """
    ``TXPARAMLOAD`` opcode (``0xB0``).

    Load a 32-byte word from a transaction parameter and push it onto
    the stack.

    Stack: [selector, index, offset] → [value]

    Parameters
    ----------
    evm :
        The current EVM frame.

    """
    # STACK
    selector = pop(evm.stack)
    index = pop(evm.stack)
    offset = pop(evm.stack)

    # GAS
    charge_gas(evm, GAS_VERY_LOW)

    # OPERATION
    value_bytes, size = _get_txparam_value(evm, int(selector), int(index))

    # Read 32 bytes starting at offset, zero-padding beyond bounds
    off = int(offset)
    result = bytearray(32)
    for i in range(32):
        idx = off + i
        if idx < len(value_bytes):
            result[i] = value_bytes[idx]
        # else: remains 0 (zero-padded)

    push(evm.stack, U256.from_be_bytes(bytes(result)))

    # PROGRAM COUNTER
    evm.pc += Uint(1)


def txparamsize(evm: Evm) -> None:
    """
    ``TXPARAMSIZE`` opcode (``0xB1``).

    Push the size of a transaction parameter onto the stack.

    Stack: [in1, in2] → [size]

    Parameters
    ----------
    evm :
        The current EVM frame.

    """
    # STACK
    in1 = pop(evm.stack)
    in2 = pop(evm.stack)

    # GAS
    charge_gas(evm, GAS_BASE)

    # OPERATION
    _, size = _get_txparam_value(evm, int(in1), int(in2))

    push(evm.stack, U256(size))

    # PROGRAM COUNTER
    evm.pc += Uint(1)


def txparamcopy(evm: Evm) -> None:
    """
    ``TXPARAMCOPY`` opcode (``0xB2``).

    Copy transaction parameter data to memory.

    Stack: [in1, in2, dest_offset, src_offset, length] → []

    Parameters
    ----------
    evm :
        The current EVM frame.

    """
    # STACK
    in1 = pop(evm.stack)
    in2 = pop(evm.stack)
    dest_offset = pop(evm.stack)
    src_offset = pop(evm.stack)
    length = pop(evm.stack)

    # GAS
    extend_memory = calculate_gas_extend_memory(
        evm.memory, [(dest_offset, length)]
    )
    words = Uint(length + U256(31)) // Uint(32)
    copy_gas = GAS_VERY_LOW + GAS_VERY_LOW * words
    charge_gas(evm, copy_gas + extend_memory.cost)

    # Extend memory
    evm.memory += b"\x00" * extend_memory.expand_by

    # OPERATION
    value_bytes, _ = _get_txparam_value(evm, int(in1), int(in2))

    # Extract the requested slice, zero-padding beyond bounds
    src_off = int(src_offset)
    copy_len = int(length)
    result = bytearray(copy_len)
    for i in range(copy_len):
        idx = src_off + i
        if idx < len(value_bytes):
            result[i] = value_bytes[idx]
        # else: remains 0 (zero-padded)

    memory_write(evm.memory, dest_offset, Bytes(bytes(result)))

    # PROGRAM COUNTER
    evm.pc += Uint(1)
