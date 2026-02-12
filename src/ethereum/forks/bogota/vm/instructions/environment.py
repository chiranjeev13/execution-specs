"""
Ethereum Virtual Machine (EVM) Environmental Instructions.

.. contents:: Table of Contents
    :backlinks: none
    :local:

Introduction
------------

Implementations of the EVM environment related instructions.
"""

from ethereum_types.bytes import Bytes, Bytes0, Bytes32
from ethereum_types.numeric import U256, Uint, ulen

from ethereum.crypto.hash import keccak256
from ethereum.utils.numeric import ceil32

# track_address_access removed - now using state_changes.track_address()
from ...fork_types import EMPTY_ACCOUNT
from ...state import get_account
from ...state_tracker import track_address
from ...transactions import (
    FrameTransaction,
    calculate_frame_tx_intrinsic_cost,
    signing_hash_8141,
)
from ...utils.address import to_address_masked
from ...vm.memory import buffer_read, memory_write
from .. import Evm
from ..exceptions import ExceptionalHalt, OutOfBoundsRead
from ..gas import (
    GAS_BASE,
    GAS_BLOBHASH_OPCODE,
    GAS_COLD_ACCOUNT_ACCESS,
    GAS_COPY,
    GAS_FAST_STEP,
    GAS_PER_BLOB,
    GAS_RETURN_DATA_COPY,
    GAS_VERY_LOW,
    GAS_WARM_ACCESS,
    calculate_blob_gas_price,
    calculate_gas_extend_memory,
    charge_gas,
)
from ..stack import pop, push


class FrameTxNotActiveError(ExceptionalHalt):
    """Raised when frame tx opcodes are used outside a frame transaction."""


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


def address(evm: Evm) -> None:
    """
    Pushes the address of the current executing account to the stack.

    Parameters
    ----------
    evm :
        The current EVM frame.

    """
    # STACK
    pass

    # GAS
    charge_gas(evm, GAS_BASE)

    # OPERATION
    push(evm.stack, U256.from_be_bytes(evm.message.current_target))

    # PROGRAM COUNTER
    evm.pc += Uint(1)


def balance(evm: Evm) -> None:
    """
    Pushes the balance of the given account onto the stack.

    Parameters
    ----------
    evm :
        The current EVM frame.

    """
    # STACK
    address = to_address_masked(pop(evm.stack))

    # GAS
    is_cold_access = address not in evm.accessed_addresses
    gas_cost = GAS_COLD_ACCOUNT_ACCESS if is_cold_access else GAS_WARM_ACCESS
    if is_cold_access:
        evm.accessed_addresses.add(address)

    charge_gas(evm, gas_cost)

    # OPERATION
    # Non-existent accounts default to EMPTY_ACCOUNT, which has balance 0.
    state = evm.message.block_env.state
    balance = get_account(state, address).balance
    track_address(evm.state_changes, address)

    push(evm.stack, balance)

    # PROGRAM COUNTER
    evm.pc += Uint(1)


def origin(evm: Evm) -> None:
    """
    Pushes the address of the original transaction sender to the stack.
    The origin address can only be an EOA.

    Parameters
    ----------
    evm :
        The current EVM frame.

    """
    # STACK
    pass

    # GAS
    charge_gas(evm, GAS_BASE)

    # OPERATION
    push(evm.stack, U256.from_be_bytes(evm.message.tx_env.origin))

    # PROGRAM COUNTER
    evm.pc += Uint(1)


def caller(evm: Evm) -> None:
    """
    Pushes the address of the caller onto the stack.

    Parameters
    ----------
    evm :
        The current EVM frame.

    """
    # STACK
    pass

    # GAS
    charge_gas(evm, GAS_BASE)

    # OPERATION
    push(evm.stack, U256.from_be_bytes(evm.message.caller))

    # PROGRAM COUNTER
    evm.pc += Uint(1)


def callvalue(evm: Evm) -> None:
    """
    Push the value (in wei) sent with the call onto the stack.

    Parameters
    ----------
    evm :
        The current EVM frame.

    """
    # STACK
    pass

    # GAS
    charge_gas(evm, GAS_BASE)

    # OPERATION
    push(evm.stack, evm.message.value)

    # PROGRAM COUNTER
    evm.pc += Uint(1)


def calldataload(evm: Evm) -> None:
    """
    Push a word (32 bytes) of the input data belonging to the current
    environment onto the stack.

    Parameters
    ----------
    evm :
        The current EVM frame.

    """
    # STACK
    start_index = pop(evm.stack)

    # GAS
    charge_gas(evm, GAS_VERY_LOW)

    # OPERATION
    value = buffer_read(evm.message.data, start_index, U256(32))

    push(evm.stack, U256.from_be_bytes(value))

    # PROGRAM COUNTER
    evm.pc += Uint(1)


def calldatasize(evm: Evm) -> None:
    """
    Push the size of input data in current environment onto the stack.

    Parameters
    ----------
    evm :
        The current EVM frame.

    """
    # STACK
    pass

    # GAS
    charge_gas(evm, GAS_BASE)

    # OPERATION
    push(evm.stack, U256(len(evm.message.data)))

    # PROGRAM COUNTER
    evm.pc += Uint(1)


def calldatacopy(evm: Evm) -> None:
    """
    Copy a portion of the input data in current environment to memory.

    This will also expand the memory, in case that the memory is insufficient
    to store the data.

    Parameters
    ----------
    evm :
        The current EVM frame.

    """
    # STACK
    memory_start_index = pop(evm.stack)
    data_start_index = pop(evm.stack)
    size = pop(evm.stack)

    # GAS
    words = ceil32(Uint(size)) // Uint(32)
    copy_gas_cost = GAS_COPY * words
    extend_memory = calculate_gas_extend_memory(
        evm.memory, [(memory_start_index, size)]
    )
    charge_gas(evm, GAS_VERY_LOW + copy_gas_cost + extend_memory.cost)

    # OPERATION
    evm.memory += b"\x00" * extend_memory.expand_by
    value = buffer_read(evm.message.data, data_start_index, size)
    memory_write(evm.memory, memory_start_index, value)

    # PROGRAM COUNTER
    evm.pc += Uint(1)


def codesize(evm: Evm) -> None:
    """
    Push the size of code running in current environment onto the stack.

    Parameters
    ----------
    evm :
        The current EVM frame.

    """
    # STACK
    pass

    # GAS
    charge_gas(evm, GAS_BASE)

    # OPERATION
    push(evm.stack, U256(len(evm.code)))

    # PROGRAM COUNTER
    evm.pc += Uint(1)


def codecopy(evm: Evm) -> None:
    """
    Copy a portion of the code in current environment to memory.

    This will also expand the memory, in case that the memory is insufficient
    to store the data.

    Parameters
    ----------
    evm :
        The current EVM frame.

    """
    # STACK
    memory_start_index = pop(evm.stack)
    code_start_index = pop(evm.stack)
    size = pop(evm.stack)

    # GAS
    words = ceil32(Uint(size)) // Uint(32)
    copy_gas_cost = GAS_COPY * words
    extend_memory = calculate_gas_extend_memory(
        evm.memory, [(memory_start_index, size)]
    )
    charge_gas(evm, GAS_VERY_LOW + copy_gas_cost + extend_memory.cost)

    # OPERATION
    evm.memory += b"\x00" * extend_memory.expand_by
    value = buffer_read(evm.code, code_start_index, size)
    memory_write(evm.memory, memory_start_index, value)

    # PROGRAM COUNTER
    evm.pc += Uint(1)


def gasprice(evm: Evm) -> None:
    """
    Push the gas price used in current environment onto the stack.

    Parameters
    ----------
    evm :
        The current EVM frame.

    """
    # STACK
    pass

    # GAS
    charge_gas(evm, GAS_BASE)

    # OPERATION
    push(evm.stack, U256(evm.message.tx_env.gas_price))

    # PROGRAM COUNTER
    evm.pc += Uint(1)


def extcodesize(evm: Evm) -> None:
    """
    Push the code size of a given account onto the stack.

    Parameters
    ----------
    evm :
        The current EVM frame.

    """
    # STACK
    address = to_address_masked(pop(evm.stack))

    # GAS
    is_cold_access = address not in evm.accessed_addresses
    access_gas_cost = (
        GAS_COLD_ACCOUNT_ACCESS if is_cold_access else GAS_WARM_ACCESS
    )
    if is_cold_access:
        evm.accessed_addresses.add(address)

    charge_gas(evm, access_gas_cost)

    # OPERATION
    state = evm.message.block_env.state
    code = get_account(state, address).code
    track_address(evm.state_changes, address)

    codesize = U256(len(code))
    push(evm.stack, codesize)

    # PROGRAM COUNTER
    evm.pc += Uint(1)


def extcodecopy(evm: Evm) -> None:
    """
    Copy a portion of an account's code to memory.

    Parameters
    ----------
    evm :
        The current EVM frame.

    """
    # STACK
    address = to_address_masked(pop(evm.stack))
    memory_start_index = pop(evm.stack)
    code_start_index = pop(evm.stack)
    size = pop(evm.stack)

    # GAS
    words = ceil32(Uint(size)) // Uint(32)
    copy_gas_cost = GAS_COPY * words
    extend_memory = calculate_gas_extend_memory(
        evm.memory, [(memory_start_index, size)]
    )

    is_cold_access = address not in evm.accessed_addresses
    access_gas_cost = (
        GAS_COLD_ACCOUNT_ACCESS if is_cold_access else GAS_WARM_ACCESS
    )
    total_gas_cost = access_gas_cost + copy_gas_cost + extend_memory.cost

    if is_cold_access:
        evm.accessed_addresses.add(address)

    charge_gas(evm, total_gas_cost)

    # OPERATION
    evm.memory += b"\x00" * extend_memory.expand_by
    state = evm.message.block_env.state
    code = get_account(state, address).code
    track_address(evm.state_changes, address)

    value = buffer_read(code, code_start_index, size)
    memory_write(evm.memory, memory_start_index, value)

    # PROGRAM COUNTER
    evm.pc += Uint(1)


def returndatasize(evm: Evm) -> None:
    """
    Pushes the size of the return data buffer onto the stack.

    Parameters
    ----------
    evm :
        The current EVM frame.

    """
    # STACK
    pass

    # GAS
    charge_gas(evm, GAS_BASE)

    # OPERATION
    push(evm.stack, U256(len(evm.return_data)))

    # PROGRAM COUNTER
    evm.pc += Uint(1)


def returndatacopy(evm: Evm) -> None:
    """
    Copies data from the return data buffer to memory.

    Parameters
    ----------
    evm :
        The current EVM frame.

    """
    # STACK
    memory_start_index = pop(evm.stack)
    return_data_start_position = pop(evm.stack)
    size = pop(evm.stack)

    # GAS
    words = ceil32(Uint(size)) // Uint(32)
    copy_gas_cost = GAS_RETURN_DATA_COPY * words
    extend_memory = calculate_gas_extend_memory(
        evm.memory, [(memory_start_index, size)]
    )
    charge_gas(evm, GAS_VERY_LOW + copy_gas_cost + extend_memory.cost)
    if Uint(return_data_start_position) + Uint(size) > ulen(evm.return_data):
        raise OutOfBoundsRead

    evm.memory += b"\x00" * extend_memory.expand_by
    value = evm.return_data[
        return_data_start_position : return_data_start_position + size
    ]
    memory_write(evm.memory, memory_start_index, value)

    # PROGRAM COUNTER
    evm.pc += Uint(1)


def extcodehash(evm: Evm) -> None:
    """
    Returns the keccak256 hash of a contract’s bytecode.

    Parameters
    ----------
    evm :
        The current EVM frame.

    """
    # STACK
    address = to_address_masked(pop(evm.stack))

    # GAS
    is_cold_access = address not in evm.accessed_addresses
    access_gas_cost = (
        GAS_COLD_ACCOUNT_ACCESS if is_cold_access else GAS_WARM_ACCESS
    )
    if is_cold_access:
        evm.accessed_addresses.add(address)

    charge_gas(evm, access_gas_cost)

    # OPERATION
    state = evm.message.block_env.state
    account = get_account(state, address)
    track_address(evm.state_changes, address)

    if account == EMPTY_ACCOUNT:
        codehash = U256(0)
    else:
        code = account.code
        codehash = U256.from_be_bytes(keccak256(code))

    push(evm.stack, codehash)

    # PROGRAM COUNTER
    evm.pc += Uint(1)


def self_balance(evm: Evm) -> None:
    """
    Pushes the balance of the current address to the stack.

    Parameters
    ----------
    evm :
        The current EVM frame.

    """
    # STACK
    pass

    # GAS
    charge_gas(evm, GAS_FAST_STEP)

    # OPERATION
    # Non-existent accounts default to EMPTY_ACCOUNT, which has balance 0.
    balance = get_account(
        evm.message.block_env.state, evm.message.current_target
    ).balance

    push(evm.stack, balance)

    # PROGRAM COUNTER
    evm.pc += Uint(1)


def base_fee(evm: Evm) -> None:
    """
    Pushes the base fee of the current block on to the stack.

    Parameters
    ----------
    evm :
        The current EVM frame.

    """
    # STACK
    pass

    # GAS
    charge_gas(evm, GAS_BASE)

    # OPERATION
    push(evm.stack, U256(evm.message.block_env.base_fee_per_gas))

    # PROGRAM COUNTER
    evm.pc += Uint(1)


def blob_hash(evm: Evm) -> None:
    """
    Pushes the versioned hash at a particular index on to the stack.

    Parameters
    ----------
    evm :
        The current EVM frame.

    """
    # STACK
    index = pop(evm.stack)

    # GAS
    charge_gas(evm, GAS_BLOBHASH_OPCODE)

    # OPERATION
    if int(index) < len(evm.message.tx_env.blob_versioned_hashes):
        blob_hash = evm.message.tx_env.blob_versioned_hashes[index]
    else:
        blob_hash = Bytes32(b"\x00" * 32)
    push(evm.stack, U256.from_be_bytes(blob_hash))

    # PROGRAM COUNTER
    evm.pc += Uint(1)


def blob_base_fee(evm: Evm) -> None:
    """
    Pushes the blob base fee on to the stack.

    Parameters
    ----------
    evm :
        The current EVM frame.

    """
    # STACK
    pass

    # GAS
    charge_gas(evm, GAS_BASE)

    # OPERATION
    blob_base_fee = calculate_blob_gas_price(
        evm.message.block_env.excess_blob_gas
    )
    push(evm.stack, U256(blob_base_fee))

    # PROGRAM COUNTER
    evm.pc += Uint(1)


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
    value_bytes, _ = _get_txparam_value(evm, int(selector), int(index))

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
