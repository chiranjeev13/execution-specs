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
    APPROVE_SCOPE_MASK,
    FRAME_MODE_VERIFY,
    FrameTransaction,
    calculate_intrinsic_cost,
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
    """Raised when a frame index argument is out of bounds."""

class InvalidFrameParam(ExceptionalHalt):
    """Raised when ``FRAMEPARAM`` is called with an invalid parameter id."""


def _get_frame_tx(evm: Evm) -> FrameTransaction:
    """Get the active frame transaction or raise."""
    tx = evm.message.tx_env.frame_tx
    if tx is None or not isinstance(tx, FrameTransaction):
        raise FrameTxNotActiveError
    return tx


def _txparam_word(evm: Evm, param: int) -> bytes:
    """
    Return the 32-byte word for ``TXPARAM`` as defined in [EIP-8141].

    Only selectors ``0x00`` … ``0x0A`` are valid.

    [EIP-8141]: https://eips.ethereum.org/EIPS/eip-8141
    """
    tx = _get_frame_tx(evm)
    tx_env = evm.message.tx_env
    block_env = evm.message.block_env

    if param == 0x00:
        return U256(0x06).to_be_bytes32()
    elif param == 0x01:
        return U256(tx.nonce).to_be_bytes32()
    elif param == 0x02:
        return b"\x00" * 12 + bytes(tx.sender)
    elif param == 0x03:
        return U256(tx.max_priority_fee_per_gas).to_be_bytes32()
    elif param == 0x04:
        return U256(tx.max_fee_per_gas).to_be_bytes32()
    elif param == 0x05:
        return U256(tx.max_fee_per_blob_gas).to_be_bytes32()
    elif param == 0x06:
        tx_gas_limit, _ = calculate_intrinsic_cost(tx)
        effective_gas_price = tx_env.gas_price
        blob_count = len(tx.blob_versioned_hashes)
        blob_gas_price = calculate_blob_gas_price(block_env.excess_blob_gas)
        blob_fees = Uint(blob_count) * Uint(GAS_PER_BLOB) * Uint(blob_gas_price)
        max_cost_val = Uint(tx_gas_limit) * Uint(effective_gas_price) + blob_fees
        return U256(max_cost_val).to_be_bytes32()
    elif param == 0x07:
        return U256(len(tx.blob_versioned_hashes)).to_be_bytes32()
    elif param == 0x08:
        return bytes(signing_hash_8141(tx))
    elif param == 0x09:
        return U256(len(tx.frames)).to_be_bytes32()
    elif param == 0x0A:
        frame_idx = tx_env.current_frame_index
        if frame_idx is None:
            raise FrameTxNotActiveError
        return U256(frame_idx).to_be_bytes32()
    else:
        raise InvalidTxParamSelector


def _frame_bytes_at_index(tx: FrameTransaction, index: int) -> bytes:
    """Return ``frame.data`` bytes (empty when the frame is ``VERIFY``)."""
    if index >= len(tx.frames):
        raise TxParamOutOfBounds
    frame = tx.frames[index]
    if frame.mode == FRAME_MODE_VERIFY:
        return b""
    return bytes(frame.data)


def _frameparam_word(evm: Evm, param: int, frame_index: int) -> bytes:
    """
    Return the 32-byte word for ``FRAMEPARAM`` as defined in [EIP-8141].

    [EIP-8141]: https://eips.ethereum.org/EIPS/eip-8141
    """
    tx = _get_frame_tx(evm)
    tx_env = evm.message.tx_env

    if frame_index >= len(tx.frames):
        raise TxParamOutOfBounds

    frame = tx.frames[frame_index]

    if param == 0x00:
        if isinstance(frame.target, Bytes0) or frame.target == Bytes0(b""):
            return b"\x00" * 32
        return b"\x00" * 12 + bytes(frame.target)
    elif param == 0x01:
        return U256(frame.gas_limit).to_be_bytes32()
    elif param == 0x02:
        return U256(frame.mode).to_be_bytes32()
    elif param == 0x03:
        return U256(frame.flags).to_be_bytes32()
    elif param == 0x04:
        data_b = _frame_bytes_at_index(tx, frame_index)
        return U256(len(data_b)).to_be_bytes32()
    elif param == 0x05:
        cur = tx_env.current_frame_index
        if cur is None:
            raise FrameTxNotActiveError
        if frame_index >= cur:
            raise TxParamOutOfBounds
        statuses = tx_env.frame_statuses
        if statuses is None or frame_index >= len(statuses):
            raise TxParamOutOfBounds
        return U256(statuses[frame_index]).to_be_bytes32()
    elif param == 0x06:
        allowed = Uint(frame.flags) & APPROVE_SCOPE_MASK
        return U256(allowed).to_be_bytes32()
    elif param == 0x07:
        batch = (Uint(frame.flags) >> Uint(2)) & Uint(1)
        return U256(batch).to_be_bytes32()
    elif param == 0x08:
        return frame.value.to_be_bytes32()
    else:
        raise InvalidFrameParam


def txparam(evm: Evm) -> None:
    """
    ``TXPARAM`` opcode (``0xB0``).

    Transaction-scoped parameters per [EIP-8141].

    Stack: ``param`` → ``word``

    [EIP-8141]: https://eips.ethereum.org/EIPS/eip-8141
    """
    param = pop(evm.stack)

    charge_gas(evm, GAS_BASE)

    word = _txparam_word(evm, int(param))
    push(evm.stack, U256.from_be_bytes(word))

    evm.pc += Uint(1)


def frameparam(evm: Evm) -> None:
    """
    ``FRAMEPARAM`` opcode (``0xB3``).

    Per-frame parameters per [EIP-8141]. Stack top first: ``frameIndex``,
    then ``param``.

    [EIP-8141]: https://eips.ethereum.org/EIPS/eip-8141
    """
    fparam = pop(evm.stack)
    frame_index = pop(evm.stack)

    charge_gas(evm, GAS_BASE)

    word = _frameparam_word(evm, int(fparam), int(frame_index))
    push(evm.stack, U256.from_be_bytes(word))

    evm.pc += Uint(1)


def framedataload(evm: Evm) -> None:
    """
    ``FRAMEDATALOAD`` opcode (``0xB1``).

    Stack top first: ``offset``, ``frameIndex``.

    [EIP-8141]: https://eips.ethereum.org/EIPS/eip-8141
    """
    offset = pop(evm.stack)
    frame_index = pop(evm.stack)

    charge_gas(evm, GAS_VERY_LOW)

    tx = _get_frame_tx(evm)
    data_b = _frame_bytes_at_index(tx, int(frame_index))
    off = int(offset)
    result = bytearray(32)
    for i in range(32):
        idx = off + i
        if idx < len(data_b):
            result[i] = data_b[idx]

    push(evm.stack, U256.from_be_bytes(bytes(result)))

    evm.pc += Uint(1)


def framedatacopy(evm: Evm) -> None:
    """
    ``FRAMEDATACOPY`` opcode (``0xB2``).

    Stack top first: ``frameIndex``, ``length``, ``dataOffset``, ``memOffset``.

    [EIP-8141]: https://eips.ethereum.org/EIPS/eip-8141
    """
    mem_offset = pop(evm.stack)     # top - 0 = memOffset
    data_offset = pop(evm.stack)    # top - 1 = dataOffset
    length = pop(evm.stack)         # top - 2 = length
    frame_index = pop(evm.stack)        # top - 3 = frameIndex

    tx = _get_frame_tx(evm)
    data_b = _frame_bytes_at_index(tx, int(frame_index))

    words = ceil32(Uint(length)) // Uint(32)
    extend_memory = calculate_gas_extend_memory(
        evm.memory, [(mem_offset, length)]
    )
    copy_gas_cost = GAS_COPY * words
    charge_gas(evm, GAS_VERY_LOW + copy_gas_cost + extend_memory.cost)

    evm.memory += b"\x00" * extend_memory.expand_by

    src_off = int(data_offset)
    copy_len = int(length)
    result = bytearray(copy_len)
    for i in range(copy_len):
        idx = src_off + i
        if idx < len(data_b):
            result[i] = data_b[idx]

    memory_write(evm.memory, mem_offset, Bytes(bytes(result)))

    evm.pc += Uint(1)


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
