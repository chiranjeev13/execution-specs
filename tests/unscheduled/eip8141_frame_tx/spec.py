"""Defines EIP-8141 specification constants and helpers."""

from dataclasses import dataclass

from execution_testing import Address, Bytes


@dataclass(frozen=True)
class ReferenceSpec:
    """Defines the reference spec version and git path."""

    git_path: str
    version: str


ref_spec_8141 = ReferenceSpec(
    "EIPS/eip-8141.md", "d56e73ada787162d8a366267e549bd27484699e3"
)


@dataclass(frozen=True)
class Spec:
    """Parameters from the EIP-8141 specifications."""

    FRAME_TX_TYPE = 0x06
    FRAME_TX_INTRINSIC_COST = 15_000
    ENTRY_POINT = Address(0xAA)
    MAX_FRAMES = 64
    FRAME_TX_PER_FRAME_COST = 475

    MODE_DEFAULT = 0
    MODE_VERIFY = 1
    MODE_SENDER = 2

    APPROVE_OPCODE = 0xAA
    TXPARAM_OPCODE = 0xB0
    FRAMEDATALOAD_OPCODE = 0xB1
    FRAMEDATACOPY_OPCODE = 0xB2
    FRAMEPARAM_OPCODE = 0xB3

    MAX_CHAIN_ID = 2**256 - 1
    MAX_NONCE = 2**64 - 1

    APPROVE_SCOPE_NONE = 0x0
    APPROVE_PAYMENT = 0x1
    APPROVE_EXECUTION = 0x2
    APPROVE_PAYMENT_AND_EXECUTION = 0x3
    APPROVE_BOTH = APPROVE_PAYMENT_AND_EXECUTION

    STATUS_FAILURE = 0
    STATUS_SUCCESS = 1

    EMPTY_BYTES = Bytes(b"")
