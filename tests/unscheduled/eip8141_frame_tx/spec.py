"""Defines EIP-8141 specification constants and helpers."""

from dataclasses import dataclass

from execution_testing import Address, Bytes


@dataclass(frozen=True)
class ReferenceSpec:
    """Defines the reference spec version and git path."""

    git_path: str
    version: str


ref_spec_8141 = ReferenceSpec(
    "EIPS/eip-8141.md", "0626637feb8ff09789c02682588d80cfd22caa4c"
)


@dataclass(frozen=True)
class Spec:
    """Parameters from the EIP-8141 specifications."""

    FRAME_TX_TYPE = 0x06
    FRAME_TX_INTRINSIC_COST = 15_000
    ENTRY_POINT = Address(0xAA)
    MAX_FRAMES = 10**3

    MODE_DEFAULT = 0
    MODE_VERIFY = 1
    MODE_SENDER = 2

    APPROVE_OPCODE = 0xAA
    TXPARAMLOAD_OPCODE = 0xB0
    TXPARAMSIZE_OPCODE = 0xB1
    TXPARAMCOPY_OPCODE = 0xB2

    MAX_CHAIN_ID = 2**256 - 1
    MAX_NONCE = 2**64 - 1

    APPROVE_EXECUTION = 0x0
    APPROVE_PAYMENT = 0x1
    APPROVE_BOTH = 0x2

    STATUS_FAILURE = 0
    STATUS_SUCCESS = 1

    EMPTY_BYTES = Bytes(b"")
