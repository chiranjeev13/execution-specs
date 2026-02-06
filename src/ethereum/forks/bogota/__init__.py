"""
The Bogota fork includes block-level access lists and frame transactions.

### Changes

- [EIP-7928: Block-Level Access Lists][EIP-7928]
- [EIP-8141: Frame Transaction][EIP-8141]

### Releases

[EIP-7928]: https://eips.ethereum.org/EIPS/eip-7928
[EIP-8141]: https://eips.ethereum.org/EIPS/eip-8141
"""

from ethereum.fork_criteria import ForkCriteria, Unscheduled

FORK_CRITERIA: ForkCriteria = Unscheduled(order_index=4)
