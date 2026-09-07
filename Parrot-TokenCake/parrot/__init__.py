# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


"""Parrot: Efficient Serving LLM-based Applications with Dependent Semantic Variables."""

__version__ = "0.01"

import torch.utils._pytree as _torch_pytree

if not hasattr(_torch_pytree, "register_pytree_node") and hasattr(
    _torch_pytree, "_register_pytree_node"
):
    def _register_pytree_node_compat(
        typ,
        flatten_fn,
        unflatten_fn,
        *,
        serialized_type_name=None,
        to_dumpable_context=None,
        from_dumpable_context=None,
        flatten_with_keys_fn=None,
    ):
        return _torch_pytree._register_pytree_node(
            typ,
            flatten_fn,
            unflatten_fn,
            to_dumpable_context=to_dumpable_context,
            from_dumpable_context=from_dumpable_context,
        )

    _torch_pytree.register_pytree_node = _register_pytree_node_compat

# Import PFunc frontend
import parrot.frontend.pfunc as P
