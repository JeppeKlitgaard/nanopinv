from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import jaxtyping
from jax.tree_util import DictKey, GetAttrKey, SequenceKey, tree_map_with_path

__NEW_KEY_DTYPE = jax.random.key(0).dtype


@dataclass(slots=True)
class StatefulRNGKey:
    key: jaxtyping.Key

    def __init__(self, key: jaxtyping.Key | int):
        """
        Note: Key should be new style (made with `jax.random.key`, not `jax.random.PRNGKey`)
        """
        if isinstance(key, int):
            key = jax.random.key(key)

        elif key.dtype != __NEW_KEY_DTYPE:
            key = jax.random.wrap_key_data(key)

        self.key = key

    def shaped(self, shape: tuple[int, ...]):
        num_new_keys = jnp.prod(jnp.array(shape)) + 1
        new_keys = jax.random.split(self.key, num=num_new_keys)

        self.key = new_keys[0]
        return new_keys[1:].reshape((*shape, *new_keys.shape[1:]))

    def __call__(self, n: int | None = None):
        if n is None:
            self.key, subkey = jax.random.split(self.key)
            return subkey

        new_keys = jax.random.split(self.key, num=n + 1)
        self.key = new_keys[0]
        return new_keys[1:]


def make_pytree_spec(pytree, spec: dict):
    """
    Creates a vmap axis spec for a PyTree using a dictionary of string path rules.

    Supports Pythonic syntax for sequences and dicts (e.g., "layers[0].weight").

    Rules are evaluated from most specific to least specific:
    1. Exact match (e.g., "temperature" or "layers[0].weight")
    2. Sub-spec structural match (maps a nested PyTree/Dict spec to a subtree)
    3. Explicit wildcard match (e.g., "proposal_dist.*" or "layers.*")
    4. Global wildcard (e.g., "*")
    """

    def path_to_string(path):
        res = ""
        for p in path:
            if isinstance(p, GetAttrKey):
                res += f".{p.name}" if res else p.name
            elif isinstance(p, SequenceKey):
                res += f"[{p.idx}]"
            elif isinstance(p, DictKey):
                res += f"[{repr(p.key)}]"
            else:
                res += f".{p}" if res else str(p)
        return res

    def map_fn(path, leaf):
        path_str = path_to_string(path)

        # 1. Check for an Exact Match
        if path_str in spec:
            return spec[path_str]

        # 2. Check for Sub-spec matches (composing nested PyTrees/Dicts)
        # We check valid structural path prefixes from longest to shortest
        for i in range(len(path) - 1, 0, -1):
            prefix_str = path_to_string(path[:i])
            if prefix_str in spec:
                val = spec[prefix_str]
                remainder = path[i:]

                res = val
                for p in remainder:
                    # Prevent accidental iteration over primitives or strings
                    if isinstance(res, str):
                        raise ValueError(
                            f"Rule '{prefix_str}' matched as a prefix for '{path_str}', but its value is a string. "
                            f"If you intended to broadcast a wildcard, use '{prefix_str}.*'."
                        )

                    if isinstance(p, GetAttrKey):
                        if isinstance(res, dict) and p.name in res:
                            res = res[p.name]
                        elif hasattr(res, p.name):
                            res = getattr(res, p.name)
                        else:
                            raise ValueError(
                                f"Rule '{prefix_str}' matched as a prefix for '{path_str}', "
                                f"but the provided sub-spec is missing attribute '{p.name}'. "
                                f"If you intended to broadcast a wildcard, use '{prefix_str}.*'."
                            )
                    elif isinstance(p, SequenceKey):
                        try:
                            res = res[p.idx]
                        except TypeError, IndexError, KeyError:
                            raise ValueError(
                                f"Rule '{prefix_str}' matched as a prefix for '{path_str}', "
                                f"but the provided sub-spec cannot be indexed by [{p.idx}]. "
                                f"If you intended to broadcast a wildcard, use '{prefix_str}.*'."
                            )
                    elif isinstance(p, DictKey):
                        try:
                            res = res[p.key]
                        except TypeError, KeyError:
                            raise ValueError(
                                f"Rule '{prefix_str}' matched as a prefix for '{path_str}', "
                                f"but the provided sub-spec does not contain key [{repr(p.key)}]. "
                                f"If you intended to broadcast a wildcard, use '{prefix_str}.*'."
                            )
                    else:
                        if hasattr(p, "name") and hasattr(res, p.name):
                            res = getattr(res, p.name)
                        else:
                            raise ValueError(
                                f"Cannot traverse path element {p} on value provided for '{prefix_str}'."
                            )

                return res

        # 3. Check for Explicit Prefix Wildcards
        best_match_len = -1
        best_val = None

        for key, val in spec.items():
            if key.endswith(".*"):
                base_key = key[:-2]

                # A path is a child if it's followed by an attribute (.) or index ([)
                is_match = path_str.startswith(base_key + ".") or path_str.startswith(
                    base_key + "["
                )

                if is_match:
                    if len(base_key) > best_match_len:
                        best_match_len = len(base_key)
                        best_val = val

        if best_match_len != -1:
            return best_val

        # 4. Check for Global Wildcard
        if "*" in spec:
            return spec["*"]

        raise ValueError(
            f"No rule found for leaf path: '{path_str}', and no '*' fallback provided."
        )

    return tree_map_with_path(map_fn, pytree)
