"""Configuration for the CEQ causal LM.

Self-contained on purpose. The Hub copies this file and the modeling file into a
flat repository, so it may not import anything from the `ceq` package.

THE DEFAULTS ARE THE MEASURED OPERATING POINT, NOT A PREFERENCE.
`operator="sgate", rho=1.5, lam=0.10, hops=2` is the point that reached val-loss
**1.0334x** softmax (median over 5 seeds, 1.0199-1.0413) at **3,319,296
parameters exactly on both arms**, byte-level TinyStories, 600 steps, lr 1e-3 for
both arms. Any other setting of these four is untested. The previous default --
`operator="signed", rho=0.9, hops=3` -- measured **1.337x** at 600 steps with the
gap WIDENING as the budget grew, and it is kept selectable only as the negative
control.

WHAT THAT NUMBER IS NOT. Parity at 3.3M and seq 128. It is not a scaling claim,
it is not measured past 600 steps on this operator, and the property that
distinguishes the operator from softmax decays as `s^-1.389` in context length --
see `COSTS` in `modeling_ceq.py`, which ships in the same repository.

`operator`, `rho`, `lam` and `hops` all default to `None` rather than to a value,
and that is load-bearing rather than stylistic: it is the only way this file can
tell a fresh construction from a `config.json` written before the `operator` key
existed. Setting any of the three knobs without naming the operator raises.
"""
from __future__ import annotations

from transformers import PretrainedConfig

#: (rho, lam, hops) -- the iteration-16 parity point. Named so a reader can grep
#: for it and so a drifted default fails
#: `test_the_shipped_default_config_is_the_parity_point`.
PARITY_POINT = (1.5, 0.10, 2)

#: operator -> its own (rho, lam, hops) defaults. SEPARATE ON PURPOSE. rho = 1.5
#: belongs to `sgate`; every number on record for the L1 `signed` operator was
#: measured at rho = 0.9 hops = 3, so a shared default would silently move the
#: control arm the moment the default arm moved.
OPERATOR_DEFAULTS = {"sgate": PARITY_POINT, "signed": (0.9, 0.10, 3)}
OPERATORS = tuple(OPERATOR_DEFAULTS)


class CEQConfig(PretrainedConfig):
    """`model_type` must not collide with any name already in transformers'
    CONFIG_MAPPING. A collision routes the Auto classes to the LIBRARY
    implementation and the custom code is never executed -- no error raised,
    wrong model returned. `test_the_model_type_does_not_collide_with_a_builtin`
    checks the live registry rather than a list written from memory.
    """

    model_type = "ceq"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 256,
        hidden_size: int = 1280,
        num_hidden_layers: int = 24,
        num_attention_heads: int = 20,
        intermediate_size: int | None = None,
        max_position_embeddings: int = 2048,
        operator: str | None = None,
        rho: float | None = None,
        lam: float | None = None,
        hops: int | None = None,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-5,
        tie_word_embeddings: bool = False,
        use_cache: bool = False,
        **kwargs,
    ):
        #: A KNOB WITHOUT AN OPERATOR IS REFUSED, and this guard exists because
        #: changing the default created the hazard it catches. Every `config.json`
        #: written before 2026-08-25 carries `"rho": 0.9, "hops": 3` and NO
        #: `"operator"` key, because there was only one operator. `from_dict`
        #: calls `cls(**config_dict)`, so such a checkpoint would silently load as
        #: `sgate` at rho=0.9 hops=3 -- an operating point nobody has measured --
        #: and run it on weights trained with `signed`. Every shape matches and
        #: nothing complains. Bound by
        #: `test_a_config_written_before_the_operator_key_existed_is_refused`.
        if operator is None:
            if any(v is not None for v in (rho, lam, hops)):
                raise ValueError(
                    "operator knobs were set without naming an operator "
                    "(rho={!r}, lam={!r}, hops={!r}). A config.json written "
                    "before the `operator` key existed looks exactly like this, "
                    "and loading it now would run {!r} on weights trained with "
                    "'signed'. Set `operator` explicitly: 'signed' for a "
                    "checkpoint from before 2026-08-25, 'sgate' for the "
                    "operator that reached parity."
                    .format(rho, lam, hops, OPERATORS[0]))
            operator = OPERATORS[0]
        #: VALIDATED, not defaulted. `PretrainedConfig.__init__` absorbs any
        #: unknown keyword into an attribute without complaint, so a typo in a
        #: config.json -- `"operator": "sgaet"` -- would otherwise be stored,
        #: ignored by the block, and train the other operator for a week with
        #: nothing raised anywhere.
        if operator not in OPERATOR_DEFAULTS:
            raise ValueError(
                "unknown operator {!r}; expected one of {}. The shipped default "
                "is 'sgate', the operator the 1.0334 parity median was measured "
                "on. 'signed' is the L1-normalized negative control and measured "
                "1.337x.".format(operator, OPERATORS))
        default_rho, default_lam, default_hops = OPERATOR_DEFAULTS[operator]
        rho = default_rho if rho is None else rho
        lam = default_lam if lam is None else lam
        hops = default_hops if hops is None else hops
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size or 4 * hidden_size
        self.max_position_embeddings = max_position_embeddings
        #: "sgate" -- `rho * (softmax(w) - lam * softmax(-w)) / (1 + lam)`. Both
        #: halves keep `exp` sharpening. THIS IS THE OPERATOR THAT REACHED
        #: PARITY; bitwise-checked against `ceq/lm.py`, where the run happened,
        #: by `test_the_package_implements_the_operator_that_reached_parity`.
        #:
        #: "signed" -- `rho * w / sum|w|`. Signed but LINEAR in the logits and
        #: exactly homogeneous of degree zero, so it has no temperature channel
        #: at all: peak row weight was 0.100336 at logit scales 0.25, 1.0, 4.0
        #: and 16.0 alike, spread exactly 0.000e+00 over a 64x sweep. That is why
        #: it lost, and it stays selectable as the control.
        self.operator = operator
        #: Row L1 budget of the operator, `||A||_inf <= rho` for both forms.
        #:
        #: rho = 1.5 IS GREATER THAN 1 AND THAT COSTS A GUARANTEE. The tail of a
        #: truncated path sum is bounded by `rho^(K+1)/(1-rho)` only for rho < 1;
        #: at rho = 1.5 that expression is NEGATIVE (-6.75 at hops=2) and
        #: `ceq.attention.truncation_bound` now raises rather than returning it.
        #: Exactness still exists but only through nilpotency, which needs
        #: hops >= S-1, not 2. So the shipped point is a genuine truncation with
        #: no geometric bound, and the contraction results in `CEQ.Contraction`
        #: do not apply to it. `CEQ.Nilpotent.pow_card_eq_zero` still does: it is
        #: stated over `[CommRing R]` with the single hypothesis
        #: `forall i j, i <= j -> A i j = 0`, so it is blind to both sign and
        #: magnitude.
        self.rho = rho
        #: Weight on the negative softmax half. lam = 1 makes every row sum
        #: EXACTLY zero, which annihilates the constant vector (measured row sum
        #: 0.000000e+00, |A @ 1| = 1.192e-07) and lets the path sum add only
        #: deviations, never signal level. lam = 0 deletes the negative half and
        #: leaves `rho * softmax(w)`, which is non-negative and therefore back in
        #: tier 2. 0.10 is the swept optimum between the two.
        self.lam = lam
        #: Number of hops in the path sum. Each hop is one extra [S,S] @ [S,D]
        #: matmul per layer per forward. hops=2 is the parity point and is also
        #: where the content-conditional sign rate peaks (0.1641 at hops=2
        #: against 0.0234 at hops=1 and 0.1484 at hops=3).
        self.hops = hops
        #: UNTIED BY DEFAULT, and this is a rollback rather than a preference.
        #: `PreTrainedModel.is_remote_code()` is `cls._auto_class is not None`,
        #: so `register_for_auto_class()` -- the only way save_pretrained copies
        #: these .py files into the repo -- puts the model on a load path that
        #: refuses to tie and leaves `lm_head.weight` on the META device, with
        #: no exception. The forward then RUNS and returns uninitialized memory:
        #: measured 1.0053620544046395e+30. Bound by
        #: `test_registering_for_auto_class_does_not_break_weight_tying` and, for
        #: the caller who overrides this anyway, by
        #: `test_an_unmaterialized_lm_head_raises_instead_of_returning_garbage`.
        #: Cost of untying: one extra vocab x d matrix, 40.96M parameters on the
        #: 515.57M 0.5B configuration, +7.9%.
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        #: FORCED OFF, and not a default the caller may flip. `(A^h v)_i` is a
        #: contraction over j < i, so during decode the operator row is [1, N] and
        #: `A^2` is undefined -- the multi-hop path sum cannot be computed from a
        #: standard KV cache. `ceq/hopcache.py` is the structure that fixes this,
        #: at K cached vectors per position, and it is not wired in here.
        self.use_cache = False
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads


CEQConfig.register_for_auto_class()
