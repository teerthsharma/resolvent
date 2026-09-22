"""RED stub: every mechanism unimplemented. test_rivals.py run against this
module must fail every row before rivals_impl.py exists."""


def _no(*a, **k):
    raise NotImplementedError


a2_gate = qwen_g1_headwise = qwen_g1_elementwise = _no
fox_logf = gdn_logdecay = decay_mask_from_logf = _no
gdn_recurrent = linear_attn_masked = softmax_attn_masked = _no
kda_transition = gdn_transition = householder_general = su2_from_quat = _no
lightning_mask = alibi_mask = nsa_combine = _no
