# Prior art for the mass channel: output gates on softmax attention

Written before arm (a2) of design4x5 runs, not after it reports.

## The claim this section forecloses

Arm (a2) is an SDPA twin with a per-head sigmoid gate on the attention output,
applied before `o_proj`, attention pattern unchanged. Its contrast `C_mass`
prices what an output mass-gate buys. **If `C_mass` carries the result, the
mechanism is published work and this project is replicating it, not finding
it.** That sentence is on the page before the arm runs.

## The three sources, fetched

**Qwen Team (Alibaba) et al., 2025 — the direct overlap.**
Zihan Qiu, Zekun Wang, Bo Zheng, Zeyu Huang, Kaiyue Wen, Songlin Yang, Rui Men,
Le Yu, Fei Huang, Suozhi Huang, Dayiheng Liu, Jingren Zhou, Junyang Lin,
"Gated Attention for Large Language Models: Non-linearity, Sparsity, and
Attention-Sink-Free," arXiv:2505.06708v1 [cs.CL], 10 May 2025; NeurIPS 2025
Oral. Abstract, verbatim: *"Our central finding is that a simple
modification—applying a head-specific sigmoid gate after the Scaled Dot-Product
Attention (SDPA)—consistently improves performance."* The gate is
`Y' = Y ⊙ σ(X W_θ)` with `X` the pre-normalised hidden state; position `G1`
(after SDPA, before `W_o`) beat the four alternatives, including `G5` (after
`W_o`). Granularity, verbatim: *"Headwise: A single scalar gating score
modulates the entire output of an attention head"* — and *"Applying headwise
gating at G₁ and G₂ introduces very few additional parameters (less than 2M)
but still delivers substantial improvements."*

**Miller, 2023 — the denominator route to the same freedom.**
Evan Miller, "Attention Is Off By One," evanmiller.org, 24 July 2023, section
"Softmax One and Quiet Attention"
(<https://www.evanmiller.org/attention-is-off-by-one.html>). An essay, not a
refereed paper. Verbatim: *"All I did was added one to the denominator. This
lets the vector as a whole tend to zero if it wants."* Formula as printed:
`(softmax₁(x))ᵢ = exp(xᵢ) / (1 + Σⱼ exp(xⱼ))`.

**Xiao, Tian, Chen, Han, Lewis, 2023 — why the mass has to go somewhere.**
"Efficient Streaming Language Models with Attention Sinks," arXiv:2309.17453v4
(29 Sep 2023, rev. 7 Apr 2024), ICLR 2024 (`xiao-2023-attentionsinks`).
Verbatim: *"The nature of the SoftMax function prevents all attended tokens
from having zero values"*; *"the model tends to dump unnecessary attention
values to specific tokens."* Their §3.3 tries Miller's denominator as the
"Zero Sink," *"which does not require the attention scores on all contextual
tokens to sum up to one,"* and reports it insufficient on its own; a learnable
sink token trained from scratch works better.

## Does an output mass-gate on softmax attention already exist?

**Yes, in exactly arm (a2)'s form.** Qwen's `G1` headwise variant is a
per-head scalar sigmoid computed from the layer input, multiplied into the SDPA
output before the output projection. That is (a2) as specified, down to the
gate's input, its position relative to `o_proj`, and the leaving of the
attention pattern untouched. Qwen also report the mechanism's effect on the
normalisation artefact Xiao named: *"Input-dependent, head-specific gating of
the SDPA output introduces significant sparsity, thereby mitigating the
attention sink."* Arm (a2) is therefore a **replication at toy scale on a
different bed**, and `C_mass` is a measurement of a known effect, not a
discovery of one.

Three differences, stated so the overlap is not inflated:

1. **Scale and corpus.** Qwen: 1.7B dense and 15B-MoE (2.54B active), up to
   3.5T tokens, 4096 context. Arm (a2): 727,704 parameters, hidden 128, 3
   layers, 8 heads, vocab 256, seq 512, 3,538 steps. Nothing here replicates
   their result; it tests whether the same mechanism is worth anything at this
   size.
2. **Granularity.** Qwen's headline is head-specific gating with elementwise
   granularity available and finer; (a2) is the headwise scalar, which their
   own ablation reports as the cheaper, still-substantial variant, not the
   best one.
3. **What is measured.** Qwen report perplexity, MMLU and sink rate. `C_mass`
   is a CRN-paired nat-level contrast against a matched SDPA twin. Note the
   arms are *not* parameter-matched: (a) has 724,608 parameters and (a2) has
   727,704, so `C_mass` prices the gate plus 3,096 parameters (+0.43%). The
   `(f)/(f0)` pair is matched at 725,391 either side; `(a)/(a2)` is not.

Miller's `softmax₁` is **not** the same mechanism and should not be folded in:
it removes the sum-to-one constraint inside the attention row, changing the
pattern. (a2) leaves the row stochastic and scales the result afterwards. Both
free a head to output near-zero; only one of them touches the distribution.

## One line kept out of the cited claims

The log-partition identity — `log Σᵢ exp fᵢ = sup_Q [ E_Q f − KL(Q‖U) ] + log n`,
maximiser `softmax(f)`, which would make softmax attention the Donsker–Varadhan
optimiser and the log-partition a free energy — is stated here as an identity
that can be derived in a line, and is **deliberately carried without a source
locator**: no primary text was fetched for it in this pass, and this project's
recent failure mode is a standard result attached to the wrong paper. It is a
Lean target, not yet a citation.
