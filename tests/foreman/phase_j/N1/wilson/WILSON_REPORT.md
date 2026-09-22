# WILSON: facts behind Addendum N (v1 and v2), round N1

- **Scope files read whole:** `tests/foreman/phase_j/ADDENDUM_N_CONTRACT.md` (v1, 146 lines) and `ADDENDUM_N_CONTRACT_v2.md` (v2, 226 lines).
- **Nurses spawned: 0.** Wilson fetched every source himself, by curl of the page text or the arXiv API. Nothing was relayed.
- **Quotes are limited to one sentence per source.** Titles are bibliographic, and everything else is paraphrase.
- **Evidence location.** All evidence files named below are in `scratchpad/phase_j/N1/wilson/`.

## 1. Instance scripts: diag_leap.py, crossties.py, kr_density.py, pf_regime.py

- **None of the four files exists anywhere under C:/Users/seal.** VERIFIED.
  - Patterns searched, case-insensitive: `diag_leap*`, `crossties*`, `kr_density*` and `pf_regime*`.
  - `find4.txt`: a separate `find` over each of 233 directories returns 0 hits. That covers every top-level directory of C:/Users/seal, all dot-directories, and every subdirectory of AppData/Local, AppData/Roaming and AppData/LocalLow (138 of the 233).
  - Desktop, Documents and Downloads each return rc=0 with 0 errors.
  - `ps_search.txt`: PowerShell `Get-ChildItem -Recurse -Force` over Desktop, Documents and Downloads returns 0 hits and 0 errors.
  - `ls -a` of C:/Users/seal and the AppData roots shows no file with these names.
- **One directory could only be partly read.** VERIFIED. In `AppData/Local/npm-cache`, two entries under `_npx/` (af767e0d99514be2 and fbb43b1786f81b3f) are listed but return "No such file or directory". Their contents are UNVERIFIED.
- **A single whole-tree `find /c/Users/seal` is not a valid negative.** VERIFIED.
  - It exits rc=1 with the error "failed to read file names from file system at or below '/c/Users/seal'", after 26 errors on AppData/Local entries (`find3_err.txt`).
  - It aborts partway, so empty results from a single whole-tree run cover only the part of the tree traversed before the abort. That includes Wilson's first runs, `find_names*.txt`.
  - The per-directory run above replaces it.
- **No git history touches the four names.** VERIFIED.
  - `gitdirs.txt` lists 170 `.git` entries under Desktop and Documents (maxdepth 6), and 0 under Downloads.
  - `git log --all --name-only` with the patterns `*diag_leap*`, `*crossties*`, `*cross_ties*`, then `*kr_density*`, `*pf_regime*`, returns nothing in 164 repositories (`git_history_search.txt`, `git_history_search2.txt`).
  - Documents/GitHub/firefox and Documents/GitHub/pytorch timed out at 90 s. Rerun with all four patterns and a 1800 s limit, both return rc=0 with empty output, in 212 s and 140 s respectively (`git_history_slow.txt`).
  - The repository Desktop/New folder (32) is among the empty results.
- **Four `.git` entries are empty directories, not repositories:** Documents/Codex, Documents/Codex/2026-07-01/can, Documents/Codex/2026-07-22/thi and Documents/GitHub/Demon-of-inkosei. VERIFIED: `ls` shows them empty, and git reports "not a git repository". Their working-tree files are covered by the file search.
- **Not searched:** repositories below maxdepth 6, bare repositories, and git objects that no ref reaches. UNVERIFIED.

## 2. Citations, checked at source

**Jean Leray**
- **French, 7 Nov 1906 – 10 Nov 1998.** VERIFIED on MacTutor (https://mathshistory.st-andrews.ac.uk/Biographies/Leray/) and Wikipedia's raw text (https://en.wikipedia.org/w/index.php?title=Jean_Leray&action=raw).
- **Prisoner of war in Austria, 1940–1945.** VERIFIED on MacTutor: "He was captured in 1940 and sent to a prisoner of war camp in Austria where he remained until the end of the war in 1945."
- **The camp was at Edelbach, Austria.** VERIFIED only on Wikipedia, which is a secondary source. Its link target is "Edelbach, Allentsteig". MacTutor does not name the camp.
  - The primary memoirs are UNVERIFIED because both returned HTTP 403. They are Borel, Henkin and Lax, Notices AMS 47(3) 2000 (https://www.ams.org/notices/200003/mem-leray.pdf), and Andler, Biogr. Mem. FRS 52 (2006), doi 10.1098/rsbm.2006.0011.
- **He concealed his expertise and did topology.** VERIFIED on Wikipedia: "He concealed his expertise on differential equations, fearing that its connections with applied mathematics could lead him to be asked to aid the German war effort."
  - MacTutor names the hidden field as hydrodynamics, not differential equations. It says he presented himself as a topologist.
- **Sheaves and spectral sequences came out of the captivity work.** VERIFIED. MacTutor credits his papers of that period with introducing sheaves and the spectral sequence of a continuous map.
- **The joint paper with Schauder is from 1934.** VERIFIED. Crossref for doi 10.24033/asens.836: "Topologie et équations fonctionnelles", Leray and Schauder, Annales scientifiques de l'ENS 51 (1934) 45–78.
  - MacTutor says the paper defines the Leray–Schauder degree and uses it to prove existence of solutions. Wikipedia says it proves existence for PDEs lacking uniqueness.
  - In Wikipedia's raw text, the phrase "Leray–Schauder theorem" is a piped link to the article on the Schauder fixed-point theorem.
- **Weak solutions of Navier–Stokes, 1934.** VERIFIED. Crossref for doi 10.1007/BF02547354: "Sur le mouvement d'un liquide visqueux emplissant l'espace", Leray, Acta Mathematica 63 (1934) 193–248. Wikipedia credits this paper with founding the study of weak solutions.
- **Wolf Prize 1979.** VERIFIED at the primary source, the Wolf Foundation page https://wolffund.org.il/jean-leray/, which is headed "Wolf Prize Laureate in Mathematics 1979". MacTutor agrees.

**Qiu et al. 2025, gated attention**
- **Title, authors and identifier.** VERIFIED from the arXiv API.
  - Title: "Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free".
  - Authors: Zihan Qiu, Zekun Wang, Bo Zheng, Zeyu Huang, Kaiyue Wen, Songlin Yang, Rui Men, Le Yu, Fei Huang, Suozhi Huang, Dayiheng Liu, Jingren Zhou, Junyang Lin.
  - arXiv 2505.06708, v1, 2025-05-10. That is the only version, and it carries no comment and no journal-ref.
- **What the gate is.** The paper's gate is a head-specific sigmoid gate applied after SDPA. Abstract, VERIFIED: "Notably, we find this sparse gating mechanism mitigates 'attention sink' and enhances long-context extrapolation performance".
- **"NeurIPS 2025 Best Paper" is correct.** VERIFIED on the NeurIPS Blog post "Announcing the NeurIPS 2025 Best Paper Awards", 2025-11-26 (https://blog.neurips.cc/2025/11/26/announcing-the-neurips-2025-best-paper-awards/).
  - The post announces four best papers and three runners-up.
  - This paper, with the same 13 authors, is the second entry in the best-paper block, above the "Runners Up" heading.

**Yang et al. 2025, PaTH**
- **Title, authors and identifier.** VERIFIED from the arXiv API.
  - Title: "PaTH Attention: Position Encoding via Accumulating Householder Transformations".
  - Authors: Songlin Yang, Yikang Shen, Kaiyue Wen, Shawn Tan, Mayank Mishra, Liliang Ren, Rameswar Panda, Yoon Kim.
  - arXiv 2505.16381, v1 of 2025-05-22. The current v2's comment calls it the NeurIPS 2025 camera-ready.
- **The product sits on the q·k side.** VERIFIED from the LaTeX in the arXiv HTML of v2.
  - The attention is A_ij ∝ exp(k_jᵀ(∏_{s=j+1}^{i} H_s) q_i), so the product sits inside the logit.
  - Each factor is H_t = I − β_t w_t w_tᵀ, with β_t = 2·sigmoid(·) ∈ (0,2).
- **The factors are only "Householder-like", not Householder.** VERIFIED. A footnote contrasts H_t with a true Householder matrix, I − (2/‖u‖²)uuᵀ: "hence our matrix is only Householder-like." The abstract writes "Householder(like)".
- **The state-tracking claims are in the body, not the abstract.** VERIFIED. The abstract does not mention state tracking. The body claims that a constant-layer PaTH transformer solves an NC¹-complete problem, and runs flip-flop and word-problem experiments. The contract's [U] is therefore resolved: the claims exist in the body.

**Nakanishi 2025, SSMax**
- **Title, author and identifier.** VERIFIED: "Scalable-Softmax Is Superior for Attention", Ken M. Nakanishi, arXiv 2501.19399, v1 of 2025-01-31. The arXiv metadata gives no venue.
- **The exact equation.** VERIFIED from the LaTeX of Eq. 2 in the arXiv HTML of v1:
  - z_i ↦ n^{s z_i} / Σ_{j=1}^{n} n^{s z_j} = e^{(s log n) z_i} / Σ_j e^{(s log n) z_j}.
  - The paper adds: "where s ∈ ℝ is a learnable scaling parameter."
  - s is learned per layer and per head.
  - A bias variant is also given: e^{(s log n + b) z_i} / Σ_j e^{(s log n + b) z_j}.

**Barbero et al. 2025, "Why do LLMs attend to the first token?"**
- **Authors and identifier.** VERIFIED from the arXiv API. Federico Barbero, Álvaro Arroyo, Xiangming Gu, Christos Perivolaropoulos, Michael Bronstein, Petar Veličković, Razvan Pascanu. arXiv 2504.02732, v1 of 2025-04-03, currently v4.
- **Abstract, VERIFIED:** "Large Language Models (LLMs) tend to attend heavily to the first token in the sequence -- creating a so-called attention sink." Section 5 of the paper asks whether ⟨bos⟩ is special.
- **Venue COLM 2025: UNVERIFIED at a primary source.** OpenReview (id tu4dFUsW5z) served a browser-verification page, which was not bypassed. Only a search snippet and a third-party GitHub note name COLM'25.

**Erel et al. 2025, "Attention (as Discrete-Time Markov) Chains"**
- **Authors, identifier and venue.** VERIFIED.
  - Authors: Yotam Erel, Olaf Dünkel, Rishabh Dabral, Vladislav Golyanik, Christian Theobalt, Amit H. Bermano.
  - arXiv 2507.17657, v1 of 2025-07-23, currently v2.
  - Venue: NeurIPS 2025, confirmed by the poster page https://neurips.cc/virtual/2025/poster/115284, whose title names the paper.
- **Abstract, VERIFIED:** "We introduce a new interpretation of the attention matrix as a discrete-time Markov chain." The body works with the post-softmax right-stochastic matrix.

**Grazzi et al. 2024 (added in v2)**
- **Title, authors, identifier and venue.** VERIFIED.
  - Title: "Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues".
  - Authors: Riccardo Grazzi, Julien Siems, Arber Zela, Jörg K. H. Franke, Frank Hutter, Massimiliano Pontil.
  - arXiv 2411.12537, v1 of 2024-11-19, currently v5.
  - Venue: ICLR 2025 oral, confirmed by https://iclr.cc/virtual/2025/oral/31836, whose title names the paper.
- **v2 of the arXiv version corrected two theorems.** Per its comment, v2 corrected Theorems 1 and 2 and point 3 of Proposition 1.
- **The claim as the contract states it.** VERIFIED in the arXiv HTML of v5: "We show that the eigenvalue range of Mamba and DeltaNet can be extended to [−1,1] without compromising efficiency or training stability."
- **The [0,1] parity failure for diagonal models is Sarrof et al.'s result, not Grazzi's.** VERIFIED. The abstract credits Sarrof et al. (2024) with the [0,1] failure for diagonal LRNNs. Grazzi et al. prove the general form: finite-precision LRNNs whose eigenvalues are all positive cannot solve parity.

**DeltaProduct, Siems et al. (added in v2)**
- **Title, authors, identifier and venue.** VERIFIED.
  - Title: "DeltaProduct: Improving State-Tracking in Linear RNNs via Householder Products".
  - Authors: Julien Siems, Timur Carstensen, Arber Zela, Frank Hutter, Massimiliano Pontil, Riccardo Grazzi.
  - arXiv 2502.10297, v1 of 2025-02-14, currently v7. The comment reports acceptance at NeurIPS 2025.
  - Venue: NeurIPS 2025, confirmed by https://neurips.cc/virtual/2025/poster/117900, whose title names the paper.
- **"Products of n_h generalized Householders per token" is correct.** VERIFIED from the abstract. The state transition is diagonal plus rank n_h, built as a product of n_h generalized Householder transformations.
- **"With negative eigenvalues" is supported.** VERIFIED. Each factor has eigenvalues 1 (multiplicity n−1) and 1−β_i. The appendix proofs take β ∈ [0,2]; the case β ∈ (1,2] gives 1−β < 0.
- **The v2 [U] is resolved: the paper itself states the rotation and names the groups.** v2 §8a calls the two-reflection rotation linear algebra rather than a quoted claim, and marks the exact groups [U]. VERIFIED in the arXiv HTML of v7:
  - The Figure 3 caption reads "Two reflections produce a 2D rotation".
  - The body identifies A₅ with the rotation group of the dodecahedron, and says such groups need only n_h = 2 with keys of size 3 (its Theorem 4).
  - Table 1 lists S_n, Z_n, D_n, O(n) and SO(n).

## 3. The operator against the contract's conventions

The file is `ceq/arm_smprime.py`, last commit f655a595, with no working-tree changes.

- **Z sums |G_ij| e^{s_ij}.** VERIFIED.
  - s_ij = qk·(q_i·k_j)/√d (line 297), and e = exp(s) (line 301).
  - The modulus row is returned as `rh * e` (line 324), where `rh` is `path_product(m)` = ∏ m_k (line 211).
  - Z is `mod.sum(-1) ** beta` (line 345), and the operator divides by it (line 346).
  - On a random float64 draw, max ||G_ij| − R_ij| = 1.1e−16.
  - So Z_i = Σ_j |G_ij| e^{s_ij}, with the trained `qk` inside s.
- **G is the path product, and G_ii = 1.** VERIFIED.
  - `path_product` (lines 149–177) computes G_ij = ∏_{k=j+1}^{i} a_k, with a_k = m_k e^{iθ_k}.
  - It does this as a reversed cumprod (line 174), shifted by one position (line 176), with zeros above the diagonal (line 177).
  - `torch.equal(diag(G), 1)` returns True.
- **m_i enters G_{i,i−1}.** VERIFIED. G_{i,i−1} = a_i = m_i e^{iθ_i}.
  - max ||G_{i,i−1}| − m_i| = 1.1e−16.
  - G[3,0] equals m₁m₂m₃e^{i(θ₁+θ₂+θ₃)} to within 4.2e−17.
- **How the magnitude m is computed.** VERIFIED.
  - m = magnitude(lerp(1, u, g)) (line 134). The default `magnitude` is clamp(u, 0, 1) (line 118).
  - For arm (f), `tests/foreman/design4x5/design4x5.py:334,342` swaps in `R.FORMS["hard_concrete"]`, which is clamp(1.2·sigmoid(u) − 0.1, 0, 1) (`tests/chase/gate/r1_gate.py:141-149`).
- **β is a trained parameter, not fixed at 1.** VERIFIED.
  - The default argument is `beta=1.0` (line 329).
  - β is an `nn.Parameter` in `ArmSMPrime` (line 723, initialised to 1.0) and in the HF LM (`ceq/hf/modeling_ceq.py:444`, from `config.smp_beta`, used at line 487).
- **Trained values in `d45_ckpt_f_ss0_pair/model.pt`: β is 1.061, 0.930 and 0.900 by layer.** VERIFIED, read on CPU. The file's top-level keys are `state_dict, n_params, operator, seed, split_seed`, with `operator=smprime`, `seed=0`, `split_seed=0` and `n_params=725391`.

  | tensor `model.layers.N.self_attn.*` | layer 0 | layer 1 | layer 2 |
  |---|---|---|---|
  | `beta` | 1.0605748891830444 | 0.9300132989883423 | 0.8995795845985413 |
  | `qk` | 1.0872255563735962 | 1.33349609375 | 1.3895158767700195 |
  | `g` | 0.9823850989341736 | 1.009325623512268 | 0.9927084445953369 |
  | `m_head.bias` | 0.9179627299308777 | 0.9315248131752014 | 0.9418838024139404 |
  | `theta_head.bias` | 0.048726487904787064 | −0.05971629172563553 | −0.030459320172667503 |

  `m_head.weight` has shape (1,128) in each layer.
- **The state dict alone cannot show exact zeros of m, because m depends on the input.** So Wilson ran a CPU forward pass (`gate_zeros.py`, output in `gate_zeros_out.txt`).
  - Setup: hard-concrete gate installed; `DocByteBatches(val_frac=0.1, split_seed=0)`; 8 validation batches of 8×512 drawn with `Generator(20260921)`, the eval seed `run_j.py` uses.
  - Result, 32,768 positions per layer. VERIFIED.

  | layer | exact zeros | exact ones | m < 0.05 | min m | median m |
  |---|---|---|---|---|---|
  | 0 | 0 | 0 | 65 | 0.04717 | 0.6730 |
  | 1 | 0 | 0 | 0 | 0.09669 | 0.4406 |
  | 2 | 0 | 0 | 0 | 0.09255 | 0.5300 |

- **The earlier rowmass measurement is of a different checkpoint.** VERIFIED.
  - `tests/chase/rowmass/step0_rowmass.md:53-63` stopped because the q2 arm (f) directories hold only `run_record.json`.
  - `rowmass_opus_check.py` then measured `q3_ckpt_seed0_hard_concrete`. That checkpoint is smprime with the hard-concrete gate, but it has 4 layers, hidden size 512 and 800 steps. It is neither the d45 checkpoint nor the q2 one.
  - Its results are in `house-events.jsonl:4473` and `:4480`, and in `scratchpad/rowmass_opus_check.json`:

  | quantity, by layer | layer 0 | layer 1 | layer 2 | layer 3 |
  |---|---|---|---|---|
  | β | 1.0907 | 0.9616 | 0.9624 | 0.9913 |
  | row mass, mean | 0.9607 | 0.7930 | 0.7971 | 0.7423 |
  | row mass, min | 0.00483 | 0.01053 | 0.00957 | 0.00232 |
  | fraction of rows with mass < 0.5 | 0.00156 | 0.1322 | 0.1134 | 0.1651 |
  | exact-zero fraction of m | 0.0 | 0.00244140625 (10/4,096) | 0.0 | 0.0 |
  | exact-zero fraction of W, causal entries (8,404,992 per layer) | 0.6686 | 0.7542 | 0.7143 | 0.5938 |

## 4. BOS on the byte-level TinyStories bed

- **The byte-level bed has no BOS token.** VERIFIED from `tests/foreman/eval/r3_eval.py`, `class DocByteBatches` (lines 56–96).
  - Documents are split on "\n\n" (line 69) and joined back with "\n\n" (line 79).
  - Bytes are the UTF-8 encoding mod `vocab_size=256` (line 80), so no id is reserved outside 0–255.
  - `batch()` takes windows at random offsets, `torch.randint(len(d) - seq - 1, …)` (line 93), so position 0 of a window is an arbitrary byte.
  - `grep -ic bos` returns 0 in `r3_eval.py`, `ceq/hf/modeling_ceq.py`, `ceq/hf/train.py` and `ceq/hf/configuration_ceq.py`.
  - The d45 `embed_tokens.weight` has shape (256,128).

## 5. Mathlib (read-only)

The project copy is `lean/.lake/packages/mathlib` at rev a45ae63747140c1b2cbad9d46f518015c047047a, dated 2024-04-04. `lean/lake-manifest.json` gives inputRev v4.7.0.

- **`Matrix.det_of_lowerTriangular` exists.** VERIFIED at `Mathlib/LinearAlgebra/Matrix/Block.lean:265`, inside `namespace Matrix` (line 46):
  `theorem det_of_lowerTriangular [LinearOrder m] (M : Matrix m m R) (h : M.BlockTriangular toDual) : M.det = ∏ i : m, M i i`
  `det_of_upperTriangular` is at line 259.
- **No lemma in this copy states directly that the characteristic polynomial is invariant under conjugation.** VERIFIED. A grep of `Mathlib/` for `charpoly_units_conj`, `charpoly_conj`, `charpoly_mul_comm`, `charpoly_of_similar` and `IsSimilar` returns nothing. The nearest lemmas present:
  - `LinearMap.charpoly_toMatrix` (`Mathlib/LinearAlgebra/Charpoly/ToMatrix.lean:46`) states `(toMatrix b b f).charpoly = f.charpoly` for any basis b.
  - `Matrix.charpoly_reindex` (`Mathlib/LinearAlgebra/Matrix/Charpoly/Basic.lean:107`) covers reindexing by an equivalence only.
  - `Matrix.det_conj` (`Mathlib/LinearAlgebra/Matrix/NonsingularInverse.lean:783`) states `(h : IsUnit M) (N) : det (M * N * M⁻¹) = det N`. That is the determinant, not the characteristic polynomial.
- **Newer Mathlib: UNVERIFIED.** Whether a later Mathlib version has such a lemma was not checked.
