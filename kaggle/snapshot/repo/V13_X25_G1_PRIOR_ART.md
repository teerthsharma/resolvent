# V13 / X₂₅ — G1 PRIOR ART: eager verb operators and dataflow-scheduled execution

Contract item X₂₅, global stop G1. Fetched 2026-08-31. Normal English by the
artifact exemption.

**Scope.** Four lineages are covered: neural module networks; recursive networks
over structure; program/computation graphs as networks (dataflow readiness);
verbs-as-functions. Markov state models, energy-based models, committor
learning, saddle-search and the options/hierarchical-RL framework belong to
`V13_TIER6_PRIOR_ART.md` and are not covered here. One overlap is flagged in
§3.4 and deferred.

**Marks.** `[V]` = source fetched and read this session. `[V-t]` = title/abstract
confirmed this session, body not read. `[U]` = owed, unfetched. Nothing is marked
`[V]` from memory.

---

## Verdict

Dataflow-scheduled eager execution inside a neural forward pass is published
whole: DAGNN (arXiv:2101.07965) and D-VAE (arXiv:1904.11088) each compute a
node's state only after every predecessor's state exists, and D-VAE says in as
many words that this encodes *computations* rather than *structures*.
Predicate-as-operator is likewise published whole: MV-RNN (Socher, Huval,
Manning, Ng, EMNLP 2012) gives every word an $n \times n$ matrix and composes by
$p = g(W[Ba;\,Ab])$, and the categorical-compositional line learns a matrix per
relational word and applies it to argument vectors. Neither half is available to
X₂₅ as novelty. The composite — one operator indexed by the verb *lexeme*, fired
by operand readiness, inside a language model's forward pass — was not found
under any name in this session's searches, so G1 does not fire on the item as
scoped. The most dangerous single overlap is Thost & Chen's DAGNN, whose §2.2
scheduling rule ("DAGNN processes nodes sequentially owing to the nature of the
aggregator $\mathcal{G}^\ell$, obeying the partial order") is X₂₅'s scheduling
rule with the words changed; the runner-up is Gupta et al.'s text-domain module
network (arXiv:1912.04971), which already predicts an operator tree over a
paragraph and executes it bottom-up over token representations. X₂₅'s residual
novelty is therefore the *binding of operator identity to the verb lexeme rather
than to a fixed typed module inventory*, plus the domain — the scheduling rule
is not novel and no claim sentence may rest on it.

---

## 1. Neural Module Networks

### 1.1 NMN — Andreas, Rohrbach, Darrell, Klein `[V]`

arXiv:1511.02799 (v1 2015-11-09, v4 2017-07-24), CVPR 2016. The v4 `Comments`
field reads: *"Corrects an error in the evaluation of the NMN-only ablation
experiment."*

**Composition.** §4.1 of the arXiv text gives the modules as *type signatures
with prose*, not as displayed formulas — this is a fetch finding, not an
omission here:

$$\mathrm{mod}_{\text{attend}} : \text{Image} \to \text{Attention}, \quad
\mathrm{mod}_{\text{re-attend}} : \text{Attention} \to \text{Attention}$$
$$\mathrm{mod}_{\text{combine}} : \text{Attention} \times \text{Attention} \to \text{Attention}$$
$$\mathrm{mod}_{\text{classify}} : \text{Image} \times \text{Attention} \to \text{Label}, \quad
\mathrm{mod}_{\text{measure}} : \text{Attention} \to \text{Label}$$

The displayed algebraic forms appear only in the CVPR camera-ready, which did not
resolve this session (see Owed, O-1). The type discipline is nevertheless the
load-bearing part: a module is a *typed function over the representations its
children produced*, exactly X₂₅'s $f_{\text{verb}}(h_{\text{args}})$ shape.

**Layout and execution.** §4.2: a dependency parse yields a symbolic query
(*"what color is the truck"* $\to$ `mod_color(truck)`), and the assignment is
*"fully determined by the structure of the parse"* — *"All leaves become
mod_attend modules, all internal nodes become mod_re-attend or mod_combine
modules dependent on their arity, and root nodes become mod_measure modules for
yes/no questions and mod_classify modules for all other question types."* No
soft relaxation over layouts: the layout is built deterministically and then run
as a fixed computation graph.

> **Delta.** NMN parses an utterance into a typed operator tree and executes it
> eagerly over sub-representations; X₂₅ parses an utterance into a typed
> operator graph and executes it eagerly over sub-representations. The
> difference is the *indexing of the operator*: NMN's operator identity comes
> from a closed inventory of five signature classes plus a per-word parameter
> $c$, while X₂₅'s comes from the verb lexeme itself; and NMN's structure is a
> tree, where eager evaluation and post-order traversal coincide, so nothing is
> gained by calling the schedule "dataflow".

### 1.2 D-NMN — "Learning to Compose Neural Networks for Question Answering" `[V]`

arXiv:1601.01705, NAACL 2016. Displayed module semantics:

$$[\![\texttt{lookup}[i]]\!] = e_{f(i)}$$
$$[\![\texttt{find}[i]]\!] = \operatorname{softmax}(a \odot \sigma(B v^i \oplus C W \oplus d))$$
$$[\![\texttt{relate}[i](h)]\!] = \operatorname{softmax}(a \odot \sigma(B v^i \oplus C W \oplus D \bar{w}(h) \oplus e))$$
$$[\![\texttt{and}(h^1, h^2, \ldots)]\!] = h^1 \odot h^2 \odot \cdots$$
$$[\![\texttt{exists}(h)]\!] = \operatorname{softmax}\big((\textstyle\max_k h_k)\, a + b\big)$$
$$[\![\texttt{describe}[i](h)]\!] = \operatorname{softmax}(A\,\sigma(B \bar{w}(h) + v^i))$$

Layouts are fragments of a dependency parse assembled into candidates, then a
layout $z$ is *sampled* from $p(z \mid x; \theta_\ell)$ and executed.

> **Delta.** `relate[i](h)` is literally a relational predicate applied to the
> hidden state of its argument, with the predicate word $i$ selecting the
> parameter vector $v^i$. X₂₅ does the same with a verb selecting
> $f_{\text{verb}}$. The difference is that $v^i$ enters *additively into a
> shared operator* $(\,B v^i \oplus C W \oplus D\bar w(h)\,)$, whereas X₂₅
> proposes that the verb select the *function itself*. That is a real but small
> difference: an additive lexeme embedding inside one shared operator is the
> rank-1 special case of a lexeme-indexed operator family.

### 1.3 N2NMN — End-to-End Module Networks `[V]`

arXiv:1704.05526. Layout policy §3.2:
$$p(l \mid q) = \prod_{m^{(t)} \in l} p\big(m^{(t)} \mid m^{(1)}, \ldots, m^{(t-1)}, q\big)$$
Training §3.3 optimises $L(\theta) = \mathbb{E}_{l \sim p(l \mid q; \theta)}[\tilde L(\theta, l, q, I)]$
with the REINFORCE estimator
$$\nabla_\theta L \approx \tfrac{1}{M}\textstyle\sum_m \big[\tilde L(\theta, l_m)\nabla_\theta \log p(l_m \mid q;\theta) + \nabla_\theta \tilde L(\theta, l_m)\big].$$
Execution is over a **discrete sampled layout**, not a soft mixture.

> **Delta.** N2NMN learns the graph and executes it eagerly; X₂₅ takes the graph
> from the parse and executes it eagerly. The difference is that X₂₅ does not
> pay the REINFORCE variance N2NMN pays — which is a *training-cost* difference,
> not a capability difference, and cannot appear in a capability table.

### 1.4 Stack-NMN — the field's move away from eager execution `[V]`

arXiv:1807.08556. Soft module execution, §3.3:
$$A^{(t+1)} = \sum_{m \in M} A_m^{(t)} \cdot w_m^{(t)}$$
Differentiable stack, §3.2: push is $p := \texttt{1d\_conv}(p, [0,0,1])$,
$A_i := A_i(1-p_i) + z\,p_i$; pop is $z := \sum_{i=1}^{L} A_i p_i$,
$p := \texttt{1d\_conv}(p, [1,0,0])$. §3: *"the model makes soft layout selection
with a differentiable stack structure, by giving each module a continuous-valued
weight parameter and averaging the outputs of all modules according to their
weights. This makes the execution procedure fully differentiable so that our
model is trainable with back-propagation."*

> **Delta.** Stack-NMN replaces eager discrete execution with a soft relaxation
> and keeps the accuracy; X₂₅ proposes moving in the opposite direction, from a
> soft token-mixing lane to eager discrete execution. The difference is
> directional, and the direction the field took is the one that says the eager
> half was not what carried the performance. **This is the single strongest
> prior-art argument that X₂₅'s deciding test will show no separation.**

### 1.5 IEP — Inferring and Executing Programs `[V]`

arXiv:1705.03633. §3.2 serialises the program by *"prefix traversal to serialize
the syntax tree"*, argmax-decoded at test time, re-treeified because *"the arity
of each function is known"*. §3.3: *"the program $z$ is used to assemble a
question-specific neural network that is composed from a set of modules"*; unary
modules are a residual block with two $3\times3$ convolutions, binary modules
concatenate on channels and project $2C \to C$ with a $1\times1$ convolution.
§3.4 uses REINFORCE with negative zero-one loss as reward.

> **Delta.** IEP executes a discrete predicted operator tree eagerly, with arity
> supplied by the function type — the identical mechanism to X₂₅'s eager firing,
> down to arity-driven graph reconstruction from a linear sequence. The
> difference is zero on the execution mechanism; the difference is the operator
> inventory and the modality.

### 1.6 Text-domain module networks — Gupta et al. `[V]`

arXiv:1912.04971. Program prediction §2.1 is grammar-constrained top-down
decoding producing *"a linearized abstract syntax tree (in an in-order
traversal)"*, marginalised as $J = \sum_{\mathbf z} p(y^* \mid \mathbf z)\,p(\mathbf z \mid q)$.
Representative module semantics §3.2:
$$\texttt{find}: \quad S_{ij} = w_f^\top[\,Q_{i:};P_{j:};Q_{i:}\circ P_{j:}\,], \qquad P = \textstyle\sum_i Q_i \cdot A_{i:}$$
$$\texttt{filter}: \quad M_j = \sigma\big(w_{\text{filter}}^\top[\,q;P_{j:};q\circ P_{j:}\,]\big), \qquad P_{\text{filtered}} = \operatorname{normalize}(M \circ P)$$
$$\texttt{relocate}: \quad R_{ij} = w_{\text{relocate}}^\top[\,(q+P_{i:});P_{j:};(q+P_{i:})\circ P_{j:}\,], \qquad P_{\text{reloc}} = \textstyle\sum_i P_i \cdot R_{i:}$$
$$\texttt{compare-num-lt}: \quad P_{\text{out}} = p(N_1<N_2)\,P_1 + p(N_2<N_1)\,P_2$$

The paper states `relocate` *"is used to find the arguments for paragraph
spans"* — i.e. an operator over predicate arguments, in text, executed on a
predicted tree.

> **Delta.** Gupta et al. run a parsed operator tree eagerly over token
> representations in a text task with argument-finding operators; X₂₅ runs a
> parsed operator graph eagerly over token representations in a text task with
> verb operators. The difference is that Gupta's operators are drawn from a
> hand-designed 10-module inventory typed by *answer semantics* (numbers, dates,
> counts) rather than by the verb lexeme, and the tree is a tree. This is the
> closest published work to X₂₅ in the same modality.

---

## 2. Recursive networks over structure

### 2.1 Tree-LSTM — Tai, Socher, Manning `[V]`

arXiv:1503.00075. Child-Sum, §3.1:
$$\tilde h_j = \sum_{k \in C(j)} h_k$$
$$i_j = \sigma(W^{(i)}x_j + U^{(i)}\tilde h_j + b^{(i)}), \qquad f_{jk} = \sigma(W^{(f)}x_j + U^{(f)}h_k + b^{(f)})$$
$$o_j = \sigma(W^{(o)}x_j + U^{(o)}\tilde h_j + b^{(o)}), \qquad u_j = \tanh(W^{(u)}x_j + U^{(u)}\tilde h_j + b^{(u)})$$
$$c_j = i_j \odot u_j + \sum_{k \in C(j)} f_{jk}\odot c_k, \qquad h_j = o_j \odot \tanh(c_j)$$
$N$-ary, §3.2, replaces $U^{(\cdot)}\tilde h_j$ with $\sum_{\ell=1}^{N} U^{(\cdot)}_{\ell} h_{j\ell}$
and $f_{jk} = \sigma(W^{(f)}x_j + \sum_{\ell} U^{(f)}_{k\ell}h_{j\ell} + b^{(f)})$.

**Structure provenance: given.** §5.1: *"Standard binarized constituency parse
trees are provided for each sentence in the dataset"*; dependency parses from the
Stanford Neural Network Dependency Parser and constituency parses from the
Stanford PCFG Parser (footnotes, §5).

> **Delta.** Tree-LSTM composes children into a parent by a *position-indexed*
> gate family $U_\ell$ — the operator depends on the child's slot, never on the
> parent word's identity, and $x_j$ enters only additively. X₂₅ makes the parent
> word select the operator. The difference is real: Tree-LSTM's $U_\ell$ is
> shared across all parents, so it cannot express "a different function per
> verb" except through the additive $W x_j$ term.

### 2.2 MV-RNN — Socher, Huval, Manning, Ng `[V]`

EMNLP 2012, ACL Anthology D12-1110. §2:
$$p = f_{A,B}(a,b) = g\big(W[Ba;\,Ab]\big), \qquad P = W_M[A;B]$$
Each word carries both a vector and an $n\times n$ *"matrix that operates on"*
the other constituent's vector.

> **Delta.** MV-RNN is verb-as-operator already: the predicate's matrix $B$
> multiplies the argument's vector $a$ before composition, and the operator is
> indexed by the *lexeme*, which is exactly the indexing X₂₅ claims as its
> residual novelty. The difference is that MV-RNN applies *both* words' matrices
> symmetrically ($Ba$ *and* $Ab$) and then mixes through a shared $W$, so it
> never commits to an asymmetric $f_{\text{verb}}(h_{\text{args}})$; and its
> schedule is a post-order walk of a given binary parse, not a readiness rule.
> On the operator-indexing axis alone, **the difference is close to zero.**

### 2.3 SPINN — Bowman et al. `[V]`

arXiv:1603.06021. Composition §3.2:
$$[\vec i, \vec f_l, \vec f_r, \vec o, \vec g] = [\sigma,\sigma,\sigma,\sigma,\tanh]\big(W_{\text{comp}}[\vec h_s^1, \vec h_s^2, \vec e] + \vec b_{\text{comp}}\big)$$
$$\vec c = \vec f_l \odot \vec c_s^2 + \vec f_r \odot \vec c_s^1 + \vec i \odot \vec g, \qquad \vec h = \vec o \odot \tanh(\vec c)$$
Transitions §3.1: $\textsf{shift}: \langle S, x\mid B\rangle \to \langle x\mid S, B\rangle$;
$\textsf{reduce}: \langle x\mid y\mid S, B\rangle \to \langle (x,y)\mid S, B\rangle$.
Introduction: *"SPINN executes the computations of a tree-structured model in a
linearized sequence"*, and the model *"can incorporate a neural network parser
that produces the required parse structure on the fly."*

> **Delta.** SPINN answers X₂₅'s scheduling question in the opposite direction:
> it shows that a structure-respecting computation can be *re-linearised* into
> token order without changing what is computed. The difference is that X₂₅
> claims the schedule matters; SPINN is a published demonstration that on trees
> the schedule is a performance detail, because shift/reduce order and eager
> evaluation produce identical values.

### 2.4 Gumbel Tree-LSTM — structure learned, not given `[V-t]`

arXiv:1707.02786, AAAI 2018. Abstract: *"learns how to compose task-specific
tree structures only from plain text data"* using a Straight-Through
Gumbel-Softmax estimator, and *"outperforms or is at least comparable to
previous models."*

> **Delta.** X₂₅ takes the operator graph from the verb's argument structure —
> given, not learned. Gumbel Tree-LSTM is evidence that the *given* structure was
> not carrying the result on these tasks: learned, task-specific structure
> matched parser-given structure. The difference is that X₂₅ has not yet shown
> its planted bed makes the given structure load-bearing, and this paper is the
> reason that burden exists.

### 2.5 Goller & Küchler — backpropagation through structure `[U]`

`doi:10.1109/ICNN.1996.548916`, ICNN'96, pp. 347–352. Identifier resolved from a
third-party index only; the source was not fetched. Owed as O-2.

---

## 3. Program and computation graphs as networks — the dataflow question

This is the sharpest section. The question posed to G1 was: *does an architecture
already exist in which node firing order is determined by operand availability
inside a neural forward pass?* **Yes. Twice, explicitly, at equation level.**

### 3.1 DAGNN — Thost & Chen `[V]`

arXiv:2101.07965. §2.1:
$$m_v^\ell := \mathcal{G}^\ell\big(\{h_u^\ell : u \in \mathcal{P}(v)\},\, h_v^{\ell-1}\big) = \sum_{u \in \mathcal{P}(v)} \alpha_{vu}^\ell h_u^\ell$$
$$h_v^\ell = \mathcal{F}^\ell\big(h_v^{\ell-1}, m_v^\ell\big) = \mathrm{GRU}^\ell\big(h_v^{\ell-1}, m_v^\ell\big)$$
§2.2, verbatim: *"A key difference to MPNN is that DAGNN processes nodes
sequentially owing to the nature of the aggregator $\mathcal{G}^\ell$, obeying
the partial order."* The topological-batching procedure: *"All nodes without
direct predecessors form the initial batch. Iteratively, remove the batch just
formed from the graph, as well as the edges emitting from these nodes. The nodes
without direct predecessors in the remaining graph form the next batch."*
Theorem 1 gives the minimality of that batching.

> **Delta.** DAGNN fires a node when and only when all its operands are
> available, inside a neural forward pass, and batches the frontier optimally.
> X₂₅ fires an operator when and only when all its operands are available,
> inside a neural forward pass. **The difference is zero on the scheduling
> rule.** What remains to X₂₅ is that its nodes are verbs and its DAG comes from
> predicate-argument structure rather than from a given graph input.

### 3.2 D-VAE — Zhang, Jiang, Cui, Garnett, Chen `[V]`

arXiv:1904.11088. §3.1:
$$h_v = \mathcal{U}(x_v, h_v^{\text{in}}), \qquad h_v^{\text{in}} = \mathcal{A}\big(\{h_u : u \to v\}\big)$$
§3 opening, verbatim: *"d-vae uses an asynchronous message passing scheme to
encode and decode dags. In contrast to the simultaneous message passing in
traditional gnns, d-vae allows encoding computations rather than structures."*
§3.1: *"in d-vae the message passing for a node must wait until all of its
predecessors' hidden states have already been computed"*, achieved by *"message
passing for nodes following a topological ordering of the dag."*

> **Delta.** D-VAE states X₂₅'s thesis — that scheduling by operand readiness
> makes a network encode a *computation* rather than a *structure* — as its own
> headline contribution, three years earlier, with a proof of injectivity over
> computations. The difference is zero on the thesis; the difference is that
> D-VAE's DAGs are neural-architecture and Bayesian-network graphs, not sentences.

### 3.3 Dynamic batching / TensorFlow Fold — Looks et al. `[V]`

arXiv:1702.02181, §2, verbatim: *"Assign a depth to each node in the graph. Nodes
with no dependencies (constants) are assigned depth zero. Nodes with only
dependencies of depth zero are assigned depth one, nodes whose dependencies have
a maximum depth of one get assigned depth two, etc."*; then *"Insert
pass-through (identity) operations so that an operation at depth $d+1$ only
refers to results at depth $d$"* and *"Batch together all nodes invoking the same
operation at the same depth into a single node."* Execution: *"Each iteration of
the loop will evaluate all of the operations at a particular depth."*

> **Delta.** Fold schedules neural operations by input availability (depth), not
> sequence position, and does so for the express purpose of batching
> per-example-varying graphs — which is the engineering problem X₂₅'s eager arm
> will hit on the first batch. The difference is zero on the scheduling
> principle; the difference is that Fold treats it as an *implementation*
> concern with no claimed effect on what the model computes. That framing is
> itself adverse evidence for X₂₅: the field's position is that dataflow
> scheduling changes throughput, not function.

### 3.4 Neural Programmer-Interpreters — Reed & de Freitas `[V]`

arXiv:1511.06279, §3.1:
$$s_t = f_{\text{enc}}(e_t, a_t), \qquad h_t = f_{\text{lstm}}(s_t, p_t, h_{t-1})$$
$$r_t = f_{\text{end}}(h_t), \quad k_t = f_{\text{prog}}(h_t), \quad a_{t+1} = f_{\text{arg}}(h_t)$$
$$i^* = \arg\max_{i=1..N} (M^{\text{key}}_{i,:})^\top k_t, \qquad p_{t+1} = M^{\text{prog}}_{i^*,:}$$
Control flow is a stack machine: *"control is returned to the caller by popping
the caller's LSTM hidden units and program embedding off of a program call
stack"* (§3.1, Algorithm 1).

> **Delta.** NPI selects an operator by key-addressed lookup $\arg\max_i
> (M^{\text{key}}_{i,:})^\top k_t$ — a learned, content-addressed operator
> inventory, which is the mechanism X₂₅ would need if verb operators are to be
> shared across a large lexicon rather than stored one-per-verb. The difference
> is that NPI's schedule is a *call stack driven by the controller's own
> sequence*, not by operand readiness: it is eager but not dataflow.
> **Sibling overlap (deferred):** NPI's subprogram call/return structure is
> also the options framework's initiation/termination structure; that reading
> belongs to `V13_TIER6_PRIOR_ART.md` and is not developed here.

### 3.5 Graph networks — Battaglia et al. `[V]`

arXiv:1806.01261, §3.2.2:
$$e'_k = \phi^e(e_k, v_{r_k}, v_{s_k}, u), \quad v'_i = \phi^v(\bar e'_i, v_i, u), \quad u' = \phi^u(\bar e', \bar v', u)$$
$$\bar e'_i = \rho^{e\to v}(E'_i), \quad \bar e' = \rho^{e\to u}(E'), \quad \bar v' = \rho^{v\to u}(V')$$
Algorithm 1 fixes a six-step order, but §3.2.3 states: *"Note, though we assume
this sequence of steps here, the order is not strictly enforced: it is possible
to reverse the update functions to proceed from global, to per-node, to
per-edge updates, for example."*

> **Delta.** The GN block is the canonical *synchronous* alternative: every node
> updates every round regardless of readiness. X₂₅ is the asynchronous variant of
> this. The difference is exactly the difference DAGNN (§3.1) and AMP (§3.6)
> already named and measured, so X₂₅ inherits their result rather than
> establishing it.

### 3.6 Asynchronous message passing — Faber & Wattenhofer `[V-t]`

arXiv:2205.12245. Abstract: *"Existing graph neural networks use the synchronous
distributed computing model and aggregate their neighbors in each round, which
causes problems such as oversmoothing and limits their expressiveness ... AMP is
based on the asynchronous model, where nodes react to messages of their
neighbors individually. We prove that (i) AMP can simulate synchronous GNNs and
that (ii) AMP can theoretically distinguish any pair of graphs."*

> **Delta.** AMP proves the asynchronous model strictly simulates the
> synchronous one and separates graphs the synchronous one cannot. X₂₅ needs
> exactly such a separation argument for verbs. The difference is zero on the
> existence of the separation mechanism; the difference is that AMP's separation
> is a graph-isomorphism-expressiveness result, which does not transfer to a
> language task without a bed that plants the distinguishing structure.

### 3.7 Neural GPU `[V-t]` and DNC `[V-t]` — negative controls

Neural GPU, arXiv:1511.08228: *"the Neural GPU is highly parallel"*, contrasted
with the NTM's *"sequential nature ... not parallel and hard to train due to
their large depth when unfolded."* DNC: Graves, Wayne, Reynolds et al., *Hybrid
computing using a neural network with dynamic external memory*, Nature 538(7626)
471–476, 2016, `doi:10.1038/nature20101` — title, venue and DOI confirmed this
session from index records; the article body is paywalled and was not fetched.

> **Delta.** Both compute a *fixed* number of steps in a fixed order and route
> data through memory addressing rather than through a schedule. X₂₅ varies the
> number and order of operator firings with the input's argument structure. The
> difference is real but it is the same difference DAGNN already owns.

### 3.8 "Eager" as a term of art `[V-t]`

TensorFlow Eager, arXiv:1903.01855: *"an imperative front-end to TensorFlow that
executes operations immediately and a JIT tracer that translates Python
functions composed of TensorFlow operations into executable dataflow graphs."*

> **Delta.** In the framework literature "eager" means *executes immediately when
> called*, and it is opposed to *staged/graph* execution — not to token-order
> execution. X₂₅'s use of "eager" means *fires when operands are ready*, which is
> the dataflow-graph side of that opposition, not the eager side. The difference
> is terminological, and the term as X₂₅ uses it will be read backwards by
> anyone from systems; the item should say "operand-ready firing", not "eager".

---

## 4. Verbs as functions

### 4.1 MV-RNN — see §2.2 `[V]`

The strongest single hit for lexeme-indexed operators; delta recorded there.

### 4.2 Categorical compositional distributional semantics `[V-t]`

Coecke, Sadrzadeh, Clark, *Mathematical Foundations for a Compositional
Distributional Model of Meaning*, arXiv:1003.4394 (2010). Abstract: a framework
unifying *"the distributional theory of meaning in terms of vector space models,
and a compositional theory for grammatical types, for which we rely on the
algebra of Pregroups"*, so that pregroup type reductions are *"lifted to
morphisms in a category, a procedure that transforms meanings of constituents
into a meaning of the (well-typed) whole."*

Grefenstette & Sadrzadeh, *Experimental Support for a Categorical Compositional
Distributional Model of Meaning*, EMNLP 2011, arXiv:1106.4058. Abstract: *"The
implementation is based on unsupervised learning of matrices for relational words
and applying them to the vectors of their arguments"*, evaluated on intransitive
and transitive sentences, with *"general improvement in results with increase in
syntactic complexity."*

> **Delta.** This lineage *is* $f_{\text{verb}}(h_{\text{args}})$: a noun is a
> vector, an intransitive verb is a matrix $\mathbb{R}^n \to \mathbb{R}^n$, a
> transitive verb is a third-order tensor $\mathbb{R}^n \otimes \mathbb{R}^n \to
> \mathbb{R}^n$, the arity comes from the grammatical type, and the application
> order comes from the parse. **The difference on the semantics of
> $f_{\text{verb}}$ is zero** — X₂₅ must cite this line for what $f_{\text{verb}}$
> *means*. What is left to X₂₅ is that these are learned as fixed multilinear
> maps over static distributional vectors, not as modules applied to
> contextual hidden states inside a trained network with a schedule.

### 4.3 CCG-typed tensor semantics `[U]`

Krishnamurthy & Mitchell, *Vector Space Semantic Parsing: A Framework for
Compositional Vector Space Models*, ACL Anthology `W13-3201`, CVSC 2013. The PDF
was fetched but did not yield extractable text this session; equations are owed
as O-3. The framework assigns tensors to CCG syntactic categories so that
function application becomes a tensor contraction, with composition order fixed
by the CCG derivation.

> **Delta.** Provisionally, the difference is the same as §4.2 — type-driven
> operator application over a given derivation — with the addition that CCG makes
> the *arity and directionality* of a verb part of its lexical entry, which is
> what X₂₅ needs to schedule. Marked `[U]` because no equation was read.

### 4.4 Neural semantic role labelling treats the predicate as a token `[V]`

*Neural-Davidsonian Semantic Proto-role Labeling*, arXiv:1804.07976, §3:
$$h_{ea} = [h_e; h_a], \qquad \mathrm{Score}(\text{attr}, h_{ea}) = W_{\text{attr}}\big[g(W_{\text{shared}}[h_{ea}])\big]$$
where $h_e$ and $h_a$ are BiLSTM states at the syntactic heads of the predicate
and the argument. The predicate representation is *concatenated with* the
argument representation; it does not act on it.

> **Delta.** This is the exact baseline X₂₅ calls "verbs-as-tokens", published
> under a Davidsonian banner: the neo-Davidsonian framing is used to define the
> *label space* (attributes of $(e,a)$ pairs), while the *computation* stays
> symmetric concatenation. The difference between this and X₂₅ is precisely the
> difference the deciding test is built to measure, which makes this the correct
> citation for the control arm rather than for the treatment arm.

### 4.5 Executable semantic parsing with a symbolic executor `[V]`, `[V-t]`

NS-CL, arXiv:1904.12584 `[V]`, §3.1: *"Our program executor works in a symbolic
and deterministic manner. This feature ensures a transparent execution trace of
the program"*, over probabilistic object masks, with concept scoring
$\sigma\big((\langle \texttt{ShapeOf}(o_i), v_{\texttt{Cube}}\rangle - \gamma)/\tau\big)$
and operators such as $\texttt{out}_i := \min(\texttt{in}_i, \texttt{ObjClassify}(oc)_i)$
(Appendix C).

LEFT, *What's Left? Concept Grounding with Logic-Enhanced Foundation Models*,
arXiv:2310.16035, NeurIPS 2023 `[V-t]`: *"a differentiable, domain-independent,
first-order logic-based program executor"* with an LLM producing the program and
trainable domain-specific grounding modules executing it.

> **Delta.** Both execute a parsed predicate-argument program eagerly and
> deterministically over learned representations, with predicates realised as
> differentiable operators — the whole of X₂₅'s pipeline except that the
> predicate vocabulary is a closed concept set and the schedule is a plain
> post-order walk. The difference is the open verb lexicon and the DAG schedule;
> the difference on *"predicate becomes a differentiable operator over argument
> representations, run on a parse"* is zero.

### 4.6 Linguistic framing for what $f_{\text{verb}}$ should mean `[U]`

Davidson's event argument and Parsons' neo-Davidsonian decomposition fix the
target: a verb denotes a predicate of events, and thematic roles are separate
binary relations, so $\texttt{eat}(e) \wedge \mathrm{Agent}(e, x) \wedge
\mathrm{Patient}(e, y)$ replaces $\texttt{eat}(x,y)$. Under that reading
$f_{\text{verb}}$ is not an $n$-ary function of its arguments at all: it is a
unary predicate on an event node, and the arguments attach by *typed edges*.
Primary sources unfetched — O-4.

> **Delta.** Neo-Davidsonian semantics says X₂₅'s $f_{\text{verb}}(h_{\text{args}})$
> is the *Davidsonian* (fixed-arity) form, not the neo-Davidsonian one. The
> difference matters for the architecture: a neo-Davidsonian $f_{\text{verb}}$
> would be a node-feature update plus role-typed edges — which is a GN block
> (§3.5) with edge types, i.e. it collapses X₂₅ into an existing architecture.
> If the item wants the collapse not to happen, it must commit to fixed arity
> and say so.

---

## Owed

| id | Item | Why still `[U]` |
|----|------|-----------------|
| O-1 | NMN CVPR 2016 camera-ready §4.1 displayed module formulas | CVF open-access returned HTTP 403 and the Berkeley mirror refused the connection; the ar5iv render of arXiv:1511.02799 carries type signatures and prose only. The algebraic forms of `attend`/`re-attend`/`combine`/`classify`/`measure` are unread. Needed only if a claim compares X₂₅'s operator algebra to NMN's; the §1.1 delta does not depend on it. |
| O-2 | Goller & Küchler 1996, `doi:10.1109/ICNN.1996.548916` | Identifier taken from a third-party index; IEEE text not fetched. Historical priority for backpropagation-through-structure; affects attribution, not novelty. |
| O-3 | Krishnamurthy & Mitchell 2013, ACL `W13-3201` | PDF fetched but text extraction failed. Its CCG-to-tensor mapping is the cleanest statement of arity-from-type, which is the rule X₂₅ needs to build the DAG. Highest-value outstanding fetch. |
| O-4 | Davidson 1967; Parsons 1990 | Book/chapter sources, not resolvable by URL fetch this session. Needed to fix the arity commitment in §4.6. |
| O-5 | Socher, Lin, Ng, Manning, ICML 2011, *Parsing Natural Scenes and Natural Language with Recursive Neural Networks* | PDF fetched, text extraction failed. Supplies the "structure searched jointly with composition" data point; §2.4 covers the same point with a source that did resolve. |
| O-6 | Socher et al. EMNLP 2013 RNTN, `D13-1170` | Not attempted. The tensor composition $p = \tanh(b^\top V^{[1:d]}c + W[b;c])$ is the bilinear middle ground between Tree-LSTM's shared operator and MV-RNN's per-lexeme matrix; relevant to sizing $f_{\text{verb}}$ at matched parameters. |
| O-7 | Shi, Williams et al. on trivial/random trees matching parsed trees | Recalled, not fetched, no identifier resolved. If it says what it is remembered to say it is the strongest single no-separation prior; it must be fetched before the deciding test is interpreted. |
| O-8 | DNC body, `doi:10.1038/nature20101` | Paywalled; title/venue/DOI confirmed only. Used only as a negative control in §3.7. |

---

## Deciding-test implications

**The scheduling half of X₂₅ cannot separate on a tree.** SPINN (§2.3)
demonstrates that a tree-structured computation re-linearised into a token
sequence computes the same values, and TensorFlow Fold (§3.3) treats
availability-order scheduling as a batching concern with no effect on the
function computed. If the planted bed's argument structure is a tree, the eager
arm and a parse-aware token arm compute the same function up to reparametrisation
and the literature predicts no separation before the experiment runs.

**A mechanism for separation exists, and it is narrow.** DAGNN (§3.1) and AMP
(§3.6) both report gains for order-respecting asynchronous computation over
synchronous rounds, and AMP proves a strict expressiveness separation — but the
gains are claimed on *reconvergent DAGs* and on *long-distance propagation*, not
on trees. The corresponding condition for X₂₅ is that the planted bed must
contain (a) genuine argument sharing, so that one argument node feeds two or more
verb operators and the graph is not a tree, and (b) operator-graph depth
exceeding the number of layers the verbs-as-tokens arm can spend simulating it at
matched parameter count. Without both, the eager arm's advantage is a constant
factor in routing that a matched-parameter transformer can absorb.

**The operator-indexing half is where the deltas are non-zero, and it is not what
the test measures.** MV-RNN (§2.2) and the categorical line (§4.2) already index
the operator by the lexeme; Tree-LSTM (§2.1) and neural SPRL (§4.4) do not. The
contrast X₂₅ has fixed — eager-verb versus verbs-as-tokens — conflates *operator
indexing* with *scheduling*, so a positive result cannot attribute the separation
to execution semantics. If the arms separate, the ablation that decides the
attribution is a third arm: verb-indexed operators evaluated in token order.
Without it a GREEN is uninterpretable, and §1.4's Stack-NMN result — the field
replaced eager discrete execution with a soft relaxation and kept the accuracy —
is the prior that a two-arm design cannot argue against.

**Standing prediction.** On a tree-shaped planted bed at matched parameters with
$N=8$ seeds, the prior art predicts the two one-sided tests will conclude
equivalence and the kill clause will fire. On a bed with planted argument sharing
and depth beyond the token arm's layer budget, the prior art leaves a plausible
mechanism open, and the result is genuinely uncertain.
