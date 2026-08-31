"""[V-eq] probe 2 -- what an FCI-class algorithm's input can and cannot fix about a latent.

Sources transcribed:
  Colombo, Maathuis, Kalisch, Richardson 2012, Ann. Statist. 40(1):294-321,
  arXiv:1104.5617.
    Section 3 (oracle problem statement): the algorithm's input is "oracle
      information about all conditional independence relationships between pairs
      of variables Xi and Xj in X given sets Y u S where Y subset of X\\{Xi,Xj}".
    Definition 3.1 (FCI-PAG), condition (i): absence of an edge implies
      Xi _||_ Xj | (Y u S) for some Y; condition (ii): presence of an edge implies
      Xi _|/|_ Xj | (Y u S) for ALL Y.
    Theorem 3.1: under faithfulness, the output is an FCI-PAG of G.
    Completeness (Zhang 2008, AIJ 172:1873-1896): the output is maximally
      informative, "for every circle mark there exists at least one MAG in the
      Markov equivalence class represented by the PAG where the mark is oriented
      as a tail, and at least one where it is oriented as an arrowhead".

The theorem's input is the CI oracle. So anything two models share in their CI
oracle, FCI must return identically -- and if their observed covariance matrices
are equal, so are their CI oracles. That is what is computed here, exactly, in
the linear-Gaussian case where m-separation <=> vanishing partial correlation.

  (A) counting/locating latents is NOT fixed: one latent parent of {X,Y,Z}
      versus three pairwise latents give the SAME observed covariance to ~1e-16,
      hence the same CI oracle, hence the same PAG. Latent count 1 vs 3.
  (B) the EXISTENCE of a latent common cause on a named pair IS fixed: adding
      L -> {X1,X3} to a chain destroys X1 _||_ X3 | X2 by O(1).
  CONTROL: the marginal correlation in place of the partial correlation -- an
      O(1) mis-transcription of the CI test -- misreads (B) entirely.
"""
import itertools
import numpy as np

np.set_printoptions(precision=6, suppress=True)


def cov_from_lin_gauss(B, err_var):
    """Sigma = (I-B)^{-1} diag(err_var) (I-B)^{-T} for x = Bx + e, B strictly lower
    triangular in a causal order."""
    A = np.linalg.inv(np.eye(len(B)) - B)
    return A @ np.diag(err_var) @ A.T


def partial_corr(S, i, j, K):
    """Partial correlation of i,j given the index set K, from the covariance S."""
    idx = [i, j] + list(K)
    M = np.linalg.inv(S[np.ix_(idx, idx)])
    return -M[0, 1] / np.sqrt(M[0, 0] * M[1, 1])


def ci_oracle(S, obs, tol=1e-9):
    """Every (i,j | K) over the observed indices with |partial corr| < tol."""
    out = []
    for i, j in itertools.combinations(obs, 2):
        rest = [k for k in obs if k not in (i, j)]
        for r in range(len(rest) + 1):
            for K in itertools.combinations(rest, r):
                if abs(partial_corr(S, i, j, K)) < tol:
                    out.append((i, j, K))
    return set(out)


# ---------------------------------------------------------------- (A) one latent vs three
# order: X=0, Y=1, Z=2, then latents.
# Model A: L -> X, Y, Z   (one latent, index 3)
a = np.array([0.9, 0.7, -0.6])
BA = np.zeros((4, 4))
BA[0, 3], BA[1, 3], BA[2, 3] = a
SA_full = cov_from_lin_gauss(BA, np.array([2.0, 2.0, 2.0, 1.0]))
SA = SA_full[:3, :3]

# Model B: three pairwise latents L_xy (3), L_yz (4), L_xz (5); no single common cause.
# Solve the three loadings so that B reproduces A's off-diagonals exactly.
c_xy, c_yz, c_xz = SA[0, 1], SA[1, 2], SA[0, 2]
lam_xy = np.sqrt(abs(c_xy)); s_xy = np.sign(c_xy)
lam_yz = np.sqrt(abs(c_yz)); s_yz = np.sign(c_yz)
lam_xz = np.sqrt(abs(c_xz)); s_xz = np.sign(c_xz)
BB = np.zeros((6, 6))
BB[0, 3], BB[1, 3] = s_xy * lam_xy, lam_xy          # L_xy -> X, Y
BB[1, 4], BB[2, 4] = s_yz * lam_yz, lam_yz          # L_yz -> Y, Z
BB[0, 5], BB[2, 5] = s_xz * lam_xz, lam_xz          # L_xz -> X, Z
# error variances chosen so the observed diagonals match A exactly
ev = np.ones(6)
ev[0] = SA[0, 0] - BB[0, 3] ** 2 - BB[0, 5] ** 2
ev[1] = SA[1, 1] - BB[1, 3] ** 2 - BB[1, 4] ** 2
ev[2] = SA[2, 2] - BB[2, 4] ** 2 - BB[2, 5] ** 2
SB = cov_from_lin_gauss(BB, ev)[:3, :3]

print("(A) latent count in the two generating models     : 1  vs  3")
print("(A) max |Sigma_obs(1 latent) - Sigma_obs(3 latents)|:", np.abs(SA - SB).max())
print("(A) error variances of model B positive            :", bool(np.all(ev[:3] > 0)))
print("(A) CI oracles equal                               :",
      ci_oracle(SA, [0, 1, 2]) == ci_oracle(SB, [0, 1, 2]),
      "| both empty:", len(ci_oracle(SA, [0, 1, 2])) == 0)

# ------------------------------------------- (B) existence of a latent on a named pair
# Chain X1 -> X2 -> X3 (no latent) vs the same chain plus L -> {X1, X3}.
BP = np.zeros((3, 3)); BP[1, 0] = 0.8; BP[2, 1] = 0.6
SP = cov_from_lin_gauss(BP, np.ones(3))
BQ = np.zeros((4, 4)); BQ[1, 0] = 0.8; BQ[2, 1] = 0.6; BQ[0, 3] = 0.9; BQ[2, 3] = 0.7
SQ = cov_from_lin_gauss(BQ, np.ones(4))[:3, :3]
p_chain = partial_corr(SP, 0, 2, [1])
p_latent = partial_corr(SQ, 0, 2, [1])
print("(B) partial corr X1,X3 | X2  chain, no latent      :", p_chain)
print("(B) partial corr X1,X3 | X2  chain + latent on pair:", p_latent)
print("(B) O(1) separation                                :", abs(p_latent - p_chain))

# CONTROL: marginal correlation instead of the partial correlation.
m_chain = SP[0, 2] / np.sqrt(SP[0, 0] * SP[2, 2])
m_latent = SQ[0, 2] / np.sqrt(SQ[0, 0] * SQ[2, 2])
print("control (marginal corr, conditioning dropped)      :", m_chain, m_latent,
      "-> declares dependence in BOTH, difference", abs(m_latent - m_chain))
