# sun_budget.py - what a funded ladder costs. Kaplan/Chinchilla arithmetic; every price and MFU is an ASSUMPTION to replace with a quote.
peak = 989e12        # H100 SXM dense bf16 peak (vendor spec)
mfu = 0.40           # assumed model FLOPs utilization
price = 2.50         # assumed $ per H100-hour (replace with the actual quote)
ctx, arms, seeds = 4096, 5, 3
print(f"assumptions: H100 dense bf16 {peak/1e12:.0f} TF, MFU {mfu}, ${price}/GPU-h, context {ctx}, {arms} arms x {seeds} seeds per rung")
tot = 0.0
for name, L, d in (("125M", 12, 768), ("350M", 24, 1024), ("1.3B", 24, 2048), ("2.7B", 32, 2560), ("7B", 32, 4096)):
    N = 12 * L * d * d
    D = 20 * N
    C = (6 * N + 6 * L * ctx * d) * D
    h = C / (peak * mfu) / 3600
    grid = h * arms * seeds
    tot += grid * price
    print(f"{name:>5}: N={N/1e9:5.2f}B params, D={D/1e9:6.1f}B tokens, C={C:.2e} FLOP -> {h:8.1f} GPU-h per run, "
          f"${h*price:>10,.0f} per run; grid {arms}x{seeds} = {grid:9.0f} GPU-h = ${grid*price:>12,.0f}; cumulative ${tot:>12,.0f}")
