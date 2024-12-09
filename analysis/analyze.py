import re
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# If the results are in a file, you can read from it:
with open("results.txt", "r") as f:
    raw_data = f.read()

lines = raw_data.split("\n")

current_set = None
data = []

N, E, S = None, None, None
implementation = None
preamble, main_algo, write_out = None, None, None

for line in lines:
    line = line.strip()
    # Detect which set we are in
    if "BEGIN STANDARD SET" in line:
        current_set = "STANDARD"
    elif "BEGIN DENSE SET" in line:
        current_set = "DENSE"
    elif "BEGIN HIGHLY SPARSE SET" in line:
        current_set = "HIGHLY SPARSE"
    elif "END" in line and "SET" in line:
        current_set = None

    # Parse parameters
    if line.startswith("Parameters:"):
        # Example: Parameters: N=10000, E=50000, Sparsity=5.00000000e-04
        parts = line.split(',')
        N_part = parts[0].split('=')[1].strip()
        E_part = parts[1].split('=')[1].strip()
        S_part = parts[2].split('=')[1].strip()
        N = int(N_part)
        E = int(E_part)
        S = float(S_part)

    # Check for implementation lines
    if "Running pagerank-cpp (Naive CPU)..." in line:
        implementation = "Naive CPU"
        preamble, main_algo, write_out = None, None, None
    elif "Running pagerank-opt (Optimized CPU)..." in line:
        implementation = "Optimized CPU"
        preamble, main_algo, write_out = None, None, None
    elif "Running pagerank-cuda (CUDA)..." in line:
        implementation = "CUDA"
        preamble, main_algo, write_out = None, None, None

    # Timing lines
    if line.startswith("Preamble:"):
        # e.g. "Preamble: 31ms" or "Preamble: 31 ms"
        val = re.search(r'Preamble:\s+(\d+)\s?ms', line)
        if val:
            preamble = float(val.group(1))
    elif line.startswith("Main Algorithm:"):
        val = re.search(r'Main Algorithm:\s+(\d+)\s?ms', line)
        if val:
            main_algo = float(val.group(1))
    elif line.startswith("Write Output:"):
        val = re.search(r'Write Output:\s+(\d+)\s?ms', line)
        if val:
            write_out = float(val.group(1))
    elif line.startswith("Total:"):
        # Once we hit Total, we have all components for this implementation
        val = re.search(r'Total:\s+(\d+)\s?ms', line)
        if val and implementation and current_set and N is not None and E is not None and S is not None:
            total = float(val.group(1))
            # Store the data
            data.append({
                'Set': current_set,
                'N': N,
                'E': E,
                'Sparsity': S,
                'Implementation': implementation,
                'Preamble': preamble if preamble is not None else 0.0,
                'Main': main_algo if main_algo is not None else 0.0,
                'Write': write_out if write_out is not None else 0.0,
                'Total': total
            })
            # Reset implementation info after storing
            implementation = None

df = pd.DataFrame(data)

if df.empty:
    print("No data was parsed. Check your input format in results.txt.")
    exit(1)

sets = df['Set'].unique()

# Define colors for each component
preamble_color = 'tab:blue'
main_color = 'tab:orange'
write_color = 'tab:green'

for s in sets:
    subset = df[df['Set'] == s].copy()
    # Sort by N, E for consistent ordering
    subset.sort_values(by=['N', 'E'], inplace=True)
    subset['Scenario'] = subset.apply(lambda row: f"N={row['N']}\nE={row['E']}\nS={row['Sparsity']:.1e}", axis=1)

    # Pivot tables for each component
    pivot_preamble = subset.pivot(index='Scenario', columns='Implementation', values='Preamble')
    pivot_main = subset.pivot(index='Scenario', columns='Implementation', values='Main')
    pivot_write = subset.pivot(index='Scenario', columns='Implementation', values='Write')

    impl_order = ['Naive CPU', 'Optimized CPU', 'CUDA']

    # Ensure each pivot has all columns
    for pivot_df in [pivot_preamble, pivot_main, pivot_write]:
        for impl in impl_order:
            if impl not in pivot_df.columns:
                pivot_df[impl] = 0.0
        pivot_df = pivot_df[impl_order]

    scenarios_list = pivot_preamble.index.tolist()
    x = range(len(scenarios_list))

    fig, ax = plt.subplots(figsize=(12, 6))

    width = 0.6 / 3.0  # total group width approx 0.6, divided by 3 bars
    offsets = [-width, 0, width]

    for i, impl in enumerate(impl_order):
        pre = pivot_preamble[impl].values
        main = pivot_main[impl].values
        wri = pivot_write[impl].values

        # stack bars with consistent colors
        bar_bottom = ax.bar([xx + offsets[i] for xx in x], pre, width,
                            label='Preamble' if i == 0 else None,
                            color=preamble_color)
        bar_middle = ax.bar([xx + offsets[i] for xx in x], main, width,
                            bottom=pre, label='Main' if i == 0 else None,
                            color=main_color)
        bar_top = ax.bar([xx + offsets[i] for xx in x], wri, width,
                         bottom=[p+m for p,m in zip(pre,main)],
                         label='Write Output' if i == 0 else None,
                         color=write_color)

    ax.set_xticks(x)
    ax.set_xticklabels(scenarios_list, rotation=0)
    ax.set_ylabel("Time (ms)")
    ax.set_title(f"Performance Components - {s} SET _NVDA_T4")

    # Create legends
    # Component legend from labels defined in first implementation's bars
    component_legend = ax.legend(loc='upper left', title="Components")
    ax.add_artist(component_legend)

    # Implementation legend
    # from matplotlib.patches import Patch
    # impl_patches = [Patch(facecolor='gray', alpha=0.5, label=impl) for impl in impl_order]
    # ax.legend(handles=impl_patches, title="Implementation", loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
    # Add a comment at the bottom
    plt.figtext(0.5, 0.00, "left bar is cpu-naive, center is cpu-optimized, and the right is cuda", 
            ha="center", fontsize=13, color="black")
    plt.tight_layout()
    plt.savefig(f"pagerank_components_{s.lower().replace(' ', '_')}_set.png", bbox_inches='tight')
    plt.close()

print("Charts generated successfully.")
