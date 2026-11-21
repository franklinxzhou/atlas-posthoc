EXAMPLE_CONFIG = """\
[project]
# Relative project.root is resolved from this TOML file. All other paths use it.
root = "."

[chains]
files = [
  "inputs/chains/run1.jsonl.gz",
  "inputs/chains/run2.jsonl.gz",
  "inputs/chains/run3.jsonl.gz",
  "inputs/chains/run4.jsonl.gz",
]
# Omit ids to infer chain labels from filenames.
ids = ["run1", "run2", "run3", "run4"]

[analysis]
# Omit n_districts to infer it from the AtlasIO headers.
n_districts = 15
output_frequency = 10000
thin_step = 1
# Set false to run vote-share diagnostics on chains without compactness scores.
compute_compactness = true

[inputs]
graph_json = "inputs/graph.json"

[columns]
# Omit key_cols to infer the graph node columns used to match assignment keys.
key_cols = ["NAME"]
dem_col = "G20PREDEM"
rep_col = "G20PREREP"

[outputs]
root = "outputs"

[figures]
# Set true for title-free manuscript figures saved as vector PDF.
paper_mode = false

[workflow.convergence]
vote_bin_width = 0.002
compactness_bin_width = 0.5
# vote_y_limits = [0.20, 0.82]
# compactness_y_limits = [15, 55]
# figure_prefix = "oh_"
# report_title = "Congressional chain convergence"
"""
