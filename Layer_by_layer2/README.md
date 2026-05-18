# Layer_by_layer2

This folder contains a second cleaned extraction of the legacy layer-by-layer
method, with stronger filtering aimed at improving plane extraction.

Compared with `layer_by_layer_method/`, this version adds:

1. isolation filtering inside each candidate layer
2. connected-component splitting of the candidate points
3. selection of the primary component before plane extraction
4. best-fit plane filtering with inlier rejection
5. centroids computed from the final filtered layer points

The overall idea is still proximity-based layer growth from STL mesh vertices,
but it is more conservative about which points are accepted into each layer.

## Example

```powershell
python -m Layer_by_layer2 `
  --mesh STLfiles/CustomEdgecaseThesis.stl `
  --json layer_by_layer2_output.json `
  --csv layer_by_layer2_output.csv
```

Direct file execution also works:

```powershell
python Layer_by_layer2\__main__.py `
  --mesh STLfiles/CustomEdgecaseThesis.stl `
  --json layer_by_layer2_output.json `
  --csv layer_by_layer2_output.csv
```

If `--mesh` is omitted, the CLI will try to use
`STLfiles/CustomEdgecaseThesis.stl` from the repository automatically.

## Notes

- This package is added alongside the original extraction and does not modify it.
- Rejected candidate points are tracked separately so you can inspect what was
  filtered away.
- The plane filter uses an iterative best-fit plane refinement rather than a
  full RANSAC implementation.
