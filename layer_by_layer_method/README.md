# Layer-by-Layer Method

This folder contains a clean extraction of the legacy layer-by-layer point
grouping method from `outdated3DP/STL_centreline/`.

It is intentionally kept separate from the current `SlicerProgram` pipeline.
The method here represents the earlier proximity-based layering idea:

1. Load mesh vertices from an STL file.
2. Build the first layer from the lowest `z` region.
3. Remove isolated points using a local-neighborhood rule.
4. Repeatedly assign unvisited points to the next layer when they are within a
   proximity radius of the current layer.
5. Optionally export the extracted layers to JSON and CSV.

This implementation is closest to the adjacency/proximity version developed in
`outdated3DP/STL_centreline/Levelslice.py`, but is packaged as reusable Python
code with a small CLI.

## Files

- `extractor.py`: core implementation
- `cli.py`: command-line interface
- `__main__.py`: allows `python -m layer_by_layer_method`

## Example

```powershell
python -m layer_by_layer_method `
  --mesh STLfiles/CustomEdgecaseThesis.stl `
  --json layer_by_layer_output.json `
  --csv layer_by_layer_output.csv
```

Direct file execution also works:

```powershell
python layer_by_layer_method\__main__.py `
  --mesh STLfiles/CustomEdgecaseThesis.stl `
  --json layer_by_layer_output.json `
  --csv layer_by_layer_output.csv
```

If `--mesh` is omitted, the CLI will try to use
`STLfiles/CustomEdgecaseThesis.stl` from the repository automatically.

## Notes

- The method uses mesh vertices, not volumetric samples.
- It is a legacy prototype and does not replace the branch-aware centreline
  slicer in `SlicerProgram/`.
- Existing repository files are not modified by this extraction.
