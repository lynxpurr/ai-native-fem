# View a saved result

Generate an offline report from a static result directory and open the new
HTML file in a browser. Input paths with spaces need shell quotes.

```sh
anf solve examples/structures/portal.json --output portal-result
anf view portal-result --output portal-view.html
```

`anf view` accepts the same schema 1.0 beam and rectangular plate and schema
1.1 planar structure artifacts as `anf inspect`. The output parent directory
must exist. Each export writes a new HTML file containing the saved result
samples and model identity. The page runs locally from disk with no network
requests. Treat the HTML as a copy of the displayed analysis data when sharing it.

- **Mesh.** Gray shows the original geometry and blue shows the displaced
  geometry. Set the deformation scale to 0 for the original geometry or 1
  for the saved displacement at physical scale. The geometry coordinates and
  units remain visible. Plate normal displacement uses a fixed oblique view.
- **Scalar field.** Choose an array to display its sample values, units,
  association, and coordinate system. Frame rotations appear at frame nodes.
  Member end actions use local member axes and show separate markers inset
  from the directed i and j ends along their terminal mesh segments. Read
  exact values in the field sample table.
- **Loads and supports.** Squares mark the declared locations. Open the
  respective tables for component, sign, units, and value. The plate pressure
  marker denotes the surface load. Uniform line loads are marked at midspan.
- **Verification.** The header preserves the saved solve and verification
  status and diagnostics. Numerical failures display the original mesh and
  declarations with the field and deformation controls disabled.

`--json` returns the inspection summary and the output file path. Invalid,
unsupported, damaged, or oversized artifacts fail through the standard
reader before the report is written.

## View modal results

Use the same command with a schema 1.4 artifact created by
[save_modal_result](modal.md#artifacts). Select a mode to see its frequency
in Hz, relative eigenpair residual, and saved mass-normalized coefficients
at free DOFs. Fixed translations stay at their original coordinates.
Rotations appear as scalar coefficients.

The amplitude is relative to the mesh. At amplitude 1, the largest nodal
translation is 10% of the largest world-coordinate span at the animation's
peak. Playback uses a fixed illustrative period. The physical frequency
remains visible. The sign of the mode is arbitrary and repeated frequencies
describe a subspace. Rigid-body modes are labeled individually with their
saved frequencies in the details. An artifact with an empty mode selection
opens with the original mesh and the playback controls disabled.

`view` reads an artifact without running the solver. Opening a saved modal
result works with the base installation. Batch results, frequency response,
time histories and buckling require separate artifact views.
