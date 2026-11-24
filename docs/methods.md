# Mathematical basis and limits

## Beam

For one bending plane, stationarity of
`(1/2) ∫ EI (w'')² dx - ∫ q w dx - P w(L)` defines the model.
The nodal variables are displacement and slope. Cubic Hermite interpolation
enforces their endpoint values. Integrating its curvature products with a
two-point Gauss rule gives the element matrix; the same rule integrates a
uniform line load. The root displacement and slope are zero.

The mathematical reference is D. M. Kochmann's ETH Zurich
[Introduction to Finite Element Analysis, section 12.4](https://ethz.ch/content/dam/ethz/special-interest/mavt/mechanical-systems/mm-dam/documents/Notes/IntroToFEA_red.pdf).
The equations inform this implementation; publication code, figures and teaching
text are not reproduced. Tests use `w(L)=PL³/(3EI)+qL⁴/(8EI)` and the corresponding
root force/moment balance, derived by integrating the differential equation.

## Plate

Let the nodal fields be `w, beta_x, beta_y`, with engineering curvature
`[beta_x,x, beta_y,y, beta_x,y+beta_y,x]` and shear `grad(w)-beta`.
The bending matrix is `Et³/[12(1-nu²)] * [[1,nu,0],[nu,1,0],[0,0,(1-nu)/2]]`.
The shear coefficient is `kGt`, with `G=E/[2(1+nu)]` and default `k=5/6`.
The energy is one half of bending plus shear quadratic forms, minus pressure
work. Four bilinear corner functions interpolate all three fields. Each node
has DOFs `[w, beta_x, beta_y]`. Local nodes run counterclockwise from the lower
left, so the natural-coordinate corners are `(-1,-1), (+1,-1), (+1,+1), (-1,+1)`.
Positive pressure acts in the positive `w` direction. `beta_x` and `beta_y`
are the slope components subtracted from the corresponding derivatives of `w`.
Bending, shear and pressure loads use 2 × 2 Gauss integration.

The default `formulation="q4_full"` evaluates `grad(w)-beta` directly at each
Gauss point. It can severely underestimate thin-plate deflection because of
shear locking. Select `formulation="mitc4_rectangle"` to interpolate the
transverse shear components from the four edge midpoints.

```text
gamma_x_assumed = ((1-eta)/2) gamma_x(0,-1) + ((1+eta)/2) gamma_x(0,+1)
gamma_y_assumed = ((1-xi) /2) gamma_y(-1,0) + ((1+xi) /2) gamma_y(+1,0)
```

Each edge value is the displacement difference divided by edge length, minus
the mean slope component along that edge. The axis-aligned rectangle has a
constant diagonal Jacobian, so the covariant scale for each component cancels
when it is mapped back to physical shear. In the source, ANF's corner order is
3, 4, 1, 2. ANF's `beta_x` is the negative of the paper's `beta_x`, and
`beta_y` has the same sign. The curvature vector changes sign as a whole,
which preserves its energy. See Figure 2 and equations (1)–(7) in
[Bathe and Dvorkin (1985)](https://doi.org/10.1002/nme.1620210213).

Clamping sets all boundary DOFs to zero. Hard simple support sets `w=0` and
the tangential component of beta to zero, with the normal bending moment a
natural boundary condition. `mitc4_rectangle` requires `support="hard_simply_supported"`.
The API builds a uniform axis-aligned mesh from `lx`, `ly`, `nx` and `ny` and
accepts a scalar pressure. For the independent uniform-pressure
reference, sine load modes have coefficient `16q/(pi²mn)` for odd `m,n`.
Minimizing the continuous energy mode by mode gives compliance
`1/(D lambda²)+1/(kGt lambda)`, with `lambda=(m pi/a)²+(n pi/b)²`.
The tests sum these modes directly from the continuous energy described in
[Schöberl's Reissner–Mindlin notes](https://jschoeberl.github.io/iFEM/plates_shells/2_ReissnerMindlin.html).

## Planar truss and frame

A `truss2d` member is a two-node bar with axial strain energy
`(1/2) ∫ EA (u')² dx`. A `frame2d` member adds Euler–Bernoulli bending through
cubic Hermite interpolation of the transverse displacement, so the end
rotations are nodal degrees of freedom. Each node carries `ux`, `uy` and `rz`.
Element matrices and consistent load vectors are formed in local axes and
rotated to global axes before dense assembly. The documented uniform and
linearly varying member loads contribute consistent nodal loads and fixed-end
actions. Prescribed displacements enter the free-system right-hand side, and
reactions come from the assembled force balance.

Schema 1.3 end releases condense the released end rotation out of the element
and retain the corresponding fixed-end moment in the load vector. A rotational
spring adds its stiffness between the beam end rotation and the external node
rotation. Member end actions and station values for axial force, shear and
moment are recovered from the local displacement and the member load. The
formulation is linear, small displacement, planar and rigid-jointed except for
the declared releases and springs; it does not include shear deformation or
3D torsion. See the [planar structure guide](frame2d.md) and
[load case guide](load-cases.md).

## Sparse backend

The optional SciPy backend assembles element triplets in model order, sums
duplicate entries, removes explicit zeros and stores the stiffness in CSR.
The free system is solved in CSC form with SuperLU. Nonzero prescribed
displacements enter as `f_free - K_free,fixed @ u_fixed`, and reactions come
from the original global force balance. Planar structures use diagonal
equilibration and a positive-pivot test on a separate shifted stiffness; the
solve itself uses the unshifted matrix. The free-DOF relative residual limit
remains `1e-8`, and the model fingerprint, result format and verification
checks do not depend on the backend. The dense solver remains the default.
See the [sparse guide](sparse.md).

## Modal analysis

Schema 1.4 forms the generalized eigenproblem `K phi = lambda M phi` on the free
DOFs. The truss uses the 4×4 consistent mass matrix with axial and transverse
translational inertia; it has no transverse stiffness. The frame uses the
Hermite 6×6 consistent mass matrix, including the rotational inertia implied by
the cubic transverse field. Both are rotated to global `ux/uy/rz` and assembled
densely.

Fixed supports are removed from both matrices. A free planar model has three
rigid modes; a supported model must have none. The dense
`scipy.linalg.eigh(K_ff, M_ff)` returns ascending eigenvalues and
mass-normalized modes. Each mode is checked with a relative eigenpair residual:
elastic modes use
`||K phi - lambda M phi|| / (||K phi|| + |lambda| ||M phi||)`, while rigid modes
use `||K phi|| / (||K||_F ||phi||)` because the elastic denominator vanishes.
The solver also checks mass orthogonality and rejects negative eigenvalues,
rigid-mode mismatches and zero-energy modes in supported models. Requesting more
modes than the free DOF count returns the available modes and sets `truncated`;
it never pads zero frequencies.

Modal analysis is available through Python only. It requires the `sparse` extra
(SciPy); a base installation reports `backend_unavailable`. The 600-DOF planar
cap keeps the dense eigensolver practical, and no sparse modal path or plate
mass matrix is included. See the [modal guide](modal.md).

## Restrictions

The plate model is limited to homogeneous isotropic, rectangular and
small-displacement cases. Planar models are limited to the documented node,
member, support and load forms, with releases and springs only where declared.
Schema 1.2/1.3 batch runs, parameter studies and native mesh import are Python
interfaces; the CLI and MCP accept single-solve schema 1.0/1.1 declarations.
Parameter studies cover a schema 1.0 cantilever with a tip force, and native
meshes cover planar members and uniform rectangular plates. Modal analysis is
limited to the planar constant-section truss/frame forms with consistent mass,
small displacements and zero supports; it does not include plate mass, shear
deformation, rotary inertia, damping or nonlinearity. The
[validation record](validation.md) lists the tested material, thicknesses,
boundary conditions and loads. Results require independent review for an
engineering application.

Inputs are stored as finite Python floats and integer mesh counts, including
when NumPy scalar inputs are supplied. The model-size caps limit memory use;
they do not guarantee that every parameter combination is numerically solvable.
Non-finite calculations, a singular constrained system or a free-DOF relative
residual above `1e-8` raise `ArithmeticError`. The residual divides by the actual
free-load norm, with scaling to avoid overflow/underflow; zero load requires
zero free imbalance. Invalid input raises `ValueError`.
