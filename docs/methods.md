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

## Restrictions

The plate model is limited to homogeneous isotropic, rectangular and
small-displacement cases. The [validation record](validation.md) lists the
tested material, thicknesses, boundary conditions and loads. Results require
independent review for an engineering application.

Inputs are stored as finite Python floats and integer mesh counts, including
when NumPy scalar inputs are supplied. The model-size caps limit memory use;
they do not guarantee that every parameter combination is numerically solvable.
Non-finite calculations, a singular constrained system or a free-DOF relative
residual above `1e-8` raise `ArithmeticError`. The residual divides by the actual
free-load norm, with scaling to avoid overflow/underflow; zero load requires
zero free imbalance. Invalid input raises `ValueError`.
