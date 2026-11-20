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
work. Four bilinear corner functions interpolate all three fields. Both energy
terms use a 2-by-2 Gauss rule; no reduced-integration stabilization is added.

J. Schöberl's [Reissner–Mindlin notes](https://jschoeberl.github.io/iFEM/plates_shells/2_ReissnerMindlin.html)
provide the underlying energy and discuss shear locking. The mixed discretization
in those notes is not the Q4 displacement element implemented here. The notes'
code is not used.

Clamping sets all boundary DOFs to zero. Hard simple support sets `w=0` and
the tangential component of beta to zero. For the independent uniform-pressure
reference, sine load modes have coefficient `16q/(pi²mn)` for odd `m,n`.
Minimizing the continuous energy mode by mode gives compliance
`1/(D lambda²)+1/(kGt lambda)`, with `lambda=(m pi/a)²+(n pi/b)²`.
The tests sum these modes directly without finite element assembly or fixtures.

## Restrictions

Only homogeneous isotropic, rectangular and small-displacement cases are
included. Full Q4 integration can severely underestimate thin-plate deflection;
a regression test exposes this behavior. Refinement of a thick plate is tested
against the continuous mode solution. That does not validate arbitrary geometry,
material, thickness, boundary conditions or loads. Results require independent
review for an engineering application.

Inputs are stored as finite Python floats and integer mesh counts, including
when NumPy scalar inputs are supplied. The model-size caps limit memory use;
they do not guarantee that every parameter combination is numerically solvable.
Non-finite calculations, a singular constrained system or a free-DOF relative
residual above `1e-8` raise `ArithmeticError`. The residual divides by the actual
free-load norm, with scaling to avoid overflow/underflow; zero load requires
zero free imbalance. Invalid input raises `ValueError`.
