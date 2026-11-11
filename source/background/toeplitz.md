# Why Toeplitz and How Does It Work?

The AhA operator for MRI is [Toeplitz](https://en.wikipedia.org/wiki/Toeplitz_matrix) under
arbitrary non-Cartesian trajectories ([Baron et al., MRM
2018](https://onlinelibrary.wiley.com/doi/10.1002/mrm.26928)). A Toeplitz matrix can be padded to 2X
on its original dimensions to form a [Circulant
matrix](https://en.wikipedia.org/wiki/Circulant_matrix), and be [calculated via
FFTs](https://math.mit.edu/icg/resources/teaching/18.085-spring2015/toeplitz.pdf), requiring only
$O(N \log N)$ operations instead of $O(N^2)$ for general matrix-vector multiplications. In other
words, the Toeplitz operator can be diagonalized by FFTs.

$$Tx \implies F^H\Lambda F\hat{x}, $$

>* $T \in \mathbb{C}^{N \times N}$ is a symmetric Toeplitz operator
>* $x \in \mathbb{C}^{N}$ is a vector
>* $F \in \mathbb{C}^{2N \times 2N}$ is the 2D DFT operator
>* $\Lambda \in \mathbb{C}^{2N \times 2N}$ is a diagonal matrix
>* $\hat{x} \in \mathbb{C}^{2N}$ is the zero-padded vector of $x$

The diagonal elements of $\Lambda$ can be derived based on the non-Cartesian coordinates but we
usually use NUFFT packages to estimate them (see [Baron et al., MRM
2018](https://onlinelibrary.wiley.com/doi/10.1002/mrm.26928)).

Under MR-Multitasking, the AhA operator is block-Toeplitz, which can be diagonalized similarly, with
the "diagonal" aspect being block-diagonal, a.k.a. the gridded $\Phi\Phi^H$ operator — this is
`phi2_toep`.

The challenge for 3D stack-of-stars AhA is that there is usually not enough GPU memory to hold the
gridded $\Phi\Phi^H$ operator, which has a size of $2N_x \times 2N_y \times N_z \times L \times L$.
For a $320 \times 320 \times 56$ matrix with $L = 16$, that is about 26 GB in single precision and a
24 GB GPU cannot hold it.

Luckily for us, each individual $L \times L$ matrix of the gridded $\Phi\Phi^H$ operator is
[Hermitian](https://en.wikipedia.org/wiki/Hermitian_matrix), so the lower triangle is the complex
conjugate of the upper one. Storing a single triangle — which is what `config.triangulation`
selects — halves that to about 13 GB and makes the 3D operator feasible. When even that is too
much, `config.is_masked` keeps only the upsampled k-space samples that actually carry signal.

## A note on Hermitian symmetry and the old "deconvolve" hack

The previous implementation observed that the gridded $\Phi\Phi^H$ came out slightly non-Hermitian,
and blamed the deconvolution (transfer-function division) step of the type-1 NUFFT; it worked around
this with a [FINUFFT fork](https://github.com/mikecjz/finufft/tree/toeplitz) exposing the spreading
and deconvolution stages separately.

That workaround is gone. Hermitian symmetry is now enforced *structurally*: only one triangle is
ever computed and stored, and the mirrored half is produced by conjugation inside the
block-diagonal kernel. The gridded operator is therefore Hermitian by construction, to machine
precision, whatever the NUFFT does internally — which also means unmodified upstream FINUFFT works.
[`examples/example_aha.cu`](https://github.com/mikecjz/MR_AHA/blob/main/examples/example_aha.cu) checks
$\langle y, A^HAx\rangle = \langle A^HAy, x\rangle$ end to end.
