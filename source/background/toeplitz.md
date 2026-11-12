# Why Toeplitz and How Does It Work?

## The idea

The $A^H A$ operator of MRI is [Toeplitz](https://en.wikipedia.org/wiki/Toeplitz_matrix) for
arbitrary non-Cartesian trajectories ([Baron et al., MRM
2018](https://onlinelibrary.wiley.com/doi/10.1002/mrm.26928)). A Toeplitz matrix can be embedded
in a [circulant matrix](https://en.wikipedia.org/wiki/Circulant_matrix) of twice the size in each
dimension, and a circulant matrix is [diagonalised by the
FFT](https://math.mit.edu/icg/resources/teaching/18.085-spring2015/toeplitz.pdf). Applying the
operator then takes $O(N \log N)$ operations instead of the $O(N^2)$ of a general matrix-vector
product:

$$Tx \implies F^H\Lambda F\hat{x}, $$

>* $T \in \mathbb{C}^{N \times N}$ is a symmetric Toeplitz operator
>* $x \in \mathbb{C}^{N}$ is a vector
>* $F \in \mathbb{C}^{2N \times 2N}$ is the 2D DFT operator
>* $\Lambda \in \mathbb{C}^{2N \times 2N}$ is a diagonal matrix
>* $\hat{x} \in \mathbb{C}^{2N}$ is the zero-padded vector of $x$

The diagonal of $\Lambda$ follows from the non-Cartesian coordinates. In practice it is computed
with a NUFFT, once per trajectory ([Baron et al., MRM
2018](https://onlinelibrary.wiley.com/doi/10.1002/mrm.26928)).

## With a low-rank model

With a low-rank (subspace) model the $A^H A$ operator is block-Toeplitz. It is diagonalised in the same way,
except that $\Lambda$ becomes block-diagonal: one $L \times L$ block per upsampled k-space
location. This block-diagonal multiplier is the gridded $\Phi\Phi^H$ operator, which the library
calls `phi2_toep`.

## Fitting it in GPU memory

The gridded $\Phi\Phi^H$ operator has $2N_x \times 2N_y \times N_z \times L \times L$ entries for
a stack-of-stars acquisition. For a $320 \times 320 \times 56$ matrix with $L = 12$ that is about
26 GB in single precision, more than a 24 GB GPU holds. Two properties make it fit:

* **One triangle is enough.** Each $L \times L$ block is
  [Hermitian](https://en.wikipedia.org/wiki/Hermitian_matrix), so its lower triangle is the
  complex conjugate of its upper one. The library stores a single triangle, the one selected by
  `config.triangulation`: $(L^2+L)/2$ entries per block instead of $L^2$. In the example that is
  about 14 GB.
* **A mask removes the empty locations.** `config.is_masked` keeps only the upsampled k-space
  locations that carry signal. See [Masking](../guide/masking.md).

## Hermitian by construction

Only one triangle of each block is ever computed and stored, and the block-diagonal kernel
produces the mirrored half by conjugation. The gridded operator is therefore Hermitian to machine
precision, whatever the NUFFT does internally, and the library works with unmodified upstream
FINUFFT.
[`examples/example_aha.cu`](https://github.com/mikecjz/MR_AHA/blob/main/examples/example_aha.cu)
checks $\langle y, A^HAx\rangle = \langle A^HAy, x\rangle$ end to end.
