# Linear Algebra Quick Reference

## Matrix Creation

### Basic Matrices

```octave
A = [1, 2; 3, 4]             % 2x2 matrix
zeros(3, 4)                  % 3x4 zero matrix
ones(2, 3)                   % 2x3 matrix of ones
eye(3)                       % 3x3 identity matrix
diag([1, 2, 3])             % Diagonal matrix
diag(A)                      % Extract diagonal elements
```

### Special Matrices

```octave
rand(3, 3)                   % Random matrix [0,1]
randn(3, 3)                  % Normal random matrix
magic(3)                     % Magic square
hilb(3)                      % Hilbert matrix
pascal(3)                    % Pascal matrix
```

## Matrix Operations

### Basic Operations

```octave
A + B                        % Matrix addition
A - B                        % Matrix subtraction
A * B                        % Matrix multiplication
A .* B                       % Element-wise multiplication
A / B                        % Right matrix division (A * inv(B))
A \ B                        % Left matrix division (inv(A) * B)
A ./ B                       % Element-wise division
A ^ n                        % Matrix power
A .^ n                       % Element-wise power
```

### Matrix Properties

```octave
A'                           % Transpose
A.'                          % Complex conjugate transpose
size(A)                      % Matrix dimensions
length(A)                    % Length of vector
numel(A)                     % Total number of elements
rank(A)                      % Matrix rank
det(A)                       % Determinant
trace(A)                     % Trace (sum of diagonal)
norm(A)                      % Matrix norm
cond(A)                      % Condition number
```

## Matrix Decompositions

### LU Decomposition

```octave
[L, U] = lu(A)               % LU decomposition
[L, U, P] = lu(A)            % LU with pivoting
```

### QR Decomposition

```octave
[Q, R] = qr(A)               % QR decomposition
[Q, R, P] = qr(A)            % QR with pivoting
```

### Singular Value Decomposition (SVD)

```octave
[U, S, V] = svd(A)           % Full SVD
[U, S, V] = svd(A, 'econ')   % Economy SVD
s = svd(A)                   % Singular values only
```

### Cholesky Decomposition

```octave
R = chol(A)                  % Cholesky decomposition (A = R'*R)
[R, p] = chol(A)             % With positive definiteness check
```

### Eigenvalue Decomposition

```octave
[V, D] = eig(A)              % Eigenvalues and eigenvectors
lambda = eig(A)              % Eigenvalues only
[V, D] = eig(A, B)           % Generalized eigenvalue problem
```

## Solving Linear Systems

### Direct Methods

```octave
x = A \ b                    % Solve Ax = b
x = inv(A) * b               % Using inverse (less efficient)
x = linsolve(A, b)           % General linear solver
```

### Iterative Methods

```octave
x = pcg(A, b)                % Conjugate gradient
x = gmres(A, b)              % GMRES
x = bicg(A, b)               % BiCG
```

### Least Squares

```octave
x = A \ b                    % Least squares if A is overdetermined
x = pinv(A) * b              % Using pseudoinverse
```

## Matrix Analysis

### Norms

```octave
norm(A, 1)                   % 1-norm (max column sum)
norm(A, 2)                   % 2-norm (largest singular value)
norm(A, inf)                 % Infinity norm (max row sum)
norm(A, 'fro')               % Frobenius norm
```

### Matrix Tests

```octave
issymmetric(A)               % Check if symmetric
ishermitian(A)               % Check if Hermitian
isposdef(A)                  % Check if positive definite
```

## Vector Operations

### Vector Creation

```octave
v = [1; 2; 3]                % Column vector
v = [1, 2, 3]                % Row vector
v = 1:5                      % [1, 2, 3, 4, 5]
v = linspace(0, 1, 5)        % 5 points from 0 to 1
```

### Vector Operations

```octave
dot(u, v)                    % Dot product
cross(u, v)                  % Cross product (3D vectors)
norm(v)                      % Vector norm
norm(v, 1)                   % L1 norm
norm(v, 2)                   % L2 norm (default)
norm(v, inf)                 % Infinity norm
```

## Indexing and Manipulation

### Matrix Indexing

```octave
A(i, j)                      % Element at row i, column j
A(i, :)                      % Entire row i
A(:, j)                      % Entire column j
A(1:3, 2:4)                  % Submatrix
A(:)                         % All elements as column vector
```

### Matrix Manipulation

```octave
reshape(A, m, n)             % Reshape matrix
flipud(A)                    % Flip up-down
fliplr(A)                    % Flip left-right
rot90(A)                     % Rotate 90 degrees
tril(A)                      % Lower triangular part
triu(A)                      % Upper triangular part
```

### Matrix Assembly

```octave
[A, B]                       % Horizontal concatenation
[A; B]                       % Vertical concatenation
blkdiag(A, B)                % Block diagonal
kron(A, B)                   % Kronecker product
```

## Advanced Operations

### Matrix Functions

```octave
expm(A)                      % Matrix exponential
logm(A)                      % Matrix logarithm
sqrtm(A)                     % Matrix square root
funm(A, @sin)                % General matrix function
```

### Pseudoinverse

```octave
pinv(A)                      % Moore-Penrose pseudoinverse
pinv(A, tol)                 % With tolerance
```

### Factorizations for Special Matrices

```octave
% Sparse matrices
[L, U] = luinc(A, droptol)   % Incomplete LU
[R] = cholinc(A, droptol)    % Incomplete Cholesky
```

## Common Linear Algebra Problems

### Solving Ax = b

```octave
% Square system
if rank(A) == size(A, 1)
    x = A \ b;               % Unique solution
end

% Overdetermined system (m > n)
x = A \ b;                   % Least squares solution

% Underdetermined system (m < n)
x = pinv(A) * b;            % Minimum norm solution
```

### Computing Matrix Inverse

```octave
% Check if invertible
if det(A) ~= 0
    A_inv = inv(A);
else
    A_inv = pinv(A);         % Pseudoinverse
end
```

### Eigenvalue Problems

```octave
% Standard eigenvalue problem: Ax = λx
[V, D] = eig(A);
eigenvalues = diag(D);

% Generalized eigenvalue problem: Ax = λBx
[V, D] = eig(A, B);
```

## Optimization with Linear Algebra

### Quadratic Forms

```octave
% Evaluate quadratic form x'Ax
quad_form = x' * A * x;

% Minimize quadratic: min(1/2 x'Ax - b'x)
x_opt = A \ b;
```

### Principal Component Analysis (PCA)

```octave
% Center the data
X_centered = X - mean(X, 1);

% Compute covariance matrix
C = cov(X_centered);

% Eigenvalue decomposition
[V, D] = eig(C);

% Sort by eigenvalues (descending)
[eigenvals, idx] = sort(diag(D), 'descend');
eigenvecs = V(:, idx);
```

## Numerical Considerations

### Condition Numbers

```octave
cond(A)                      % Condition number in 2-norm
cond(A, 1)                   % Condition number in 1-norm
condest(A)                   % Estimate condition number
```

### Numerical Stability

```octave
% Check for ill-conditioning
if cond(A) > 1e12
    warning('Matrix is ill-conditioned');
end

% Use regularization for least squares
lambda = 1e-6;
x = (A' * A + lambda * eye(size(A, 2))) \ (A' * b);
```

## Performance Tips

### Efficient Operations

```octave
% Avoid computing inverse explicitly
x = A \ b;                   % Good
x = inv(A) * b;              % Avoid this

% Preallocate matrices
A = zeros(n, n);             % Preallocate
for i = 1:n
    A(i, i) = i;             % Fill elements
end

% Use vectorized operations
A = A .* 2;                  % Vectorized
% Avoid element-wise loops when possible
```

### Memory Considerations

```octave
% For large matrices, consider sparse storage
A_sparse = sparse(A);

% Check memory usage
whos                         % Show variable sizes
```

## Common Patterns

### Matrix Construction Patterns

```octave
% Tridiagonal matrix
n = 5;
A = diag(-2*ones(n,1)) + diag(ones(n-1,1),1) + diag(ones(n-1,1),-1);

% Vandermonde matrix
x = [1; 2; 3; 4];
V = vander(x);

% Hankel matrix
hankel([1, 2, 3], [3, 4, 5])
```

### Iterative Refinement

```octave
x = A \ b;                   % Initial solution
r = b - A * x;               % Residual
dx = A \ r;                  % Correction
x_refined = x + dx;          % Refined solution
```

## Quick Reference Table

| Operation       | Syntax    | Description                  |
| --------------- | --------- | ---------------------------- |
| Matrix multiply | `A * B`   | Standard multiplication      |
| Element-wise    | `A .* B`  | Element-wise operations      |
| Transpose       | `A'`      | Complex conjugate transpose  |
| Inverse         | `inv(A)`  | Matrix inverse               |
| Pseudoinverse   | `pinv(A)` | Moore-Penrose pseudoinverse  |
| Solve system    | `A \ b`   | Solve Ax = b                 |
| Eigenvalues     | `eig(A)`  | Eigenvalue decomposition     |
| SVD             | `svd(A)`  | Singular value decomposition |
| Determinant     | `det(A)`  | Matrix determinant           |
| Rank            | `rank(A)` | Matrix rank                  |
| Condition       | `cond(A)` | Condition number             |
| Norm            | `norm(A)` | Matrix/vector norm           |
