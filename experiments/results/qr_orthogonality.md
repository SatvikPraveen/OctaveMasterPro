# Loss of orthogonality vs. condition number

100x30 matrices, median of 5 per kappa, kappa = 1e1..1e15.

| Method | Fitted slope d log(loss) / d log(kappa) | Theory | Max backward error |
|---|---|---|---|
| cgs | 2.03 | 2 | 1.4e-16 |
| mgs | 0.94 | 1 | 1.6e-16 |
| cgs2 | 0.00 | 0 | 1.4e-16 |
| householder | 0.00 | 0 | 1.5e-15 |

Slopes are fitted where the loss is below 1e-2, before saturation.
