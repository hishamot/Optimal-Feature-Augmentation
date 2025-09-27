### Optimal Transformations Selection at $L_i$

The optimal feature augmentation strategy can be divided into two phases:

1. Optimal feature selection phase
2. Model training phase

Similar to finding optimal transformations at the input level ($L_0$), the approach can be extended to any intermediate level feature map $F_i$ at location $L_i$.

First, consider the feature map $F_1$ (at $L_1$) defined as:

$$
F_1 = C_1(F_0')
\tag{5a}
$$

where $C_1$ denotes the convolution operation between $L_0$ and $L_1$.

At each intermediate location, consider $n$ arbitrary transformations
$\mathbf{t} = \{t_1, t_2, \dots, t_n\}$
and the corresponding binary variables
$\mathbf{x} = [x_1, x_2, \dots, x_n]$.

As discussed previously, the coefficient values $\beta_i$ and the corresponding $p$-value of the associated $x_i$ in $L_1$ are computed. The resultant subset of optimal transformations at $L_1$ is denoted by $T_1$, and the corresponding transformed feature map is denoted by $F_1'$:

$$
F_1' = T_1(F_1)
$$

At location $L_1$, the augmented feature map $\tilde{F}_1$ is computed as:

$$
\tilde{F}_1 =
\begin{cases}
[F_1, F_1'], & \text{if } T_1 \neq \varnothing \\ \\
F_1, & \text{otherwise.}
\end{cases}
\tag{6}
$$

Using $\tilde{F}_1$ as the feature map at location $L_1$ and passing it to the next layer (instead of $F_1'$) doubles the computation for the rest of the model. Consequently, the feature map $F_2$ at location $L_2$ is given by:

$$
F_2 = C_2(\tilde{F}_1)
\tag{7}
$$

where $C_2$ is the convolution operation between $L_1$ and $L_2$.

At location $L_2$, the selection of optimal features is performed using $F_2$, which has double the number of features compared to $F_1$.

In the general case, the feature map at location $L_i$ is given by:

$$
F_i = C_i(\tilde{F}_{i-1})
$$

where $C_i$ denotes the convolution block between locations $L_{i-1}$ and $L_i$, and $\tilde{F}_{i-1}$ is the augmented feature map at $L_{i-1}$.

With at least one transformation selected at each $L_i$, $F_i$ has $2^{i-1}$ times the number of feature maps as $F_1$. This $F_i$ is further used to find the set of optimal transformations $T_i$ at $L_i$.

Thus, the augmented feature map at $L_i$ is given by:

$$
F_i' = T_i(F_i)
$$

$$
\tilde{F}_i =
\begin{cases}
[F_i, F_i'], & \text{if } T_i \neq \varnothing \\ \\
F_i, & \text{otherwise.}
\end{cases}
\tag{8}
$$
