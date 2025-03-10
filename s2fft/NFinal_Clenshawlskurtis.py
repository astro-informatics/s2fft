#  Based on the function available here at:
# ttps://github.com/inducer/modepy/blob/main/modepy/quadrature/clenshaw_curtis.py

# Three options to chosse from
#  (a) clenshaw_curtis
#  (b) Fejer1
#  (c) Fejer2
# 


import numpy as np
import jax.numpy as jnp
import jax

# Test Function
jax.config.update("jax_enable_x64", True)


# Test Function
def f(x):
    return x**2 + np.sin(x)**2 + np.exp(-x)/(1+ x**2)

# Number of Nodes
num=7500

######################################################################################
def make_clenshaw_curtis_nodes_and_weights(n: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Nodes and weights of the Clenshaw-Curtis quadrature."""
######################################################################################

    if n < 1:
        raise ValueError(f"Clenshaw-Curtis order must be at least 1: n = {n}")

    if n == 1:
        return jnp.array([-1, 1]), jnp.array([1, 1])

    N = jnp.arange(1, n, 2)  # noqa: N806
    r = len(N)
    m = n - r

    # Clenshaw-Curtis nodes
    x = jnp.cos(jnp.arange(0, n + 1) * jnp.pi / n)

    # Clenshaw-Curtis weights
    w = jnp.concatenate([2 / N / (N - 2), 1 / N[-1:], jnp.zeros(m)])
    w = 0 - w[:-1] - w[-1:0:-1]
    g0: jnp.ndarray[tuple[int, ...], jnp.dtype[np.floating]] = -np.ones(n)
    g0[r] = g0[r] + n
    g0[m] = g0[m] + n
    g0 = g0 / (n**2 - 1 + (n % 2))
    w = jnp.fft.ifft(w + g0)
    assert jnp.allclose(w.imag, 0)

    wr = w.real
    return x, jnp.concatenate([wr, wr[:1]])

a, b = make_clenshaw_curtis_nodes_and_weights(num) 

print("Quadrature")
print("")

approx = sum(b * f(a))

print("Clenshaw_kurtis=", approx)


######################################################################################
def make_fejer1_nodes_and_weights(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Nodes and weights of the Fejer quadrature of the first kind."""
######################################################################################

    if n < 1:
        raise ValueError(f"Fejer1 order must be at least 1: n = {n}")

    N = jnp.arange(1, n, 2)  
    r = len(N)
    m = n - r
    K = jnp.arange(0, m)  

    # Fejer1 nodes: k = 1/2, 3/2, ..., n-1/2
    x = jnp.cos((jnp.arange(0, n) + 0.5) * jnp.pi / n)

    # Fejer1 weights
    w = jnp.concatenate([
        2 * jnp.exp(1j * jnp.pi * K / n) / (1 - 4 * (K**2)), jnp.zeros(r + 1)
        ])
    w = w[:-1] + jnp.conj(w[-1:0:-1])
    w = jnp.fft.ifft(w)

    assert jnp.allclose(w.imag, 0)
    return x, w.real


######################################################################################
def make_fejer2_nodes_and_weights(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Nodes and weights of the Fejer quadrature of the second kind."""
######################################################################################

    if n < 2:
        raise ValueError(f"Fejer2 order must be at least 2: n = {n}")

    N = jnp.arange(1, n, 2)  
    r = len(N)
    m = n - r

    # Fejer2 nodes: k=0, 1, ..., n
    x = jnp.cos(jnp.arange(1, n) * jnp.pi / n)

    # Fejer2 weights
    w = jnp.concatenate([2 / N / (N - 2), 1 / N[-1:], jnp.zeros(m)])
    w = 0 - w[:-1] - w[-1:0:-1]
    w = jnp.fft.ifft(w)[1:]

    assert jnp.allclose(w.imag, 0)
    return x, w.real


######################################################################################

    if n < 2:
        raise ValueError(f"Fejer2 order must be at least 2: n = {n}")

    N = jnp.arange(1, n, 2)  
    r = len(N)
    m = n - r

    # Fejer2 nodes: k=0, 1, ..., n
    x = jnp.cos(jnp.arange(1, n) * jnp.pi / n)

    # Fejer2 weights
    w = jnp.concatenate([2 / N / (N - 2), 1 / N[-1:], jnp.zeros(m)])
    w = 0 - w[:-1] - w[-1:0:-1]
    w = jnp.fft.ifft(w)[1:]

    assert jnp.allclose(w.imag, 0)
    return x, w.real


print()

a, b = make_fejer1_nodes_and_weights(num)

#  print()

a, b = make_fejer1_nodes_and_weights(num)

# print()
# print("fejer1_nodes = ",a)
# print("fejer1_weights  = ", b)



approx1 = sum(b * f(a))

print("fejer1 =",  approx1)



print(); 

c, d = make_fejer2_nodes_and_weights(num)


# print("fejer2_nodes = ",c)
# print("fejer2_weights  = ", d)



approx2 = sum(d * f(c))

print("Fejer2=",  approx2)

print(); print(); print()
