import numpy as np

import numpy as np

def generate_height_field_fractal(
    nx: int, ny: int,
    size_x: float, size_y: float,
    alpha: float = 2.2,        # larger => smoother; ~[1.8, 3.0]
    amp: float = 3.0,          # overall vertical scale (meters)
    noise_sigma: float = 0.0,
    seed: int | None = None,
):
    """
    Generates a non-periodic-looking terrain using 1/f^alpha spectral shaping.
    """
    rng = np.random.default_rng(seed)

    # White noise in spatial domain
    w = rng.normal(0.0, 1.0, size=(ny, nx))

    # FFT -> shape spectrum
    W = np.fft.rfft2(w)

    # Frequency grid (cycles per meter)
    fx = np.fft.rfftfreq(nx, d=size_x / (nx - 1))
    fy = np.fft.fftfreq(ny, d=size_y / (ny - 1))
    FX, FY = np.meshgrid(fx, fy, indexing="xy")
    F = np.sqrt(FX**2 + FY**2)
    F[0, 0] = np.inf  # avoid divide by zero at DC

    # Apply 1/f^(alpha/2) to amplitude (power ~ 1/f^alpha)
    W_shaped = W / (F ** (alpha / 2.0))

    z = np.fft.irfft2(W_shaped, s=(ny, nx))

    # Normalize and scale
    z = z - np.mean(z)
    z = z / (np.std(z) + 1e-12)
    z = amp * z

    if noise_sigma > 0:
        z += rng.normal(0.0, noise_sigma, size=z.shape)

    x = np.linspace(0.0, size_x, nx)
    y = np.linspace(0.0, size_y, ny)
    X, Y = np.meshgrid(x, y, indexing="xy")
    return X, Y, z

def generate_height_field(
    nx: int,
    ny: int,
    size_x: float,
    size_y: float,
    components=None,
    noise_sigma: float = 0.0,
    seed: int | None = None,
):
    """
    Synthetic height field z=h(x,y) as sum of random directional sinusoids at multiple scales.
    components: [{"amp": float, "scale": float}, ...] where scale ~ wavelength in meters.
    """
    rng = np.random.default_rng(seed)

    if components is None:
        components = [
            {"amp": 2.0, "scale": 50.0},
            {"amp": 0.8, "scale": 18.0},
            {"amp": 0.25, "scale": 6.0},
        ]

    x = np.linspace(0.0, size_x, nx)
    y = np.linspace(0.0, size_y, ny)
    X, Y = np.meshgrid(x, y, indexing="xy")
    Z = np.zeros_like(X, dtype=float)

    for c in components:
        amp = float(c["amp"])
        scale = float(c["scale"])
        theta = rng.uniform(0, 2*np.pi)
        phase1 = rng.uniform(0, 2*np.pi)
        phase2 = rng.uniform(0, 2*np.pi)

        k = 2*np.pi / max(scale, 1e-9)
        U = np.cos(theta) * X + np.sin(theta) * Y

        Z += amp * np.sin(k * U + phase1)
        Z += 0.35 * amp * np.sin(2 * k * U + phase2)

    if noise_sigma > 0:
        Z += rng.normal(0.0, noise_sigma, size=Z.shape)

    return X, Y, Z