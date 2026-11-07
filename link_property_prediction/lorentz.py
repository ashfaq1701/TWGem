"""Lorentz model of hyperbolic geometry in intrinsic coordinates (Nickel & Kiela, ICML 2018)."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import geoopt
import torch
from torch import Tensor

__all__ = ["LorentzManifold"]

_THIRD = 1.0 / 3.0
_SIXTH = 1.0 / 6.0


def _safe_sqrt(t: Tensor) -> Tensor:
    """sqrt(t) with a zero gradient at t == 0; NaN in t propagates."""
    pos = t > 0
    out = torch.where(pos, torch.sqrt(torch.where(pos, t, torch.ones_like(t))), torch.zeros_like(t))
    return torch.where(torch.isnan(t), t, out)


def _fmax(t: Tensor) -> float:
    """Largest finite value of t's dtype."""
    return torch.finfo(t.dtype).max


def _tiny(t: Tensor) -> float:
    """Smallest positive normal of t's dtype."""
    return torch.finfo(t.dtype).tiny


class LorentzManifold(geoopt.Manifold):
    """Points are x' in R^n; x0 = sqrt(k + |x'|^2) is derived, never stored. Curvature -1/k."""

    name = "LorentzManifold"
    ndim = 1
    reversible = False

    def __init__(self, k: float = 1.0) -> None:
        super().__init__()
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        self.k = float(k)
        self._sqrt_k = math.sqrt(k)
        self._inv_k = 1.0 / k
        self._inv_sqrt_k = 1.0 / math.sqrt(k)
        self._half_inv_k = 0.5 / k

    def extra_repr(self) -> str:
        return f"k={self.k}"

    def _x0(self, x: Tensor) -> Tensor:
        """Time coordinate, Eq. 6: sqrt(k + |x'|^2)."""
        return torch.sqrt(self.k + (x * x).sum(-1, keepdim=True))

    def _lift(self, x: Tensor) -> Tensor:
        """[..., n] -> [..., n+1] ambient point."""
        return torch.cat([self._x0(x), x], -1)

    def _gap(self, x: Tensor, y: Tensor) -> Tensor:
        """w = -<X,Y>_L / k - 1 >= 0, formed without cancellation for nearby points."""
        x0, y0 = self._x0(x), self._x0(y)
        d = x - y
        d0 = (d * (x + y)).sum(-1, keepdim=True) / (x0 + y0)
        w = ((d * d).sum(-1, keepdim=True) - d0 * d0) * self._half_inv_k
        return w.clamp_min(0.0)

    def projx(self, x: Tensor) -> Tensor:
        """Identity: every x' in R^n is a point."""
        return x

    def proju(self, x: Tensor, u: Tensor) -> Tensor:
        """Identity: the tangent space is all of R^n."""
        return u

    def inner(self, x: Tensor, u: Tensor, v: Optional[Tensor] = None,
              *, keepdim: bool = False) -> Tensor:
        """Pulled-back metric u.v - (x'.u)(x'.v)/x0^2, in a split form that cannot cancel."""
        same = v is None or v is u
        x0sq = self.k + (x * x).sum(-1, keepdim=True)
        qu = (x * u).sum(-1, keepdim=True) / x0sq
        if same:
            v, qv = u, qu
        else:
            qv = (x * v).sum(-1, keepdim=True) / x0sq
        res = (((u - qu * x) * (v - qv * x)).sum(-1, keepdim=True) + qu * qv * self.k)
        return res if keepdim else res.squeeze(-1)

    def egrad2rgrad(self, x: Tensor, u: Tensor) -> Tensor:
        """Riemannian gradient (Eq. 10 in this chart): u + x' (x'.u) / k."""
        return u + x * ((x * u).sum(-1, keepdim=True) * self._inv_k)

    def expmap(self, x: Tensor, u: Tensor) -> Tensor:
        """Eq. 9 at curvature -1/k; the step's tangent norm is the geodesic distance travelled."""
        qq = self.inner(x, u, keepdim=True).clamp(0.0, _fmax(u))   # ||v||_L^2
        tsq = qq * self._inv_k                               # t^2

        # Mask the sqrt's input, not its output: the untaken branch's gradient would be NaN.
        safe = qq > _tiny(qq)
        nrm = torch.where(safe, qq, torch.ones_like(qq)).sqrt()
        t = nrm * self._inv_sqrt_k
        cosh_t = torch.where(safe, torch.cosh(t), 1.0 + tsq * 0.5)
        coef = torch.where(safe, torch.sinh(t) * (self._sqrt_k / nrm),
                           1.0 + tsq * _SIXTH)
        return torch.addcmul(cosh_t * x, coef, u)

    def retr(self, x: Tensor, u: Tensor) -> Tensor:
        """The closed-form exponential map."""
        return self.expmap(x, u)

    def _check_point_on_manifold(self, x: Tensor, *, atol: float = 1e-5,
                                 rtol: float = 1e-5) -> Tuple[bool, Optional[str]]:
        """Only finiteness can fail."""
        if not torch.isfinite(x).all():
            return False, "point contains non-finite values"
        return True, None

    def _check_vector_on_tangent(self, x: Tensor, u: Tensor, *, atol: float = 1e-5,
                                 rtol: float = 1e-5) -> Tuple[bool, Optional[str]]:
        """Any finite u is tangent."""
        if not torch.isfinite(u).all():
            return False, "tangent vector contains non-finite values"
        return True, None

    def dist(self, x: Tensor, y: Tensor, *, keepdim: bool = False) -> Tensor:
        """Geodesic distance (Eq. 5) as 2 sqrt(k) asinh(sqrt(w/2)); exactly 0, zero gradient, at x == y."""
        res = (2.0 * self._sqrt_k) * torch.asinh(_safe_sqrt(self._gap(x, y) * 0.5))
        return res if keepdim else res.squeeze(-1)

    def dist0(self, x: Tensor, *, keepdim: bool = False) -> Tensor:
        """Geodesic distance to the origin, formed without cancellation near it."""
        n2 = (x * x).sum(-1, keepdim=True)
        x0 = torch.sqrt(self.k + n2)
        w = (n2 / (self._sqrt_k * (x0 + self._sqrt_k))).clamp_min(0.0)
        res = (2.0 * self._sqrt_k) * torch.asinh(_safe_sqrt(w * 0.5))
        return res if keepdim else res.squeeze(-1)

    def logmap(self, x: Tensor, y: Tensor) -> Tensor:
        """Inverse of expmap: arcosh(a) / sqrt(a^2 - 1) * (Y - a X), a = -<X,Y>_L / k."""
        w = self._gap(x, y)
        a = w + 1.0
        safe = w > _tiny(w)
        w_safe = torch.where(safe, w, torch.ones_like(w))
        num = 2.0 * torch.asinh(torch.sqrt(w_safe * 0.5))    # == arcosh(a)
        den = torch.sqrt(w_safe * (w_safe + 2.0))            # == sqrt(a^2 - 1)
        coef = torch.where(safe, num / den, 1.0 - w * _THIRD)
        return coef * (y - a * x)

    def transp(self, x: Tensor, y: Tensor, v: Tensor) -> Tensor:
        """Parallel transport T_x -> T_y; the radial/perpendicular grouping below is load-bearing."""
        x0, y0 = self._x0(x), self._x0(y)
        d = y - x
        big_p = (d * (x + y)).sum(-1, keepdim=True)
        d0 = big_p / (y0 + x0)
        dd = (d * d).sum(-1, keepdim=True)
        w = ((dd - d0 * d0) * self._half_inv_k).clamp_min(0.0)
        nx = x.norm(dim=-1, keepdim=True)
        n = x / nx.clamp_min(math.sqrt(_tiny(nx)))           # floor: the smallest |x| whose square is normal
        vr = (n * v).sum(-1, keepdim=True)
        vperp = v - vr * n
        dr = (n * d).sum(-1, keepdim=True)
        dperp = d - dr * n
        dp2 = (dperp * dperp).sum(-1, keepdim=True)
        a = nx + dr
        conj = x0 * a + y0 * nx
        direct = x0 * a - y0 * nx
        num = self.k * dr * (2.0 * nx + dr) - nx * nx * dp2
        pos = conj > 0                                       # where a < 0, `direct` is already exact
        rational = num / torch.where(pos, conj, torch.ones_like(conj))
        yv = vr * (torch.where(pos, rational, direct) / x0) \
             + (dperp * vperp).sum(-1, keepdim=True)
        coef = yv / (self.k * (2.0 + w))
        return torch.addcmul(v, coef, x + y)

    def weighted_midpoint(self, x: Tensor, w: Tensor) -> Tensor:
        """Weighted Lorentzian centroid (Law et al., ICML 2019): x [..., T, n], w [..., T] -> [..., n]."""
        wu = w.unsqueeze(-1)
        s0 = (wu * self._x0(x)).sum(dim=-2)
        s = (wu * x).sum(dim=-2)
        neg = (s0 * s0 - (s * s).sum(-1, keepdim=True)) * self._inv_k
        wsum = w.sum(-1, keepdim=True)
        return s * neg.clamp_min((wsum * wsum).clamp_min(_tiny(neg))).rsqrt()

    def to_poincare(self, x: Tensor) -> Tensor:
        """Eq. 11: x' / (x0 + sqrt(k)), unit Poincare-ball coordinates."""
        return x / (self._x0(x) + self._sqrt_k)

    def from_poincare(self, u: Tensor) -> Tensor:
        """Inverse of Eq. 11."""
        n2 = (u * u).sum(-1, keepdim=True)
        return (2.0 * self._sqrt_k) * u / (1.0 - n2).clamp_min(_tiny(u))

    def random(self, *size, dtype=None, device=None) -> Tensor:
        """Wrapped normal at the origin, u ~ N(0, I / n): d(x, O) = |u| concentrates at 1."""
        u = torch.randn(*size, dtype=dtype, device=device) * size[-1] ** -0.5
        return geoopt.ManifoldTensor(self.expmap(torch.zeros_like(u), u), manifold=self)

    def origin(self, *size, dtype=None, device=None, seed: int = 42) -> Tensor:
        """The vertex x' = 0; `seed` is unused (geoopt.Manifold.origin's signature)."""
        return geoopt.ManifoldTensor(
            torch.zeros(*size, dtype=dtype, device=device), manifold=self
        )
