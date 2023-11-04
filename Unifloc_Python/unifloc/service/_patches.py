"""
Модуль с патчами для сторонних библиотек, применяемыми в рачетных модулях
"""
import warnings as warn

import numpy as np
import scipy.integrate as integr


base_init_ode = integr.OdeSolver.__init__
base_init_rk = integr.RK23.__init__
base_step_rk = integr.RK23._step_impl


def patch_solve_ivp():
    """
    Патч для scipy.solve_ivp, передает (t,t_old), (y, y_old) вместо t, y в интегрируемую функцию.
    Используется в _pipe.py .
    """
    EPS = np.finfo(float).eps
    SAFETY = 0.9

    MIN_FACTOR = 0.2  # Minimum allowed decrease in a step size.
    MAX_FACTOR = 10

    # базовый метод
    def validate_max_step(max_step):
        """Assert that max_Step is valid and return it."""
        if max_step <= 0:
            raise ValueError("`max_step` must be positive.")
        return max_step

    # базовый метод
    def validate_tol(rtol, atol, n):
        """Validate tolerance values."""
        if rtol < 100 * EPS:
            warn("`rtol` is too low, setting to {}".format(100 * EPS))
            rtol = 100 * EPS
        atol = np.asarray(atol)
        if atol.ndim > 0 and atol.shape != (n,):
            raise ValueError("`atol` has wrong shape.")

        if np.any(atol < 0):
            raise ValueError("`atol` must be positive.")

        return rtol, atol

    # базовый метод
    def warn_extraneous(extraneous):
        if extraneous:
            warn(
                "The following arguments have no effect for a chosen solver: {}.".format(
                    ", ".join("`{}`".format(x) for x in extraneous)
                )
            )

    # доработанный метод
    def check_argumentsv2(fun, y0, support_complex):
        """Helper function for checking arguments common to all solvers."""
        y0 = np.asarray(y0)
        if np.issubdtype(y0.dtype, np.complexfloating):
            if not support_complex:
                raise ValueError(
                    "`y0` is complex, but the chosen solver does "
                    "not support integration in a complex domain."
                )
            dtype = complex
        else:
            dtype = float
        y0 = y0.astype(dtype, copy=False)

        if y0.ndim != 1:
            raise ValueError("`y0` must be 1-dimensional.")

        def fun_wrapped(t, y, t_old, y_old):
            t = (t, t_old)
            y = (y, y_old)
            retult = np.asarray(fun(t, y), dtype=dtype)
            return retult

        return fun_wrapped, y0

    # доработанный метод
    def rk_stepv2(fun, t, y, f, h, A, B, C, K):
        K[0] = f
        t_old = t
        y_old = y
        for s, (a, c) in enumerate(zip(A[1:], C[1:]), start=1):
            dy = np.dot(K[:s].T, a[:s]) * h
            K[s] = fun(t + c * h, y + dy, t_old, y_old)[0]
            t_old = t + c * h
            y_old = y + dy
        y_new = y + h * np.dot(K[:-1].T, B)
        f_new = fun(t + h, y_new, t, y)[0]

        K[-1] = f_new

        return y_new, f_new

    # доработанный метод
    def _step_implv2(self):
        t = self.t
        y = self.y

        max_step = self.max_step
        rtol = self.rtol
        atol = self.atol

        min_step = 10 * np.abs(np.nextafter(t, self.direction * np.inf) - t)

        if self.h_abs > max_step:
            h_abs = max_step
        elif self.h_abs < min_step:
            h_abs = min_step
        else:
            h_abs = self.h_abs

        step_accepted = False
        step_rejected = False

        while not step_accepted:
            if h_abs < min_step:
                return False, self.TOO_SMALL_STEP

            h = h_abs * self.direction
            t_new = t + h

            if self.direction * (t_new - self.t_bound) > 0:
                t_new = self.t_bound

            h = t_new - t
            h_abs = np.abs(h)

            y_new, f_new = rk_stepv2(
                self.fun, t, y, self.f, h, self.A, self.B, self.C, self.K
            )
            scale = atol + np.maximum(np.abs(y), np.abs(y_new)) * rtol
            error_norm = self._estimate_error_norm(self.K, h, scale)

            if error_norm < 1:
                if error_norm == 0:
                    factor = MAX_FACTOR
                else:
                    factor = min(MAX_FACTOR, SAFETY * error_norm ** self.error_exponent)

                if step_rejected:
                    factor = min(1, factor)

                h_abs *= factor

                step_accepted = True
            else:
                h_abs *= max(MIN_FACTOR, SAFETY * error_norm ** self.error_exponent)
                step_rejected = True

        self.h_previous = h
        self.y_old = y

        self.t = t_new
        self.y = y_new

        self.h_abs = h_abs
        self.f = f_new

        return True, None

    # базовый метод
    def norm(x):
        """Compute RMS norm."""
        return np.linalg.norm(x) / x.size ** 0.5

    # базовый метод
    def select_initial_step(fun, t0, y0, f0, direction, order, rtol, atol):
        if y0.size == 0:
            return np.inf

        scale = atol + np.abs(y0) * rtol
        d0 = norm(y0 / scale)
        d1 = norm(f0 / scale)
        if d0 < 1e-5 or d1 < 1e-5:
            h0 = 1e-6
        else:
            h0 = 0.01 * d0 / d1

        y1 = y0 + h0 * direction * f0

        f1 = fun(t0 + h0 * direction, y1, t0, y0)[0]
        d2 = norm((f1 - f0) / scale) / h0

        if d1 <= 1e-15 and d2 <= 1e-15:
            h1 = max(1e-6, h0 * 1e-3)
        else:
            h1 = (0.01 / max(d1, d2)) ** (1 / (order + 1))

        return min(100 * h0, h1)

    # базовый метод
    def validate_first_step(first_step, t0, t_bound):
        """Assert that first_step is valid and return it."""
        if first_step <= 0:
            raise ValueError("`first_step` must be positive.")
        if first_step > np.abs(t_bound - t0):
            raise ValueError("`first_step` exceeds bounds.")
        return first_step

    # новый init для базового класса OdeSolver
    def new_initode(self, fun, t0, y0, t_bound, vectorized, support_complex=False):
        self.t_old = None
        self.t = t0
        self._fun, self.y = check_argumentsv2(fun, y0, support_complex)
        self.t_bound = t_bound
        self.vectorized = vectorized

        if vectorized:

            def fun_single(t, y, t_old, y_old):
                return self._fun(t, y[:, None], t_old, y_old).ravel()

            fun_vectorized = self._fun
        else:
            fun_single = self._fun

            def fun_vectorized(t, y):
                f = np.empty_like(y)
                for i, yi in enumerate(y.T):
                    f[:, i] = self._fun(t, yi)
                return f

        def fun(t, y, t_old, y_old):
            self.nfev += 1
            return self.fun_single(t, y, t_old, y_old), self.t_old

        self.fun = fun
        self.fun_single = fun_single
        self.fun_vectorized = fun_vectorized

        self.direction = np.sign(t_bound - t0) if t_bound != t0 else 1
        self.n = self.y.size
        self.status = "running"

        self.nfev = 0
        self.njev = 0
        self.nlu = 0

    # новый init для метода RK23
    def new_initrk(
        self,
        fun,
        t0,
        y0,
        t_bound,
        max_step=np.inf,
        rtol=1e-3,
        atol=1e-6,
        vectorized=False,
        first_step=None,
        **extraneous
    ):
        warn_extraneous(extraneous)
        new_initode(self, fun, t0, y0, t_bound, vectorized, support_complex=True)
        self.y_old = None
        self.max_step = validate_max_step(max_step)
        self.rtol, self.atol = validate_tol(rtol, atol, self.n)
        self.f, self.ff = self.fun(self.t, self.y, self.t_old, self.y_old)
        if first_step is None:
            self.h_abs = select_initial_step(
                self.fun,
                self.t,
                self.y,
                self.f,
                self.direction,
                self.error_estimator_order,
                self.rtol,
                self.atol,
            )
        else:
            self.h_abs = validate_first_step(first_step, t0, t_bound)
        self.K = np.empty((self.n_stages + 1, self.n), dtype=self.y.dtype)
        self.error_exponent = -1 / (self.error_estimator_order + 1)
        self.h_previous = None

    # заменяем старые init новыми
    integr.OdeSolver.__init__ = new_initode
    integr.RK23.__init__ = new_initrk
    integr.RK23._step_impl = _step_implv2


def depatch_solve_ivp():
    """
    Функция для сброса измененных методов solve_ivp. Не дает произойти ошибкам в других модулях юнифлока.
    """
    integr.OdeSolver.__init__ = base_init_ode
    integr.RK23.__init__ = base_init_rk
    integr.RK23._step_impl = base_step_rk


def solve_ode(
    fun,
    t_span,
    y0,
    method="RK23",
    t_eval=None,
    dense_output=False,
    events=None,
    vectorized=False,
    args=None,
    **options
):
    """
    Функция для вызова запатченной версии solve_ivp.
    Патч для scipy.solve_ivp, передает (t,t_old), (y, y_old) вместо t, y в интегрируемую функцию.

    Альтернатива для применения в _pipe.py .
    """
    patch_solve_ivp()

    solution = integr.solve_ivp(
        fun, t_span=t_span, y0=y0, method=method, args=args, t_eval=t_eval, events=events,
    )

    depatch_solve_ivp()
    return solution
