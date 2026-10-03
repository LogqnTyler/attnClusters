from __future__ import annotations

import numpy as np
import torch
from torch.nn.functional import softmax

device = torch.device("cuda")


class Attn_1D:
    def __init__(
        self,
        K: np.ndarray | torch.Tensor | float,
        Q: np.ndarray | torch.Tensor | float,
        V: np.ndarray | torch.Tensor | float,
        X: np.ndarray | torch.Tensor | list[float],
        T: float = 10,
        dt: float = 0.05,
    ) -> None:
        K = torch.as_tensor(K, dtype=torch.float64, device=device)
        Q = torch.as_tensor(Q, dtype=torch.float64, device=device)
        V = torch.as_tensor(V, dtype=torch.float64, device=device)
        X = torch.as_tensor(X, dtype=torch.float64, device=device)

        assert X.ndim == 1, "X should be a vector in R^(num_tokens)."
        assert V.ndim == 0, "V should be a scalar."
        assert K.ndim == 0, "K should be a scalar."
        assert Q.ndim == 0, "Q should be a scalar."
        assert T > 0, "Total time T must be greater than 0."
        assert dt > 0, "Timestep dt must be greater than 0."

        self.K = K
        self.Q = Q
        self.V = V
        self.X_init = X
        self.num_tokens = X.shape[0]
        self.token_dim = 1
        self.T = T
        self.dt = dt
        self.num_steps = int(T / dt) + 1

        self.X = torch.zeros(
            (self.num_steps, self.num_tokens),
            dtype=torch.float64,
            device=device,
        )
        self.dX = torch.zeros_like(self.X)
        self.P = torch.zeros(
            self.num_steps,
            self.num_tokens,
            self.num_tokens,
            dtype=torch.float64,
            device=device,
        )

        self.X[0] = self.X_init
        self.P[0] = softmax(
            (self.Q * self.X[0]).reshape(self.num_tokens, 1)
            @ (self.K * self.X[0]).reshape(1, self.num_tokens),
            dim=-1,
        )
        self.dX[0] = self.P[0] @ (self.V * self.X[0])

    def step_dynamics(self) -> None:
        for step in range(1, self.num_steps):
            self.X[step] = self.X[step - 1] + self.dX[step - 1] * self.dt
            self.P[step] = softmax(
                (self.Q * self.X[step]).reshape(self.num_tokens, 1)
                @ (self.K * self.X[step]).reshape(1, self.num_tokens),
                dim=-1,
            )
            self.dX[step] = self.P[step] @ (self.V * self.X[step])


class Attention:
    def __init__(
        self,
        K: np.ndarray | torch.Tensor,
        Q: np.ndarray | torch.Tensor,
        V: np.ndarray | torch.Tensor,
        X: np.ndarray | torch.Tensor,
        T: float = 5,
        dt: float = 0.05,
        _store_history: bool = True,
    ) -> None:
        assert K.shape == Q.shape, "K and Q must have the same shape"
        assert X.ndim == 2, "X should have two dimensions"
        assert X.shape[0] == Q.shape[1], "Q and K must be of shape d_key x d_token"
        assert V.shape[0] == V.shape[1], "V must be square"
        assert V.shape[0] == X.shape[0], "V must have the same dimension as each token"
        assert T > 0, "Total time T must be greater than 0"
        assert dt > 0, "Timestep dt must be greater than 0"

        self.K = torch.as_tensor(K, dtype=torch.float64, device=device)
        self.Q = torch.as_tensor(Q, dtype=torch.float64, device=device)
        self.V = torch.as_tensor(V, dtype=torch.float64, device=device)
        initial_z = torch.as_tensor(X, dtype=torch.float64, device=device).clone()
        self.num_tokens = initial_z.shape[-1]
        self.token_dim = initial_z.shape[0]
        self.T = T
        self.dt = dt
        self.num_steps = int(T / dt) + 1
        self._store_history = _store_history

        if self._store_history:
            self.X_init = initial_z
            self.P = torch.zeros(
                self.num_steps,
                self.num_tokens,
                self.num_tokens,
                dtype=torch.float64,
                device=device,
            )
            self.Z = torch.zeros(
                self.num_steps,
                self.token_dim,
                self.num_tokens,
                dtype=torch.float64,
                device=device,
            )
            self.dZ = torch.zeros_like(self.Z)
            self.Z[0] = self.X_init
            self.P[0] = softmax(
                (self.Q @ self.Z[0]).T @ (self.K @ self.Z[0]),
                dim=-1,
            )
            self.dZ[0] = self.V @ (self.Z[0] @ self.P[0].T - self.Z[0])
        else:
            self.Z = initial_z
            self.X = self.Z.clone()
            self.P = softmax(
                (self.Q @ self.X).T @ (self.K @ self.X),
                dim=-1,
            )
            self.dZ = self.V @ (self.Z @ self.P.T - self.Z)
            self.dX = self.V @ (self.X @ self.P.T)

    def rescaled_dynamics(self) -> None:
        if not self._store_history:
            self._rescaled_dynamics_forgetful()
            return

        for step in range(1, self.num_steps):
            self.Z[step] = self.Z[step - 1] + self.dZ[step - 1] * self.dt
            exp_v = torch.matrix_exp(step * self.dt * self.V)
            x = exp_v @ self.Z[step]
            self.P[step] = softmax(
                (self.Q @ x).T @ (self.K @ x),
                dim=-1,
            )
            self.dZ[step] = self.V @ (self.Z[step] @ self.P[step].T - self.Z[step])

    def _rescaled_dynamics_forgetful(self) -> None:
        for step in range(1, self.num_steps):
            self.Z = self.Z + self.dZ * self.dt
            self.X = torch.matrix_exp(step * self.dt * self.V) @ self.Z
            self.P = softmax(
                (self.Q @ self.X).T @ (self.K @ self.X),
                dim=-1,
            )
            self.dZ = self.V @ (self.Z @ self.P.T - self.Z)
            self.dX = self.V @ (self.X @ self.P.T)


class ForgetfulAttention(Attention):
    def __init__(
        self,
        K: np.ndarray | torch.Tensor,
        Q: np.ndarray | torch.Tensor,
        V: np.ndarray | torch.Tensor,
        X: np.ndarray | torch.Tensor,
        T: float = 5,
        dt: float = 0.05,
    ) -> None:
        super().__init__(
            K=K,
            Q=Q,
            V=V,
            X=X,
            T=T,
            dt=dt,
            _store_history=False,
        )


__all__ = ["Attention", "Attn_1D", "ForgetfulAttention", "device"]
