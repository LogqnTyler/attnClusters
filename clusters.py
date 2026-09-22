import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import torch
    from torch.nn.functional import softmax
    from torch.linalg import matrix_rank
    import plotly.express as px

    return mo, np, softmax, torch


@app.cell
def _(np, softmax, torch):
    class Attn_1D:
        def __init__(
            self,
            K: np.ndarray | torch.Tensor | float,
            Q: np.ndarray | torch.Tensor | float,
            V: np.ndarray | torch.Tensor | float,
            X: np.ndarray
            | torch.Tensor
            | list[float],  # shape: token_dimension x num_tokens
            T=6,
            dt=0.05,
        ) -> None:

            if isinstance(K, (int, float)):
                K = torch.tensor(K)
            if isinstance(Q, (int, float)):
                Q = torch.tensor(Q)
            if isinstance(V, (int, float)):
                V = torch.tensor(V)
            if isinstance(X, list):
                X = torch.tensor(X)

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
            )
            self.dX = torch.zeros(
                (self.num_steps, self.num_tokens), dtype=torch.float64
            )

            self.P = torch.zeros(
                self.num_steps,
                self.num_tokens,
                self.num_tokens,
                dtype=torch.float64,
            )

            # initialize X(0)
            self.X[0] = self.X_init
            # Initialize P(0)
            self.P[0] = softmax(
                (self.Q * self.X[0]).reshape(self.num_tokens, 1)
                @ (self.K * self.X[0]).reshape(1, self.num_tokens),
                dim=-1,
            )
            self.dX[0] = self.P[0] @ (self.V * self.X[0])

        def step_dynamics(self):
            for step in range(1, self.num_steps):
                self.X[step] = self.X[step - 1] + self.dX[step - 1] * self.dt

                self.P[step] = softmax(
                    (self.Q * self.X[step]).reshape(self.num_tokens, 1)
                    @ (self.K * self.X[step]).reshape(1, self.num_tokens),
                    dim=-1,
                )
                self.dX[step] = self.P[step] @ (self.V * self.X[step])

    return (Attn_1D,)


@app.cell
def _(mo):
    num_tokens_input = mo.ui.number(
        start=2,
        stop=100,
        step=1,
        value=48,
        debounce=True,
        label="Tokens",
    )
    q_input = mo.ui.number(
        start=-5,
        stop=5,
        step=0.1,
        value=-2,
        debounce=True,
        label="Q",
    )
    k_input = mo.ui.number(
        start=-5,
        stop=5,
        step=0.1,
        value=-2,
        debounce=True,
        label="K",
    )
    v_input = mo.ui.number(
        start=-5,
        stop=5,
        step=0.1,
        value=0.2,
        debounce=True,
        label="V",
    )

    _parameter_controls = mo.hstack(
        [num_tokens_input, q_input, k_input, v_input],
        justify="start",
        gap=1.5,
    )
    attention_controls = mo.vstack(
        [
            mo.md("**Attention dynamics Demo**  \n"),
            _parameter_controls,
        ],
        gap=1,
    )
    return attention_controls, k_input, num_tokens_input, q_input, v_input


@app.cell
def _(attention_controls, attention_plot, mo):
    mo.vstack(
        [attention_controls, attention_plot],
        gap=1.5,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The above demo demonstrated the first main result of the paper.
    """)
    return


@app.cell
def _():
    return


@app.cell
def _(np, softmax, torch):
    class Attention:
        def __init__(
            self,
            K: np.ndarray | torch.Tensor,
            Q: np.ndarray | torch.Tensor,
            V: np.ndarray | torch.Tensor,
            X: np.ndarray
            | torch.Tensor,  # shape: token_dimension x num_tokens
            T=5,
            dt=0.05,
        ) -> None:

            assert K.shape == Q.shape, "K and Q must have the same shape"

            assert X.ndim == 2, "X should have two dimension"
            assert X.shape[0] == Q.shape[1], (
                "Q and K must be of shape d_key x d_token. Q.shape[1] does not equal d_token."
            )
            assert V.shape[0] == V.shape[1], (
                "V must be square: it is a linear operator on tokens x_i."
            )
            assert V.shape[0] == X.shape[0], (
                "V must have the same dimension as the token dimension."
            )

            assert T > 0, "Total time T must be greater than 0."
            assert dt > 0, "Timestep dt must be greater than 0."
            self.K = torch.as_tensor(K, dtype=torch.float64)
            self.Q = torch.as_tensor(Q, dtype=torch.float64)
            self.V = torch.as_tensor(V, dtype=torch.float64)
            self.X_init = torch.as_tensor(X, dtype=torch.float64)
            self.num_tokens = X.shape[-1]
            self.token_dim = X.shape[0] if X.ndim > 1 else 1
            self.T = T
            self.dt = dt
            self.num_steps = int(T / dt) + 1

            # self.X = torch.zeros(
            #     (self.num_steps, self.token_dim, self.num_tokens),
            #     dtype=torch.float64,
            # )
            # self.dX = torch.zeros(
            #     (self.num_steps, self.token_dim, self.num_tokens),
            #     dtype=torch.float64,
            # )

            self.P = torch.zeros(
                self.num_steps,
                self.num_tokens,
                self.num_tokens,
                dtype=torch.float64,
            )

            # Rescaled dynamics are our default, so we will only explicitly work with z
            self.Z = torch.zeros(
                self.num_steps,
                self.token_dim,
                self.num_tokens,
                dtype=torch.float64,
            )
            self.Z[0] = self.X_init
            # Initialize P(0)
            self.P[0] = softmax(
                (self.Q @ self.Z[0]).T @ (self.K @ self.Z[0]),
                dim=-1,
            )
            # initialize dZ
            self.dZ = torch.zeros(
                self.num_steps,
                self.token_dim,
                self.num_tokens,
                dtype=torch.float64,
            )
            # self.Z_ex =
            self.dZ[0] = self.V @ (
                (torch.matrix_exp(0 * V) @ self.Z[0]) @ self.P[0].T
                - (torch.matrix_exp(0 * V) @ self.Z[0])
            )

        def rescaled_dynamics(self):
            for step in range(1, self.num_steps):
                self.Z[step] = self.Z[step - 1] + self.dZ[step - 1] * self.dt

                self.P[step] = softmax(
                    (
                        self.Q
                        @ (
                            torch.matrix_exp(step * self.dt * self.V)
                            @ self.Z[step]
                        )
                    ).T
                    @ (
                        self.K
                        @ (
                            torch.matrix_exp(step * self.dt * self.V)
                            @ self.Z[step]
                        )
                    ),
                    dim=-1,
                )

                self.dZ[step] = self.V @ (
                    self.Z[step] @ self.P[step].T - self.Z[step]
                )

    return (Attention,)


@app.cell
def _(Attention, mo, np, num_tokens_2d_input, torch):
    import plotly.graph_objects as _go_2d

    _num_tokens_2d = int(num_tokens_2d_input.value)
    _generator_2d = torch.Generator().manual_seed(_num_tokens_2d)
    _initial_tokens_2d = (
        torch.rand(
            (2, _num_tokens_2d),
            generator=_generator_2d,
            dtype=torch.float64,
        )
        * 2
        - 1
    )

    V = torch.rand((2, 2), dtype=torch.float64) * 2 - 1
    V = V.T @ V

    Q = torch.rand((2, 2), dtype=torch.float64) * 2 - 1
    Q = Q.T @ Q

    Attn2d = Attention(
        K=torch.eye(2, dtype=torch.float64),
        Q=Q + torch.eye(2),
        V=V,
        X=_initial_tokens_2d,
        T=100,
    )
    Attn2d.rescaled_dynamics()

    _finite_frames_2d = torch.isfinite(Attn2d.P).all(
        dim=(1, 2)
    ) & torch.isfinite(Attn2d.Z).all(dim=(1, 2))
    _nonfinite_steps_2d = torch.where(~_finite_frames_2d)[0]
    _num_frames_2d = (
        int(_nonfinite_steps_2d[0])
        if _nonfinite_steps_2d.numel()
        else Attn2d.num_steps
    )
    _max_visualization_frames_2d = 401
    _frame_indices_2d = np.unique(
        np.linspace(
            0,
            _num_frames_2d - 1,
            min(_num_frames_2d, _max_visualization_frames_2d),
            dtype=int,
        )
    )
    _times_2d = _frame_indices_2d * Attn2d.dt
    _last_frame_2d = len(_frame_indices_2d) - 1
    _edge_scale_2d = 8.0
    _edge_threshold_2d = 1e-4
    _edge_bins_2d = 12

    def _connection_frame_2d(_index):
        _tokens = Attn2d.Z[_index]
        _keys = (
            (Attn2d.K @ _tokens)
            .detach()
            .cpu()
            .numpy()
            .T.astype(np.float32, copy=False)
        )
        _queries = (
            (Attn2d.Q @ _tokens)
            .detach()
            .cpu()
            .numpy()
            .T.astype(np.float32, copy=False)
        )
        _attention = Attn2d.P[_index].detach().cpu().numpy()
        _edge_coordinates = [([], []) for _ in range(_edge_bins_2d)]

        for _query_index in range(Attn2d.num_tokens):
            for _key_index in range(Attn2d.num_tokens):
                _weight = float(_attention[_query_index, _key_index])
                if _weight <= _edge_threshold_2d:
                    continue
                _bin = min(int(_weight * _edge_bins_2d), _edge_bins_2d - 1)
                _x_values, _y_values = _edge_coordinates[_bin]
                _x_values.extend(
                    [_queries[_query_index, 0], _keys[_key_index, 0], np.nan]
                )
                _y_values.extend(
                    [_queries[_query_index, 1], _keys[_key_index, 1], np.nan]
                )

        _traces = []
        for _bin, (_x_values, _y_values) in enumerate(_edge_coordinates):
            _representative_weight = (_bin + 0.5) / _edge_bins_2d
            _traces.append(
                _go_2d.Scatter(
                    x=np.asarray(_x_values, dtype=np.float32),
                    y=np.asarray(_y_values, dtype=np.float32),
                    mode="lines",
                    line={
                        "color": "rgba(0, 0, 0, 0.45)",
                        "width": _edge_scale_2d * _representative_weight,
                    },
                    hoverinfo="skip",
                    visible=bool(_x_values),
                    showlegend=False,
                )
            )

        _traces.extend(
            [
                _go_2d.Scatter(
                    x=_keys[:, 0],
                    y=_keys[:, 1],
                    mode="markers",
                    name="Kx",
                    text=[
                        f"Key token {_i + 1}"
                        for _i in range(Attn2d.num_tokens)
                    ],
                    hovertemplate="%{text}<br>(%{x:.3f}, %{y:.3f})<extra></extra>",
                    marker={
                        "size": 10,
                        "color": "#a8deb5",
                        "line": {"color": "black", "width": 1},
                    },
                ),
                _go_2d.Scatter(
                    x=_queries[:, 0],
                    y=_queries[:, 1],
                    mode="markers",
                    name="Qx",
                    text=[
                        f"Query token {_i + 1}"
                        for _i in range(Attn2d.num_tokens)
                    ],
                    hovertemplate="%{text}<br>(%{x:.3f}, %{y:.3f})<extra></extra>",
                    marker={
                        "size": 10,
                        "color": "#d91c72",
                        "line": {"color": "black", "width": 1},
                    },
                ),
            ]
        )

        _points = np.vstack([_keys, _queries])
        _x_center = float((_points[:, 0].min() + _points[:, 0].max()) / 2)
        _y_center = float((_points[:, 1].min() + _points[:, 1].max()) / 2)
        _span = max(
            float(np.ptp(_points[:, 0])),
            float(np.ptp(_points[:, 1])),
            1e-6,
        )
        _half_span = 0.6 * _span
        return (
            _traces,
            [_x_center - _half_span, _x_center + _half_span],
            [_y_center - _half_span, _y_center + _half_span],
        )

    _frame_content_2d = [
        _connection_frame_2d(int(_simulation_index_2d))
        for _simulation_index_2d in _frame_indices_2d
    ]
    _trace_indices_2d = list(range(len(_frame_content_2d[0][0])))
    _frames_2d = [
        _go_2d.Frame(
            name=str(_index_2d),
            data=_frame_content_2d[_index_2d][0],
            traces=_trace_indices_2d,
            layout=_go_2d.Layout(
                title_text=f"2D Key/Query Attention at t={_times_2d[_index_2d]:.2f}",
                xaxis={"range": _frame_content_2d[_index_2d][1]},
                yaxis={"range": _frame_content_2d[_index_2d][2]},
            ),
        )
        for _index_2d in range(len(_frame_indices_2d))
    ]
    _slider_steps_2d = [
        {
            "method": "animate",
            "args": [
                [str(_index_2d)],
                {
                    "mode": "immediate",
                    "frame": {"duration": 0, "redraw": True},
                    "transition": {"duration": 0},
                },
            ],
            "label": f"{_time_2d:.2f}",
        }
        for _index_2d, _time_2d in enumerate(_times_2d)
    ]
    _final_traces_2d, _final_x_range_2d, _final_y_range_2d = _frame_content_2d[
        _last_frame_2d
    ]
    _two_d_attention_figure = _go_2d.Figure(
        data=_final_traces_2d,
        frames=_frames_2d,
    )
    _two_d_attention_figure.update_layout(
        title=f"2D Key/Query Attention at t={_times_2d[_last_frame_2d]:.2f}",
        height=800,
        margin={"l": 55, "r": 30, "t": 90, "b": 110},
        plot_bgcolor="white",
        xaxis={
            "range": _final_x_range_2d,
            "showticklabels": False,
            "ticks": "",
            "showgrid": False,
            "zeroline": False,
            "constrain": "domain",
        },
        yaxis={
            "range": _final_y_range_2d,
            "showticklabels": False,
            "ticks": "",
            "showgrid": False,
            "zeroline": False,
            "constrain": "domain",
            "scaleanchor": "x",
            "scaleratio": 1,
        },
        legend={"orientation": "h", "x": 0.5, "xanchor": "center", "y": 1.04},
        sliders=[
            {
                "active": _last_frame_2d,
                "currentvalue": {"prefix": "Time: "},
                "pad": {"t": 45},
                "steps": _slider_steps_2d,
            }
        ],
    )
    two_d_attention_plot = mo.ui.plotly(
        _two_d_attention_figure,
        config={"responsive": True, "displaylogo": False},
    )
    mo.vstack([num_tokens_2d_input, two_d_attention_plot], gap=1.0)
    return (Attn2d,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    There are som really interesting phenomenon happening above.
    1. The phenomenon demonstrated in the paper doesn't always happen. Sometimes the $Kx_i$ and $Qx_i$ don't for two seperate clusters, but lie on affine spaces, often intersecting. I wonder why that is.
    """)
    return


@app.cell
def _(torch):
    z = torch.rand(3, 10)
    z[:, 9] = torch.tensor([-10, -10, -10]) + torch.rand(3)
    Vz = z - z.T.unsqueeze(-1)
    Vz[-1]
    return


@app.cell
def _(torch):
    a = torch.tensor([1, 2, 3, 4, 5])
    print(a.ndim, a.shape)
    print(torch.eye(2))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Section 2
    We will illustrate some of the findings of the clusters paper (GLPR 23), starting with THM 2.1. THM 2.1 states that for Q, K, and V scalars in 1-d, the attention matrix converges exponentially to a low-rank matrix.
    """)
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(Attn_1D, k_input, mo, np, num_tokens_input, q_input, torch, v_input):
    import plotly.graph_objects as _go_sim

    _num_tokens_sim = int(num_tokens_input.value)
    _generator_sim = torch.Generator().manual_seed(_num_tokens_sim)
    _initial_tokens_sim = torch.randn(
        _num_tokens_sim,
        generator=_generator_sim,
        dtype=torch.float64,
    )

    A1 = Attn_1D(
        K=float(k_input.value),
        Q=float(q_input.value),
        V=float(v_input.value),
        X=_initial_tokens_sim,
    )
    A1.step_dynamics()

    _token_labels_sim = np.arange(1, A1.num_tokens + 1)
    _times_sim = np.arange(A1.num_steps) * A1.dt
    _last_frame_sim = A1.num_steps - 1
    _title_suffix_sim = (
        f"Q={A1.Q.item():g}, K={A1.K.item():g}, V={A1.V.item():g}"
    )
    _frames_sim = [
        _go_sim.Frame(
            name=str(_index),
            data=[
                _go_sim.Heatmap(
                    z=A1.P[_index].detach().cpu().numpy(),
                    x=_token_labels_sim,
                    y=_token_labels_sim,
                    zmin=0,
                    zmax=1,
                    colorscale="Viridis",
                )
            ],
            traces=[0],
            layout=_go_sim.Layout(
                title_text=(
                    f"Attention Matrix at t={_times_sim[_index]:.2f} "
                    f"({_title_suffix_sim})"
                )
            ),
        )
        for _index in range(A1.num_steps)
    ]
    _slider_steps_sim = [
        {
            "method": "animate",
            "args": [
                [str(_index)],
                {
                    "mode": "immediate",
                    "frame": {"duration": 0, "redraw": True},
                    "transition": {"duration": 0},
                },
            ],
            "label": f"{_time:.2f}",
        }
        for _index, _time in enumerate(_times_sim)
    ]
    _attention_plot_figure_sim = _go_sim.Figure(
        data=[
            _go_sim.Heatmap(
                z=A1.P[_last_frame_sim].detach().cpu().numpy(),
                x=_token_labels_sim,
                y=_token_labels_sim,
                zmin=0,
                zmax=1,
                colorscale="Viridis",
                colorbar={"title": "Attention weight"},
                hovertemplate=(
                    "Query token %{y}<br>Key token %{x}<br>"
                    "Attention %{z:.4f}<extra></extra>"
                ),
            )
        ],
        frames=_frames_sim,
    )
    _attention_plot_figure_sim.update_layout(
        title=(
            f"Attention Matrix at t={_times_sim[_last_frame_sim]:.2f} "
            f"({_title_suffix_sim})"
        ),
        height=800,
        margin={"l": 80, "r": 80, "t": 90, "b": 110},
        xaxis={
            "title": "Key token",
            "dtick": max(1, A1.num_tokens // 20),
            "range": [0.5, A1.num_tokens + 0.5],
            "constrain": "domain",
        },
        yaxis={
            "title": "Query token",
            "dtick": max(1, A1.num_tokens // 20),
            "range": [A1.num_tokens + 0.5, 0.5],
            "constrain": "domain",
            "scaleanchor": "x",
            "scaleratio": 1,
        },
        sliders=[
            {
                "active": _last_frame_sim,
                "currentvalue": {"prefix": "Time: "},
                "pad": {"t": 45},
                "steps": _slider_steps_sim,
            }
        ],
    )
    attention_plot = mo.ui.plotly(
        _attention_plot_figure_sim,
        config={"responsive": True, "displaylogo": False},
    )
    return (attention_plot,)


@app.cell
def _(Attn2d, mo, np, torch):
    import plotly.graph_objects as _go_2d_matrix

    _finite_matrix_frames = torch.isfinite(Attn2d.P).all(dim=(1, 2))
    _nonfinite_matrix_steps = torch.where(~_finite_matrix_frames)[0]
    _num_matrix_frames = (
        int(_nonfinite_matrix_steps[0])
        if _nonfinite_matrix_steps.numel()
        else Attn2d.num_steps
    )
    _max_matrix_frames = 401
    _matrix_frame_indices = np.unique(
        np.linspace(
            0,
            _num_matrix_frames - 1,
            min(_num_matrix_frames, _max_matrix_frames),
            dtype=int,
        )
    )
    _matrix_times = _matrix_frame_indices * Attn2d.dt
    _matrix_last_frame = len(_matrix_frame_indices) - 1
    _matrix_token_labels = np.arange(
        1,
        Attn2d.num_tokens + 1,
        dtype=np.int16,
    )
    _matrix_frames = [
        _go_2d_matrix.Frame(
            name=str(_frame_position),
            data=[
                _go_2d_matrix.Heatmap(
                    z=Attn2d.P[_simulation_index]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32, copy=False),
                    x=_matrix_token_labels,
                    y=_matrix_token_labels,
                    zmin=0,
                    zmax=1,
                    colorscale="Viridis",
                )
            ],
            traces=[0],
            layout=_go_2d_matrix.Layout(
                title_text=(
                    f"2D Attention Matrix at t={_matrix_times[_frame_position]:.2f}"
                )
            ),
        )
        for _frame_position, _simulation_index in enumerate(
            _matrix_frame_indices
        )
    ]
    _matrix_slider_steps = [
        {
            "method": "animate",
            "args": [
                [str(_index)],
                {
                    "mode": "immediate",
                    "frame": {"duration": 0, "redraw": True},
                    "transition": {"duration": 0},
                },
            ],
            "label": f"{_time:.2f}",
        }
        for _index, _time in enumerate(_matrix_times)
    ]
    _matrix_final_simulation_index = _matrix_frame_indices[_matrix_last_frame]
    _two_d_matrix_figure = _go_2d_matrix.Figure(
        data=[
            _go_2d_matrix.Heatmap(
                z=Attn2d.P[_matrix_final_simulation_index]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32, copy=False),
                x=_matrix_token_labels,
                y=_matrix_token_labels,
                zmin=0,
                zmax=1,
                colorscale="Viridis",
                colorbar={"title": "Attention weight"},
                hovertemplate=(
                    "Query token %{y}<br>Key token %{x}<br>"
                    "Attention %{z:.4f}<extra></extra>"
                ),
            )
        ],
        frames=_matrix_frames,
    )
    _two_d_matrix_figure.update_layout(
        title=f"2D Attention Matrix at t={_matrix_times[_matrix_last_frame]:.2f}",
        height=800,
        margin={"l": 80, "r": 80, "t": 90, "b": 110},
        xaxis={
            "title": "Key token",
            "dtick": max(1, Attn2d.num_tokens // 20),
            "range": [0.5, Attn2d.num_tokens + 0.5],
            "constrain": "domain",
        },
        yaxis={
            "title": "Query token",
            "dtick": max(1, Attn2d.num_tokens // 20),
            "range": [Attn2d.num_tokens + 0.5, 0.5],
            "constrain": "domain",
            "scaleanchor": "x",
            "scaleratio": 1,
        },
        sliders=[
            {
                "active": _matrix_last_frame,
                "currentvalue": {"prefix": "Time: "},
                "pad": {"t": 45},
                "steps": _matrix_slider_steps,
            }
        ],
    )
    two_d_matrix_plot = mo.ui.plotly(
        _two_d_matrix_figure,
        config={"responsive": True, "displaylogo": False},
    )
    two_d_matrix_plot
    return


@app.cell(hide_code=True)
def _(mo):
    num_tokens_2d_input = mo.ui.number(
        start=2,
        stop=40,
        step=1,
        value=20,
        debounce=True,
        label="2D tokens",
    )
    return (num_tokens_2d_input,)


if __name__ == "__main__":
    app.run()
