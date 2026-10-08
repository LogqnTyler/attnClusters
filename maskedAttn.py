import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell
def _():
    return


@app.cell
def _():
    import marimo as mo
    from attention import Attention
    import torch
    from torch.nn.functional import softmax
    from torch.linalg import matrix_rank
    import numpy as np
    from sklearn.datasets import make_spd_matrix

    device = torch.device("cuda")
    torch.set_default_dtype(torch.float64)
    torch.set_default_device(device)
    return (
        Attention,
        device,
        make_spd_matrix,
        matrix_rank,
        mo,
        np,
        softmax,
        torch,
    )


@app.cell
def _(Attention, device, np, softmax, torch):
    class MaskedAttention(Attention):
        def __init__(
            self,
            K: np.ndarray | torch.Tensor,
            Q: np.ndarray | torch.Tensor,
            V: np.ndarray | torch.Tensor,
            X: np.ndarray | torch.Tensor,
            T: float = 10,
            dt: float = 0.05,
            _store_history: bool = True,
        ):
            K = torch.as_tensor(K, dtype=torch.float64, device=device)
            Q = torch.as_tensor(Q, dtype=torch.float64, device=device)
            V = torch.as_tensor(V, dtype=torch.float64, device=device)
            X = torch.as_tensor(X, dtype=torch.float64, device=device)
            super().__init__(K, Q, V, X, T, dt, _store_history)
            n = self.num_tokens

            self.mask = torch.tril(
                torch.full(
                    (n, n),
                    -torch.inf,
                    dtype=torch.float64,
                    device=device,
                ),
                diagonal=-1,
            ).T
            self.P[0] = softmax(
                (self.Q @ self.Z[0]).T @ (self.K @ self.Z[0]) + self.mask,
                dim=-1,
            )
            self.dZ[0] = self.V @ (self.Z[0] @ self.P[0].T - self.Z[0])

        def rescaled_dynamics(self) -> None:
            if not self._store_history:
                self._rescaled_dynamics_forgetful()
                return

            for step in range(1, self.num_steps):
                self.Z[step] = self.Z[step - 1] + self.dZ[step - 1] * self.dt
                exp_v = torch.matrix_exp(step * self.dt * self.V)
                x = exp_v @ self.Z[step]
                self.P[step] = softmax(
                    (self.Q @ x).T @ (self.K @ x) + self.mask,
                    dim=-1,
                )
                self.dZ[step] = self.V @ (
                    self.Z[step] @ self.P[step].T - self.Z[step]
                )

    return (MaskedAttention,)


@app.cell
def _(mo):
    num_tokens_masked_input = mo.ui.number(
        start=2,
        stop=100,
        step=1,
        value=10,
        debounce=True,
        label="Tokens",
    )
    num_tokens_masked_input
    return (num_tokens_masked_input,)


@app.cell
def _(device, make_spd_matrix, torch):
    def sampleQK(D):
        while True:
            Q = torch.rand((D, D), dtype=torch.float64, device=device) * 2 - 1
            if torch.linalg.cond(Q) < 1e4:
                break
        while True:
            M = torch.as_tensor(
                make_spd_matrix(D), dtype=torch.float64, device=device
            )
            if torch.linalg.cond(M) < 1e4:
                break
        return Q, torch.linalg.solve(Q.T, M)

    return (sampleQK,)


@app.cell
def _(
    Attention,
    MaskedAttention,
    device,
    num_tokens_masked_input,
    sampleQK,
    torch,
):
    _masked_num_tokens = int(num_tokens_masked_input.value)
    _masked_identity_matrix = torch.eye(3, dtype=torch.float64, device=device)
    masked_initial_tokens = (
        torch.rand((3, _masked_num_tokens), dtype=torch.float64, device=device)
        * 10
        - 5
    )
    masked_identity_attn = MaskedAttention(
        K=_masked_identity_matrix,
        Q=_masked_identity_matrix,
        V=_masked_identity_matrix,
        X=masked_initial_tokens.clone(),
        T=10,
        dt=0.05,
    )
    unmasked_identity_attn = Attention(
        K=_masked_identity_matrix,
        Q=_masked_identity_matrix,
        V=_masked_identity_matrix,
        X=masked_initial_tokens.clone(),
        T=10,
        dt=0.05,
    )

    Q, K = sampleQK(3)
    masked_psd_attn = MaskedAttention(
        K=K,
        Q=Q,
        V=_masked_identity_matrix,
        X=masked_initial_tokens.clone(),
        T=10,
        dt=0.05,
    )
    unmasked_psd_attn = Attention(
        K=K,
        Q=Q,
        V=_masked_identity_matrix,
        X=masked_initial_tokens.clone(),
        T=10,
        dt=0.05,
    )
    _masked_random_K = (
        torch.rand((3, 3), dtype=torch.float64, device=device) * 2 - 1
    )
    _masked_random_Q = (
        torch.rand((3, 3), dtype=torch.float64, device=device) * 2 - 1
    )
    maskedAttn = MaskedAttention(
        K=_masked_random_K,
        Q=_masked_random_Q,
        V=_masked_identity_matrix,
        X=masked_initial_tokens.clone(),
        T=10,
        dt=0.05,
    )
    unmasked_random_attn = Attention(
        K=_masked_random_K,
        Q=_masked_random_Q,
        V=_masked_identity_matrix,
        X=masked_initial_tokens.clone(),
        T=10,
        dt=0.05,
    )
    return (
        maskedAttn,
        masked_identity_attn,
        masked_psd_attn,
        unmasked_identity_attn,
        unmasked_psd_attn,
        unmasked_random_attn,
    )


@app.cell
def _(maskedAttn):
    maskedAttn.rescaled_dynamics()
    return


@app.cell
def _(
    masked_identity_attn,
    masked_psd_attn,
    unmasked_identity_attn,
    unmasked_psd_attn,
    unmasked_random_attn,
):
    masked_identity_attn.rescaled_dynamics()
    masked_psd_attn.rescaled_dynamics()
    unmasked_identity_attn.rescaled_dynamics()
    unmasked_psd_attn.rescaled_dynamics()
    unmasked_random_attn.rescaled_dynamics()
    unmasked_comparison_ready = True
    masked_comparison_ready = True
    return masked_comparison_ready, unmasked_comparison_ready


@app.cell
def _(maskedAttn, matrix_rank, torch):
    matrix_rank(maskedAttn.P[-1]).to(dtype=torch.float64)
    return


@app.cell
def _(maskedAttn, mo, np, torch):
    import plotly.graph_objects as _go_masked_p

    _masked_p_finite_frames = torch.isfinite(maskedAttn.P).all(dim=(1, 2))
    _masked_p_invalid_steps = torch.where(~_masked_p_finite_frames)[0]
    _masked_p_valid_frames = (
        int(_masked_p_invalid_steps[0])
        if _masked_p_invalid_steps.numel()
        else maskedAttn.num_steps
    )
    _masked_p_frame_indices = np.unique(
        np.linspace(
            0,
            _masked_p_valid_frames - 1,
            min(_masked_p_valid_frames, 401),
            dtype=int,
        )
    )
    _masked_p_times = _masked_p_frame_indices * maskedAttn.dt
    _masked_p_last_frame = len(_masked_p_frame_indices) - 1
    _masked_p_token_labels = np.arange(1, maskedAttn.num_tokens + 1)
    _masked_p_frames = [
        _go_masked_p.Frame(
            name=str(_frame_position),
            data=[
                _go_masked_p.Heatmap(
                    z=maskedAttn.P[_simulation_index].detach().cpu().numpy(),
                    x=_masked_p_token_labels,
                    y=_masked_p_token_labels,
                    zmin=0,
                    zmax=1,
                    colorscale="Viridis",
                )
            ],
            traces=[0],
            layout=_go_masked_p.Layout(
                title_text=(
                    f"Masked Attention Matrix at t={_masked_p_times[_frame_position]:.2f}"
                )
            ),
        )
        for _frame_position, _simulation_index in enumerate(
            _masked_p_frame_indices
        )
    ]
    _masked_p_slider_steps = [
        {
            "method": "animate",
            "args": [
                [str(_frame_position)],
                {
                    "mode": "immediate",
                    "frame": {"duration": 0, "redraw": True},
                    "transition": {"duration": 0},
                },
            ],
            "label": f"{_time:.2f}",
        }
        for _frame_position, _time in enumerate(_masked_p_times)
    ]
    _masked_p_figure = _go_masked_p.Figure(
        data=[
            _go_masked_p.Heatmap(
                z=maskedAttn.P[_masked_p_frame_indices[_masked_p_last_frame]]
                .detach()
                .cpu()
                .numpy(),
                x=_masked_p_token_labels,
                y=_masked_p_token_labels,
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
        frames=_masked_p_frames,
    )
    _masked_p_figure.update_layout(
        title=(
            f"Masked Attention Matrix at t={_masked_p_times[_masked_p_last_frame]:.2f}"
        ),
        height=800,
        margin={"l": 80, "r": 80, "t": 90, "b": 110},
        xaxis={
            "title": "Key token",
            "dtick": max(1, maskedAttn.num_tokens // 20),
            "range": [0.5, maskedAttn.num_tokens + 0.5],
            "constrain": "domain",
        },
        yaxis={
            "title": "Query token",
            "dtick": max(1, maskedAttn.num_tokens // 20),
            "range": [maskedAttn.num_tokens + 0.5, 0.5],
            "constrain": "domain",
            "scaleanchor": "x",
            "scaleratio": 1,
        },
        sliders=[
            {
                "active": _masked_p_last_frame,
                "currentvalue": {"prefix": "Time: "},
                "pad": {"t": 45},
                "steps": _masked_p_slider_steps,
            }
        ],
    )
    masked_p_plot = mo.ui.plotly(
        _masked_p_figure,
        config={"responsive": True, "displaylogo": False},
    )
    masked_p_plot
    return


@app.cell
def _(
    maskedAttn,
    masked_comparison_ready,
    masked_identity_attn,
    masked_psd_attn,
    mo,
    np,
    num_tokens_masked_input,
    unmasked_comparison_ready,
    unmasked_identity_attn,
    unmasked_psd_attn,
    unmasked_random_attn,
):
    import anywidget as _anywidget_masked_z
    import json as _json_masked_z
    import plotly.graph_objects as _go_masked_z
    import traitlets as _traitlets_masked_z
    from scipy.spatial import ConvexHull as _ConvexHull_masked_z
    from scipy.spatial import QhullError as _QhullError_masked_z

    class _PersistentPlotly3DMasked(_anywidget_masked_z.AnyWidget):
        figure = _traitlets_masked_z.Dict().tag(sync=True)
        times = _traitlets_masked_z.List(_traitlets_masked_z.Float()).tag(
            sync=True
        )

        _esm = r"""
    import Plotly from "https://esm.sh/plotly.js-dist-min@4.1.1";

    function clone(value) {
      return value == null ? null : structuredClone(value);
    }

    function cameraFromEvent(current, update) {
      if (update?.["scene.camera"]) return clone(update["scene.camera"]);
      if (update?.scene?.camera) return clone(update.scene.camera);

      const next = clone(current) ?? {};
      let changed = false;
      for (const [key, value] of Object.entries(update ?? {})) {
        if (!key.startsWith("scene.camera.")) continue;
        const path = key.slice("scene.camera.".length).split(".");
        let target = next;
        for (const part of path.slice(0, -1)) {
          target[part] ??= {};
          target = target[part];
        }
        target[path.at(-1)] = value;
        changed = true;
      }
      return changed ? next : current;
    }

    async function render({ model, el }) {
      const figure = model.get("figure");
      const times = model.get("times");
      const graph = document.createElement("div");
      const controls = document.createElement("div");
      const slider = document.createElement("input");
      const timeLabel = document.createElement("span");
      const controller = new AbortController();

      el.style.cssText = "display:block;width:100%;";
      graph.style.cssText = "width:100%;height:760px;";
      controls.style.cssText = "display:flex;align-items:center;gap:12px;padding:4px 24px 14px;";
      slider.type = "range";
      slider.min = "0";
      slider.max = String(Math.max(0, times.length - 1));
      slider.step = "1";
      slider.value = "0";
      slider.style.cssText = "flex:1;accent-color:#7c9cff;cursor:pointer;";
      timeLabel.style.cssText = "min-width:92px;font:12px ui-monospace,monospace;color:#666;";
      timeLabel.textContent = `Time: ${(times[0] ?? 0).toFixed(2)}`;
      controls.append(slider, timeLabel);
      el.append(graph, controls);

      await Plotly.newPlot(graph, figure.data, figure.layout, {
        responsive: true,
        displaylogo: false,
      });

      let camera = clone(graph._fullLayout?.scene?.camera);
      let currentIndex = 0;
      let requestedIndex = 0;
      let applying = false;

      graph.on("plotly_relayout", (update) => {
        if (!applying) camera = cameraFromEvent(camera, update);
      });

      async function applyRequestedFrame() {
        if (applying) return;
        applying = true;
        try {
          while (currentIndex !== requestedIndex) {
            const targetIndex = requestedIndex;
            const savedCamera = clone(camera);
            const traceIndices = [
              currentIndex * 3,
              currentIndex * 3 + 1,
              currentIndex * 3 + 2,
              targetIndex * 3,
              targetIndex * 3 + 1,
              targetIndex * 3 + 2,
            ];
            await Plotly.restyle(
              graph,
              { visible: [false, false, false, true, true, true] },
              traceIndices,
            );
            if (savedCamera) {
              await Plotly.relayout(graph, { "scene.camera": savedCamera });
              camera = savedCamera;
            }
            currentIndex = targetIndex;
          }
        } finally {
          applying = false;
          if (currentIndex !== requestedIndex) applyRequestedFrame();
        }
      }

      slider.addEventListener(
        "input",
        () => {
          requestedIndex = Number(slider.value);
          timeLabel.textContent = `Time: ${times[requestedIndex].toFixed(2)}`;
          applyRequestedFrame();
        },
        { signal: controller.signal },
      );

      return () => {
        controller.abort();
        Plotly.purge(graph);
      };
    }

    export default { render };
    """

    def _make_masked_z_plot(_attention, _title, _camera_revision):
        _z = _attention.Z.detach().cpu().numpy()
        _finite_steps = np.isfinite(_z).all(axis=(1, 2))
        _valid_count = (
            int(np.argmax(~_finite_steps))
            if not _finite_steps.all()
            else _attention.num_steps
        )
        _frame_indices = np.unique(
            np.linspace(0, _valid_count - 1, min(_valid_count, 201), dtype=int)
        )
        _times = _frame_indices * _attention.dt
        _token_ids = np.arange(1, _attention.num_tokens + 1)
        _values = _z[_frame_indices]
        _ranges = []
        for _dimension in range(3):
            _minimum = float(_values[:, _dimension, :].min())
            _maximum = float(_values[:, _dimension, :].max())
            _padding = max(0.05 * (_maximum - _minimum), 1e-3)
            _ranges.append([_minimum - _padding, _maximum + _padding])

        def _hull_traces(_step):
            _points = _z[_step].T
            _empty_mesh = _go_masked_z.Mesh3d(
                x=[], y=[], z=[], hoverinfo="skip", showlegend=False
            )
            _empty_edges = _go_masked_z.Scatter3d(
                x=[],
                y=[],
                z=[],
                mode="lines",
                hoverinfo="skip",
                showlegend=False,
            )
            if len(_points) < 4:
                return _empty_mesh, _empty_edges
            try:
                _hull = _ConvexHull_masked_z(_points, qhull_options="QJ")
            except _QhullError_masked_z:
                return _empty_mesh, _empty_edges

            _i, _j, _k = _hull.simplices.T
            _mesh = _go_masked_z.Mesh3d(
                x=_points[:, 0],
                y=_points[:, 1],
                z=_points[:, 2],
                i=_i,
                j=_j,
                k=_k,
                color="#7c9cff",
                opacity=0.20,
                flatshading=True,
                lighting={
                    "ambient": 0.55,
                    "diffuse": 0.85,
                    "specular": 0.15,
                    "roughness": 0.8,
                    "fresnel": 0.1,
                },
                lightposition={"x": 100, "y": 200, "z": 300},
                hoverinfo="skip",
                showlegend=False,
            )
            _edges = sorted(
                {
                    tuple(sorted((_face[_a], _face[_b])))
                    for _face in _hull.simplices
                    for _a, _b in ((0, 1), (1, 2), (2, 0))
                }
            )
            _edge_x, _edge_y, _edge_z = [], [], []
            for _start, _end in _edges:
                _edge_x.extend([_points[_start, 0], _points[_end, 0], None])
                _edge_y.extend([_points[_start, 1], _points[_end, 1], None])
                _edge_z.extend([_points[_start, 2], _points[_end, 2], None])
            _edge_trace = _go_masked_z.Scatter3d(
                x=_edge_x,
                y=_edge_y,
                z=_edge_z,
                mode="lines",
                line={"color": "rgba(255, 166, 64, 0.42)", "width": 2},
                hoverinfo="skip",
                showlegend=False,
            )
            return _mesh, _edge_trace

        def _scatter(_step):
            _points = _z[_step]
            return _go_masked_z.Scatter3d(
                x=_points[0],
                y=_points[1],
                z=_points[2],
                mode="markers",
                text=[f"Token {_token_id}" for _token_id in _token_ids],
                marker={
                    "size": 7,
                    "color": _token_ids,
                    "colorscale": "Turbo",
                    "cmin": 1,
                    "cmax": _attention.num_tokens,
                    "line": {"color": "white", "width": 0.5},
                },
                hovertemplate=(
                    "%{text}<br>Z1=%{x:.4f}<br>Z2=%{y:.4f}<br>"
                    "Z3=%{z:.4f}<extra></extra>"
                ),
                showlegend=False,
            )

        _plot_traces = []
        for _position, _step in enumerate(_frame_indices):
            _step_traces = [*_hull_traces(int(_step)), _scatter(int(_step))]
            for _trace in _step_traces:
                _trace.visible = _position == 0
            _plot_traces.extend(_step_traces)

        _figure = _go_masked_z.Figure(data=_plot_traces)
        _figure.update_layout(
            title=_title,
            uirevision=_camera_revision,
            height=760,
            margin={"l": 0, "r": 0, "t": 80, "b": 100},
            scene={
                "uirevision": _camera_revision,
                "xaxis": {"title": "Z1", "range": _ranges[0]},
                "yaxis": {"title": "Z2", "range": _ranges[1]},
                "zaxis": {"title": "Z3", "range": _ranges[2]},
                "aspectmode": "cube",
            },
        )
        return _PersistentPlotly3DMasked(
            figure=_json_masked_z.loads(_figure.to_json()),
            times=_times.tolist(),
        )

    masked_identity_z_plot = _make_masked_z_plot(
        masked_identity_attn,
        "Q = K = I3: Rescaled Token Dynamics",
        "masked-identity-z-camera",
    )
    masked_psd_z_plot = _make_masked_z_plot(
        masked_psd_attn,
        "Q random; K = (Q^T)^-1 M, M SPD: Rescaled Token Dynamics",
        "masked-psd-z-camera",
    )
    masked_z_plot = _make_masked_z_plot(
        maskedAttn,
        "Q, K independently random: Rescaled Token Dynamics",
        "masked-random-z-camera",
    )
    unmasked_identity_z_plot = _make_masked_z_plot(
        unmasked_identity_attn,
        "Unmasked: Q = K = I3: Rescaled Token Dynamics",
        "unmasked-identity-z-camera",
    )
    unmasked_psd_z_plot = _make_masked_z_plot(
        unmasked_psd_attn,
        "Unmasked: Q random; K = (Q^T)^-1 M, M SPD: Rescaled Token Dynamics",
        "unmasked-psd-z-camera",
    )
    unmasked_random_z_plot = _make_masked_z_plot(
        unmasked_random_attn,
        "Unmasked: Q, K independently random: Rescaled Token Dynamics",
        "unmasked-random-z-camera",
    )
    if not masked_comparison_ready or not unmasked_comparison_ready:
        raise RuntimeError("Comparison dynamics have not run")

    mo.vstack(
        [
            num_tokens_masked_input,
            mo.md("## Masked attention"),
            mo.md("### Q = K = I3"),
            masked_identity_z_plot,
            mo.md("### Q random; K = (Q^T)^-1 M, M SPD"),
            masked_psd_z_plot,
            mo.md("### Q, K independently random"),
            masked_z_plot,
            mo.md("## Unmasked controls"),
            mo.md("### Q = K = I3"),
            unmasked_identity_z_plot,
            mo.md("### Q random; K = (Q^T)^-1 M, M SPD"),
            unmasked_psd_z_plot,
            mo.md("### Q, K independently random"),
            unmasked_random_z_plot,
        ],
        gap=1.0,
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
