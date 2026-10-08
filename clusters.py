import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # The Emergence of Clusters in Self-Attention Dynamics
    ## Comments, Visualizations, and Notes
    This notebook is a set of visualizations and explanations for the paper [the Emergence of Clusters in Self-Attention Dynamics](https://arxiv.org/abs/2305.05465) by Geshkovski et. all. The paper proves that under fixed (not time-dependent) weights, tokens cluster toward certain limiting shapes as time tends to infinity.
    """)
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import numpy as np
    import plotly.express as px
    import torch
    from attention import Attention, Attn_1D, ForgetfulAttention, device
    from torch.linalg import matrix_rank

    return (
        Attention,
        Attn_1D,
        ForgetfulAttention,
        device,
        matrix_rank,
        mo,
        np,
        torch,
    )


@app.cell(hide_code=True)
def full_height_outputs(mo):
    mo.Html(r"""
    <style>
    .console-output-area {
        max-height: none !important;
        overflow: visible !important;
    }
    </style>
    """)
    return


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Section 2
    We will illustrate some of the findings of the clusters paper (GLPR 23), starting with THM 2.1. THM 2.1 states that for Q, K, and V scalars in 1-d, the attention matrix converges exponentially to a low-rank matrix.
    """)
    return


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
 
    """)
    return


@app.cell(hide_code=True)
def _(device, torch):
    z = torch.rand(3, 10, device=device)
    z[:, 9] = torch.tensor([-10, -10, -10], device=device) + torch.rand(
        3, device=device
    )
    Vz = z - z.T.unsqueeze(-1)
    Vz[-1]
    return


@app.cell(hide_code=True)
def _(
    Attn_1D,
    device,
    k_input,
    mo,
    np,
    num_tokens_input,
    q_input,
    torch,
    v_input,
):
    import plotly.graph_objects as _go_sim

    _num_tokens_sim = int(num_tokens_input.value)
    _generator_sim = torch.Generator(device=device).manual_seed(
        _num_tokens_sim
    )
    _initial_tokens_sim = torch.randn(
        _num_tokens_sim,
        generator=_generator_sim,
        dtype=torch.float64,
        device=device,
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


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
def _(Attention, device, mo, np, num_tokens_2d_input, torch):
    import plotly.graph_objects as _go_2d

    _num_tokens_2d = int(num_tokens_2d_input.value)
    _generator_2d = torch.Generator(device=device).manual_seed(_num_tokens_2d)
    _initial_tokens_2d = (
        torch.rand(
            (2, _num_tokens_2d),
            generator=_generator_2d,
            dtype=torch.float64,
            device=device,
        )
        * 2
        - 1
    )

    V = torch.rand((2, 2), dtype=torch.float64, device=device) * 2 - 1
    V = V.T @ V

    Q = torch.rand((2, 2), dtype=torch.float64, device=device) * 2 - 1
    Q = Q.T @ Q

    Attn2d = Attention(
        K=torch.eye(2, dtype=torch.float64, device=device),
        Q=Q + torch.eye(2, dtype=torch.float64, device=device),
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
    There are some really interesting phenomenon happening above.
    1. The phenomenon demonstrated in the paper doesn't always happen. Sometimes ($n = 24$) the $Kx_i$ and $Qx_i$ don't form two seperate clusters, but lie on parallel affine spaces, sometimes intersecting. I wonder why that is.
    2. The attention matrix doens't always stay low-rank. After the $Kx_i$ converge to almost the same point in space, their dot products with any given $Qx_i$ are almost identical, causing the attention matrix to loose its binary property. This phenomenon is visible also in the attention matrix visualization, where we can see that after a while, multiple columns appear to look the same. I wonder how the paper's proof could account for this artifact. This is also possibly a result of float imprecision.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Section 3: Clustering Toward Boundary of Convex Polytope

    Theorem 3.1: If $V = I_d, Q^T K \succ 0,$ then for any initial ${z_i(0)}_n \subset R$, there exists a convex polytope $\cal{K} \subset R$ such that ${z_i(t)}_n \rightarrow \partial \cal(K)$ as $t \rightarrow \infty.$
    """)
    return


@app.cell(hide_code=True)
def convex_token_control(mo):
    N = mo.ui.number(
        start=2,
        stop=100,
        step=1,
        value=80,
        debounce=True,
        label="N (tokens)",
    )
    return (N,)


@app.cell(hide_code=True)
def _(Attention, N, device, torch):
    A = torch.eye(3, dtype=torch.float64, device=device)
    X = (
        torch.rand((3, int(N.value)), dtype=torch.float64, device=device) * 10
        - 5
    )
    X_offset = X + 5

    convexClustering = Attention(
        K=A,
        Q=A.T,
        V=torch.eye(3, dtype=torch.float64, device=device),
        X=X,
        T=10,
    )
    OffsetConvCluster = Attention(
        K=A,
        Q=A.T,
        V=torch.eye(3, dtype=torch.float64, device=device),
        X=X_offset,
        T=10,
    )
    lambdaVConvCluster = Attention(
        K=A,
        Q=A.T,
        V=torch.eye(3, dtype=torch.float64, device=device) * 5,
        X=X,
        T=10,
    )

    convexClustering.rescaled_dynamics()
    OffsetConvCluster.rescaled_dynamics()
    lambdaVConvCluster.rescaled_dynamics()
    return OffsetConvCluster, convexClustering, lambdaVConvCluster


@app.cell(hide_code=True)
def _(N, convexClustering, mo, np):
    import anywidget as _anywidget
    import json as _json
    import plotly.graph_objects as _go_convex
    import traitlets as _traitlets
    from scipy.spatial import ConvexHull as _ConvexHull
    from scipy.spatial import QhullError as _QhullError

    class _PersistentPlotly3D(_anywidget.AnyWidget):
        figure = _traitlets.Dict().tag(sync=True)
        times = _traitlets.List(_traitlets.Float()).tag(sync=True)

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

    def make_convex_z_plot(_attention, _title, _camera_revision):
        _convex_z = _attention.Z.detach().cpu().numpy()
        _convex_finite_steps = np.isfinite(_convex_z).all(axis=(1, 2))
        _convex_valid_count = (
            int(np.argmax(~_convex_finite_steps))
            if not _convex_finite_steps.all()
            else _attention.num_steps
        )
        _convex_frame_indices = np.unique(
            np.linspace(
                0,
                _convex_valid_count - 1,
                min(_convex_valid_count, 201),
                dtype=int,
            )
        )
        _convex_times = _convex_frame_indices * _attention.dt
        _convex_token_ids = np.arange(1, _attention.num_tokens + 1)
        _convex_values = _convex_z[_convex_frame_indices]
        _convex_ranges = []
        for _convex_dimension in range(3):
            _convex_min = float(_convex_values[:, _convex_dimension, :].min())
            _convex_max = float(_convex_values[:, _convex_dimension, :].max())
            _convex_pad = max(0.05 * (_convex_max - _convex_min), 1e-3)
            _convex_ranges.append(
                [_convex_min - _convex_pad, _convex_max + _convex_pad]
            )

        def _convex_hull_traces(_convex_step):
            _convex_points = _convex_z[_convex_step].T
            _convex_empty_mesh = _go_convex.Mesh3d(
                x=[], y=[], z=[], hoverinfo="skip", showlegend=False
            )
            _convex_empty_edges = _go_convex.Scatter3d(
                x=[],
                y=[],
                z=[],
                mode="lines",
                hoverinfo="skip",
                showlegend=False,
            )
            if len(_convex_points) < 4:
                return _convex_empty_mesh, _convex_empty_edges
            try:
                _convex_hull = _ConvexHull(_convex_points, qhull_options="QJ")
            except _QhullError:
                return _convex_empty_mesh, _convex_empty_edges

            _convex_i, _convex_j, _convex_k = _convex_hull.simplices.T
            _convex_mesh = _go_convex.Mesh3d(
                x=_convex_points[:, 0],
                y=_convex_points[:, 1],
                z=_convex_points[:, 2],
                i=_convex_i,
                j=_convex_j,
                k=_convex_k,
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

            _convex_edges = sorted(
                {
                    tuple(
                        sorted(
                            (_convex_face[_convex_a], _convex_face[_convex_b])
                        )
                    )
                    for _convex_face in _convex_hull.simplices
                    for _convex_a, _convex_b in ((0, 1), (1, 2), (2, 0))
                }
            )
            _convex_edge_x, _convex_edge_y, _convex_edge_z = [], [], []
            for _convex_start, _convex_end in _convex_edges:
                _convex_edge_x.extend(
                    [
                        _convex_points[_convex_start, 0],
                        _convex_points[_convex_end, 0],
                        None,
                    ]
                )
                _convex_edge_y.extend(
                    [
                        _convex_points[_convex_start, 1],
                        _convex_points[_convex_end, 1],
                        None,
                    ]
                )
                _convex_edge_z.extend(
                    [
                        _convex_points[_convex_start, 2],
                        _convex_points[_convex_end, 2],
                        None,
                    ]
                )
            _convex_edge_trace = _go_convex.Scatter3d(
                x=_convex_edge_x,
                y=_convex_edge_y,
                z=_convex_edge_z,
                mode="lines",
                line={"color": "rgba(255, 166, 64, 0.42)", "width": 2},
                hoverinfo="skip",
                showlegend=False,
            )
            return _convex_mesh, _convex_edge_trace

        def _convex_scatter(_convex_step):
            _convex_points = _convex_z[_convex_step]
            return _go_convex.Scatter3d(
                x=_convex_points[0],
                y=_convex_points[1],
                z=_convex_points[2],
                mode="markers",
                text=[
                    f"Token {_convex_id}" for _convex_id in _convex_token_ids
                ],
                customdata=_convex_token_ids,
                marker={
                    "size": 7,
                    "color": _convex_token_ids,
                    "colorscale": "Turbo",
                    "cmin": 1,
                    "cmax": _attention.num_tokens,
                    "line": {"color": "white", "width": 0.5},
                },
                hovertemplate=(
                    "%{text}<br>Z₁=%{x:.4f}<br>Z₂=%{y:.4f}<br>Z₃=%{z:.4f}"
                    "<extra></extra>"
                ),
                showlegend=False,
            )

        _convex_plot_traces = []
        for _convex_position, _convex_step in enumerate(_convex_frame_indices):
            _convex_step_traces = [
                *_convex_hull_traces(int(_convex_step)),
                _convex_scatter(int(_convex_step)),
            ]
            _convex_is_visible = _convex_position == 0
            for _convex_trace in _convex_step_traces:
                _convex_trace.visible = _convex_is_visible
            _convex_plot_traces.extend(_convex_step_traces)

        _convex_figure = _go_convex.Figure(data=_convex_plot_traces)
        _convex_figure.update_layout(
            title=_title,
            uirevision=_camera_revision,
            height=760,
            margin={"l": 0, "r": 0, "t": 80, "b": 100},
            scene={
                "uirevision": _camera_revision,
                "xaxis": {"title": "Z₁", "range": _convex_ranges[0]},
                "yaxis": {"title": "Z₂", "range": _convex_ranges[1]},
                "zaxis": {"title": "Z₃", "range": _convex_ranges[2]},
                "aspectmode": "cube",
            },
        )

        return _PersistentPlotly3D(
            figure=_json.loads(_convex_figure.to_json()),
            times=_convex_times.tolist(),
        )

    convex_z_plot = make_convex_z_plot(
        convexClustering,
        "Centered Cluster Dynamics",
        "centered-convex-z-camera",
    )

    mo.vstack([N, convex_z_plot], gap=1.0)
    return (make_convex_z_plot,)


@app.cell(hide_code=True)
def _(OffsetConvCluster, make_convex_z_plot):
    offset_convex_z_plot = make_convex_z_plot(
        OffsetConvCluster,
        "Offset Cluster Dynamics",
        "offset-convex-z-camera",
    )
    offset_convex_z_plot
    return


@app.cell(hide_code=True)
def lambda_v_cluster_plot(lambdaVConvCluster, make_convex_z_plot):
    lambda_v_convex_z_plot = make_convex_z_plot(
        lambdaVConvCluster,
        "Lambda-V Cluster Dynamics",
        "lambda-v-convex-z-camera",
    )
    lambda_v_convex_z_plot
    return


@app.cell
def identity_rank_histograms(
    ForgetfulAttention,
    device,
    matrix_rank,
    np,
    torch,
):
    import matplotlib.pyplot as _plt_identity_rank

    _identity_token_counts = [5, 10, 20, 50, 100, 200, 500, 1000]
    _identity_rank_histories = []
    _identity_trials = 10
    _identity_matrix = torch.eye(3, dtype=torch.float64, device=device)

    for _identity_n in _identity_token_counts:
        _identity_ranks = torch.empty(
            _identity_trials, dtype=torch.int64, device=device
        )
        for _identity_trial in range(_identity_trials):
            _identity_cluster = ForgetfulAttention(
                K=_identity_matrix,
                Q=_identity_matrix,
                V=_identity_matrix,
                X=torch.rand(
                    (3, _identity_n), dtype=torch.float64, device=device
                )
                * 10
                - 5,
                T=10,
            )
            _identity_cluster.rescaled_dynamics()
            _identity_ranks[_identity_trial] = matrix_rank(_identity_cluster.P)
        _identity_rank_histories.append(_identity_ranks)

    _identity_rank_max = max(
        int(_identity_ranks.max())
        for _identity_ranks in _identity_rank_histories
    )
    _identity_rank_bins = np.arange(0.5, _identity_rank_max + 1.5, 1)
    _identity_tick_step = max(1, int(np.ceil(_identity_rank_max / 10)))
    identity_rank_histogram_figure, _identity_axes = (
        _plt_identity_rank.subplots(
            4,
            2,
            figsize=(12, 14),
            sharex=True,
            sharey=True,
        )
    )

    for _identity_axis, _identity_n, _identity_ranks in zip(
        _identity_axes.flat,
        _identity_token_counts,
        _identity_rank_histories,
    ):
        _identity_axis.hist(
            _identity_ranks.cpu().numpy(),
            bins=_identity_rank_bins,
            color="#4f9d69",
            edgecolor="white",
            linewidth=0.8,
            rwidth=0.9,
        )
        _identity_axis.set_title(f"N = {_identity_n}")
        _identity_axis.set_xlabel("Final attention rank")
        _identity_axis.set_ylabel("Frequency")
        _identity_axis.set_xticks(
            np.arange(1, _identity_rank_max + 1, _identity_tick_step)
        )
        _identity_axis.grid(axis="y", alpha=0.2)

    identity_rank_histogram_figure.suptitle(
        "Q = K = V = I₃: Distribution of Final Attention Rank",
        fontsize=15,
    )
    identity_rank_histogram_figure.tight_layout()
    identity_rank_histogram_figure
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This plot gives me some questions:
    - How can we determine the leaders from a starting set of ${z_i(0)}$?
    - What properties of our initial tokens determine what our leaders are? It is clear that the leaders are the tokens which do not change position significantly over the course of the dynamics (although they change a bit at the start), and they seem to be in some way the most "outward" of the tokens. I wonder if this can help us determine what the initial tokens will be. The offset (not zero centered) tokens clearly converge immediately to the most outward.
    - How does the value of $\lambda$ affect the speed of convergence if $V = \lambda I_d$? Just a rough eyeball of increasing $\lambda$ from $1$ to $2$ seems to yeild a 4x increase in the convergence speed.
    """)
    return


@app.cell(hide_code=True)
def non_psd_cluster_simulation(Attention, N, device, torch):
    B = torch.rand((3, 3), dtype=torch.float64, device=device) * 2 - 1

    convexClusteringPSD = Attention(
        K=B,
        Q=B.T,
        V=torch.eye(3, dtype=torch.float64, device=device),
        X=torch.rand((3, int(N.value)), dtype=torch.float64, device=device)
        * 10
        - 5,
        T=10,
    )
    convexClusteringPSD.rescaled_dynamics()
    return (convexClusteringPSD,)


@app.cell(hide_code=True)
def non_psd_cluster_plot(N, convexClusteringPSD, make_convex_z_plot, mo):
    psd_convex_z_plot = make_convex_z_plot(
        convexClusteringPSD,
        "PSD Q^T K Cluster Dynamics",
        "psd-convex-z-camera",
    )
    mo.vstack([N, psd_convex_z_plot], gap=1.0)
    return


@app.cell
def _(convexClusteringPSD, matrix_rank):
    matrix_rank(convexClusteringPSD.P[-1])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    As one can see, the attention matrix approaches a rank equal to the number of leaders/vetices of the polytope. We still frequently have degenerate polyhedra with only two vertices. Below is a plot of the distribution of the number of vertices at $n = 5, 10, 50, 100$. Interestingly, rank = 2 is far and away the most common result of the dynamics, and $\Pr\!\left(\operatorname{rank}(P [-1]) = 2\right) \approx 0.55$. I wonder why this is, and if there is a way to formalize these results.
    """)
    return


@app.cell(hide_code=True)
def rank_histograms(ForgetfulAttention, device, matrix_rank, np, torch):
    import matplotlib.pyplot as _plt_rank

    rank_N = [5, 10, 50, 100, 500, 1000, 5000, 10000]
    _rank_histories = []
    trials = 1500

    for _rank_n in rank_N:
        _ranks = torch.empty(trials, dtype=torch.int64, device=device)
        for _rank_trial in range(trials):
            _rank_B = (
                torch.rand((3, 3), dtype=torch.float64, device=device) * 2 - 1
            )
            _rank_cluster = ForgetfulAttention(
                K=_rank_B,
                Q=_rank_B.T,
                V=torch.eye(3, dtype=torch.float64, device=device),
                X=torch.rand((3, _rank_n), dtype=torch.float64, device=device)
                * 10
                - 5,
                T=10,
            )
            _rank_cluster.rescaled_dynamics()
            _ranks[_rank_trial] = matrix_rank(_rank_cluster.P)
        _rank_histories.append(_ranks)

    _rank_max = max(int(_ranks.max()) for _ranks in _rank_histories)
    _rank_bins = np.arange(0.5, _rank_max + 1.5, 1)
    _rank_tick_step = max(1, int(np.ceil(_rank_max / 10)))
    _rank_columns = min(2, len(rank_N))
    _rank_rows = int(np.ceil(len(rank_N) / _rank_columns))
    rank_histogram_figure, _rank_axes = _plt_rank.subplots(
        _rank_rows,
        _rank_columns,
        figsize=(5.5 * _rank_columns, 4 * _rank_rows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    for _rank_axis, _rank_n, _ranks in zip(
        _rank_axes.flat,
        rank_N,
        _rank_histories,
    ):
        _rank_axis.hist(
            _ranks.cpu().numpy(),
            bins=_rank_bins,
            color="#6f83d6",
            edgecolor="white",
            linewidth=0.8,
            rwidth=0.9,
        )
        _rank_axis.set_title(f"N = {_rank_n}")
        _rank_axis.set_xlabel("Final attention rank")
        _rank_axis.set_ylabel("Frequency")
        _rank_axis.set_xticks(np.arange(1, _rank_max + 1, _rank_tick_step))
        _rank_axis.grid(axis="y", alpha=0.2)

    for _rank_axis in _rank_axes.flat[len(rank_N) :]:
        _rank_axis.set_visible(False)

    rank_histogram_figure.suptitle(
        "Distribution of Final Attention-Matrix Rank",
        fontsize=15,
    )
    rank_histogram_figure.tight_layout()
    rank_histogram_figure
    return


@app.cell
def _(Attention, N, device, torch):
    convexClusteringNonPSD = Attention(
        K=torch.rand((3, 3), dtype=torch.float64, device=device) * 2 - 1,
        Q=torch.rand((3, 3), dtype=torch.float64, device=device) * 2 - 1,
        V=torch.eye(3, dtype=torch.float64, device=device),
        X=torch.rand((3, int(N.value)), dtype=torch.float64, device=device)
        * 10
        - 5,
        T=10,
    )
    convexClusteringNonPSD.rescaled_dynamics()
    return (convexClusteringNonPSD,)


@app.cell(hide_code=True)
def non_psd_cluster_plot(N, convexClusteringNonPSD, make_convex_z_plot, mo):
    non_psd_convex_z_plot = make_convex_z_plot(
        convexClusteringNonPSD,
        "Non-PSD Q^T K Cluster Dynamics",
        "non-psd-convex-z-camera",
    )
    mo.vstack([N, non_psd_convex_z_plot], gap=1.0)
    return


@app.cell
def non_psd_rank_histograms(
    ForgetfulAttention,
    device,
    matrix_rank,
    np,
    torch,
):
    import matplotlib.pyplot as _plt_non_psd_rank

    _non_psd_token_counts = [5, 10, 50, 100, 500, 1000]
    _non_psd_rank_histories = []
    _non_psd_trials = 10

    for _non_psd_n in _non_psd_token_counts:
        _non_psd_ranks = torch.empty(
            _non_psd_trials, dtype=torch.int64, device=device
        )
        for _non_psd_trial in range(_non_psd_trials):
            _non_psd_cluster = ForgetfulAttention(
                K=torch.rand((3, 3), dtype=torch.float64, device=device) * 2
                - 1,
                Q=torch.rand((3, 3), dtype=torch.float64, device=device) * 2
                - 1,
                V=torch.eye(3, dtype=torch.float64, device=device),
                X=torch.rand(
                    (3, _non_psd_n), dtype=torch.float64, device=device
                )
                * 10
                - 5,
                T=10,
            )
            _non_psd_cluster.rescaled_dynamics()
            _non_psd_ranks[_non_psd_trial] = matrix_rank(_non_psd_cluster.P)
        _non_psd_rank_histories.append(_non_psd_ranks)

    _non_psd_rank_max = max(
        int(_non_psd_ranks.max()) for _non_psd_ranks in _non_psd_rank_histories
    )
    _non_psd_rank_bins = np.arange(0.5, _non_psd_rank_max + 1.5, 1)
    _non_psd_tick_step = max(1, int(np.ceil(_non_psd_rank_max / 10)))
    non_psd_rank_histogram_figure, _non_psd_axes = _plt_non_psd_rank.subplots(
        2,
        2,
        figsize=(11, 8),
        sharex=True,
        sharey=True,
    )

    for _non_psd_axis, _non_psd_n, _non_psd_ranks in zip(
        _non_psd_axes.flat,
        _non_psd_token_counts,
        _non_psd_rank_histories,
    ):
        _non_psd_axis.hist(
            _non_psd_ranks.cpu().numpy(),
            bins=_non_psd_rank_bins,
            color="#d97941",
            edgecolor="white",
            linewidth=0.8,
            rwidth=0.9,
        )
        _non_psd_axis.set_title(f"N = {_non_psd_n}")
        _non_psd_axis.set_xlabel("Final attention rank")
        _non_psd_axis.set_ylabel("Frequency")
        _non_psd_axis.set_xticks(
            np.arange(1, _non_psd_rank_max + 1, _non_psd_tick_step)
        )
        _non_psd_axis.grid(axis="y", alpha=0.2)

    non_psd_rank_histogram_figure.suptitle(
        "Non-PSD Attention: Distribution of Final Matrix Rank",
        fontsize=15,
    )
    non_psd_rank_histogram_figure.tight_layout()
    non_psd_rank_histogram_figure
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Section 4

    The paper considers a more general case than the one considered above. If instead of assuming $V = \lambda I_d$, we assume that $V$ has a unique dominant, real, and positive eigenvalue $\lambda_1$, and the quadratic $\langle Q\cdot, K\cdot \rangle$ is positive on the eigenspace associated with $\lambda_1$, then tokens converge to union of three (usually two) parallal hyperplanes.

    To put this more formally, the paper defines the $(V, Q, K)$ as a good triple if

    1. $\lambda_1 > |\lambda_2| \geq \cdots \geq |\lambda_n|$, $\lambda_i$ is an eigenvalue of $V$.
    2. $\langle Qe_1, Ke_1 \rangle > 0$ for all $e_1 \in \ker(V - \lambda_1 I_d) \setminus \{0\}$.
    """)
    return


if __name__ == "__main__":
    app.run()
