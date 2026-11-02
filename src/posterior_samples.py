"""Read-only physical conductivity fields from a saved posterior calculation.

Ordinary arrays use the recorded canonical spatial order on their final node
axis; ``coordinates`` and ``cells`` provide immutable spatial labels. Reading
saved fields and physical summaries needs only NumPy, not a heat solver or JAX.
"""

# type imports
from dataclasses import dataclass

# numerical imports
import numpy as np

# local imports
from posterior_store import RunStore


def _immutable(values):
    array = np.asarray(values)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _index(value, name, *, minimum=0):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < minimum
    ):
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return int(value)


def _probabilities(quantiles):
    probabilities = np.asarray(quantiles)
    if (
        probabilities.ndim != 1
        or np.iscomplexobj(probabilities)
        or probabilities.dtype.kind not in "biuf"
        or not np.isfinite(probabilities).all()
        or np.any((probabilities < 0) | (probabilities > 1))
    ):
        raise ValueError("Quantile probabilities must be a finite vector in [0,1].")
    probabilities = probabilities.astype(float)
    return probabilities


@dataclass(frozen=True)
class FieldSummary:
    """Pooled pointwise physical conductivity statistics over retained states.

    Parameters
    ----------
    mean, sd : numpy.ndarray
        Physical mean and population standard deviation, with final node axis
        labelled by coordinates. These average conductivity after transformation
        from log conductivity. Every rejection repeat contributes one state.
    quantiles : numpy.ndarray
        Exact pointwise quantiles, with probability then node axes. These are
        pointwise intervals rather than simultaneous credible bands.
    probabilities : numpy.ndarray
        Requested quantile probabilities in the supplied order.
    coordinates, cells : numpy.ndarray
        Immutable canonical physical node coordinates and mesh connectivity.
    counts : numpy.ndarray
        Retained count per chain, preserving evidence of unequal partial runs.
        Statistics pool all these states; longer chains contribute more states.
    qualified : bool
        False until convergence diagnostics have separately qualified the run.
    """

    mean: np.ndarray
    sd: np.ndarray
    quantiles: np.ndarray
    probabilities: np.ndarray
    coordinates: np.ndarray
    counts: np.ndarray
    qualified: bool
    cells: np.ndarray

    def __post_init__(self):
        for name in (
            "mean",
            "sd",
            "quantiles",
            "probabilities",
            "coordinates",
            "counts",
            "cells",
        ):
            object.__setattr__(self, name, _immutable(getattr(self, name)))


class PosteriorSamples:
    """Saved physical fields, with explicit chain/draw/node array selections."""

    @classmethod
    def open(cls, output):
        """Open saved evidence for inspection, requiring only NumPy.

        Parameters
        ----------
        output : path-like
            Existing complete or interrupted calculation directory.

        Returns
        -------
        PosteriorSamples
            Read-only handle with copied metadata and immutable spatial labels.
        """
        result = cls.__new__(cls)
        result._store = RunStore.open(output)
        try:
            result._record = result._store.problem_record
            result._coordinates = _immutable(result._record.arrays["coordinates"])
            result._cells = _immutable(result._record.arrays["cells"])
        except Exception:
            result.close()
            raise
        return result

    @property
    def output(self):
        """Resolved directory holding this saved physical calculation."""
        return self._store.path

    @property
    def metadata(self):
        """Independent copy of the saved controls, starts and requested total."""
        return self._store.metadata

    @property
    def coordinates(self):
        """Immutable physical (x,y) coordinates labelling the final node axis."""
        return self._coordinates

    @property
    def cells(self):
        """Immutable oriented connectivity into the canonical physical nodes."""
        return self._cells

    def _chain(self, chain):
        chain = _index(chain, "chain")
        if chain >= self.metadata["chains"]:
            raise IndexError("chain is outside the saved calculation.")
        return chain

    def _counts(self):
        counts = []
        for chain in range(self.metadata["chains"]):
            try:
                counts.append(self._store.checkpoint(chain).draws_done)
            except KeyError:
                counts.append(0)
        return np.array(counts, dtype=np.int64)

    def _slice(self, chain, start, stop, *, count=None):
        chain = self._chain(chain)
        count = int(self._counts()[chain]) if count is None else count
        start = _index(start, "start")
        stop = count if stop is None else _index(stop, "stop")
        if start > stop or stop > count:
            raise ValueError("Requested draw range is outside the retained states.")
        return chain, start, stop

    def _record_at(self, chain, draw):
        chain = self._chain(chain)
        draw = _index(draw, "draw")
        if draw >= self._counts()[chain]:
            raise IndexError("draw is outside the retained states.")
        return next(self._store.records(chain, start=draw, stop=draw + 1))

    def field(self, *, chain, draw):
        """Return one copied physical conductivity field in canonical node order.

        Parameters
        ----------
        chain, draw : int
            Zero-based saved chain and retained-state indices.

        Returns
        -------
        numpy.ndarray
            Shape (N,), paired with ``coordinates`` and ``cells``.
        """
        return self._record_at(chain, draw).h.copy()

    def state(self, *, chain, draw):
        """Recover every inferred coordinate at one retained physical state.

        Parameters
        ----------
        chain, draw : int
            Zero-based saved chain and retained-state indices.

        Returns
        -------
        heat_inference.InferredState
            Copied inferred coordinates labelled by their posterior identity.
            This conversion imports the physical inference classes lazily.
        """
        from heat_inference import InferredState

        return InferredState(
            self._record_at(chain, draw).inferred_coordinates, self._record.problem_id
        )

    def load_draws(self, *, chains=None, start=0, stop=None):
        """Explicitly load an equal-length selection of physical fields.

        Parameters
        ----------
        chains : sequence of int or None
            Selected chains in requested order; None selects every chain.
        start, stop : int, optional
            Common retained range, with exclusive stop. Unequal lengths require
            an explicit common range or a selection of equal-length chains.

        Returns
        -------
        numpy.ndarray
            Shape (C,D,N); final nodes follow ``coordinates`` and ``cells``.
            No common length is inferred by silently truncating partial chains.
        """
        chains = (
            list(range(self.metadata["chains"])) if chains is None else list(chains)
        )
        if not chains:
            raise ValueError("Select at least one chain.")
        chains = [self._chain(chain) for chain in chains]
        if len(set(chains)) != len(chains):
            raise ValueError("Select each chain at most once.")
        counts = self._counts()
        slices = [
            self._slice(chain, start, stop, count=int(counts[chain]))
            for chain in chains
        ]
        lengths = [end - begin for _, begin, end in slices]
        if len(set(lengths)) != 1:
            raise ValueError(
                "Whole-array requests require equal lengths; supply an explicit common draw range."
            )
        result = np.empty((len(chains), lengths[0], len(self.coordinates)))
        for row, (chain, begin, end) in enumerate(slices):
            for column, record in enumerate(
                self._store.records(chain, start=begin, stop=end)
            ):
                result[row, column] = record.h
        return result

    def summary(self, *, quantiles=(0.05, 0.5, 0.95), node_chunk_size=64):
        """Compute pooled physical statistics using bounded node chunks.

        Parameters
        ----------
        quantiles : sequence of float
            Pointwise probabilities in [0,1], in requested output order.
        node_chunk_size : int
            Positive maximum number of nodes whose retained draws are loaded
            together for exact quantiles. Whole-chain arrays are never loaded.

        Returns
        -------
        FieldSummary
            Mean, population sd and exact physical quantiles over every retained
            state, excluding warmup. Counts label unequal partial chains and
            qualification remains False until diagnostics are available.
        """
        width = _index(node_chunk_size, "node_chunk_size", minimum=1)
        probabilities = _probabilities(quantiles)
        counts = self._counts()
        total = int(counts.sum())
        if total == 0:
            raise ValueError("No retained draws are available for a physical summary.")
        nodes = len(self.coordinates)
        mean, sd = np.empty(nodes), np.empty(nodes)
        limits = np.empty((len(probabilities), nodes))
        for begin in range(0, nodes, width):
            end = min(begin + width, nodes)
            block = np.empty((total, end - begin))
            row = 0
            for chain, count in enumerate(counts):
                for record in self._store.records(chain, stop=int(count)):
                    block[row] = record.h[begin:end]
                    row += 1
            mean[begin:end] = block.mean(axis=0)
            sd[begin:end] = block.std(axis=0, ddof=0)
            limits[:, begin:end] = np.quantile(block, probabilities, axis=0)
        from posterior_diagnostics import matching_qualification

        qualified = matching_qualification(self, probabilities, counts)
        return FieldSummary(
            mean,
            sd,
            limits,
            probabilities,
            self.coordinates,
            counts,
            qualified,
            self.cells,
        )

    def diagnostics(self, *, node_chunk_size=64, quantiles=(0.05, 0.5, 0.95)):
        """Assess saved conductivity and sensor convergence and save the evidence.

        Parameters
        ----------
        node_chunk_size : int
            Positive maximum number of variables inspected together, retaining
            every chain and draw. No whole-field history is implicitly loaded.
        quantiles : sequence of float
            Report Monte Carlo errors at these physical quantile probabilities;
            defaults match summary. Endpoints 0/1 remain unqualified.

        Returns
        -------
        posterior_diagnostics.DiagnosticReport
            Explicit qualified, unqualified, incomplete or unavailable status,
            physical statistics, counts, locations and reasons. The advisory JSON
            report leaves chain states and numerical identity unchanged. Summary
            only uses a saved report matching its history and probabilities.
        """
        from posterior_diagnostics import diagnose

        return diagnose(self, node_chunk_size=node_chunk_size, quantiles=quantiles)

    def sensor_predictions(self, *, chain, start=0, stop=None):
        """Read retained predicted thermometer temperatures without a heat solve.

        Parameters
        ----------
        chain : int
            Zero-based saved chain.
        start, stop : int, optional
            Retained-state range with exclusive stop.

        Returns
        -------
        numpy.ndarray
            Shape (D,S), preserving the saved physical sensor order.
        """
        chain, start, stop = self._slice(chain, start, stop)
        sensors = len(self._record.arrays["sensor_values"])
        result = np.empty((stop - start, sensors))
        for row, record in enumerate(
            self._store.records(chain, start=start, stop=stop)
        ):
            result[row] = record.sensor_temperature
        return result

    def temperature(self, *, chain, draw):
        """Privately solve the full temperature associated with one saved field.

        Parameters
        ----------
        chain, draw : int
            Zero-based chain and retained-state selection.

        Returns
        -------
        numpy.ndarray
            Shape (N,), in recorded canonical order. The reconstructed private
            heat calculation never changes a caller-owned experiment. In
            general T(mean(h)) differs from mean(T(h)).
        """
        from heat_inference import HeatInverseProblem

        h = self.field(chain=chain, draw=draw)
        with HeatInverseProblem.from_record(self._record) as problem:
            temperature, _ = problem.posterior.predict_h(h)
            return temperature

    def _model(self):
        from heat_model import FrozenHeatModel

        arrays = self._record.arrays
        return FrozenHeatModel(
            self.coordinates,
            self.cells,
            self._record.metadata["model"]["cell_type"],
            arrays["source"],
            arrays["boundary_dofs"],
            arrays["boundary_values"],
        )

    def _selection(self, chain, draw, summary, *, applying=False):
        if summary is None:
            if chain is None or draw is None:
                raise ValueError(
                    "Supply one complete chain/draw selection or one physical summary statistic."
                )
            return self.field(chain=chain, draw=draw)
        if chain is not None or draw is not None:
            raise ValueError(
                "Select exactly one draw or one physical summary statistic."
            )
        if isinstance(summary, str) and summary in ("mean", "sd"):
            if applying and summary == "sd":
                raise ValueError(
                    "A standard deviation (sd) is for plotting, not a physical conductivity to apply."
                )
            return getattr(self.summary(quantiles=()), summary)
        if (
            isinstance(summary, tuple)
            and len(summary) == 2
            and summary[0] == "quantile"
        ):
            probability = np.asarray(summary[1])
            if probability.ndim != 0:
                raise ValueError("Select a scalar quantile probability in [0,1].")
            return self.summary(quantiles=(summary[1],)).quantiles[0]
        raise ValueError("summary must be 'mean', 'sd', or ('quantile', probability).")

    def as_function(self, *, chain=None, draw=None, summary=None):
        """Create a private nodal conductivity function for existing plots.

        Parameters
        ----------
        chain, draw : int, optional
            One complete zero-based draw selection, exclusive with summary.
        summary : str or tuple, optional
            'mean', 'sd', or ('quantile', probability) of physical conductivity.

        Returns
        -------
        dolfinx.fem.Function
            Private matching mesh/function whose DOF order matches plotting
            coordinates: ``plot_scalar_mesh(f.function_space, f.x.array, 'h')``.
        """
        from dolfinx import fem
        from heat_model import canonical_dof_indices

        values = self._selection(chain, draw, summary)
        domain = self._model()._build_mesh()
        function = fem.Function(fem.functionspace(domain, ("Lagrange", 1)))
        function.name = "ThermalConductivity"
        function.x.array[canonical_dof_indices(function.function_space)] = values
        function.x.scatter_forward()
        return function

    def apply_to(self, forward, *, chain=None, draw=None, summary=None):
        """Apply a selected conductivity and solve its matching temperature.

        Parameters
        ----------
        forward : SteadyHeat2DForwardSolver
            Destination with the same mesh/connectivity, evaluated source and
            boundary temperatures. Current conductivity/temperature are ignored.
        chain, draw : int, optional
            Complete zero-based draw selection, exclusive with summary.
        summary : str or tuple, optional
            Physical 'mean' or ('quantile', probability); sd cannot be applied.

        Returns
        -------
        dolfinx.fem.Function
            Destination's preserved temperature object after a converged solve.
            Mismatches and failed applications leave the destination unchanged.
            Applying mean(h) computes T(mean(h)), not the posterior mean T.
        """
        from heat_model import FrozenHeatModel

        values = self._selection(chain, draw, summary, applying=True)
        saved, destination = self._model(), FrozenHeatModel.from_forward(forward)

        def connectivity(cells):
            cells = np.sort(cells, axis=1)
            return cells[np.lexsort(cells.T[::-1])]

        if (
            saved.cell_type != destination.cell_type
            or not np.array_equal(saved.coordinates, destination.coordinates)
            or not np.array_equal(
                connectivity(saved.cells), connectivity(destination.cells)
            )
        ):
            raise ValueError(
                "Destination mesh/connectivity differs from the saved physical experiment."
            )
        if not np.array_equal(saved.source, destination.source):
            raise ValueError(
                "Destination heat source differs from the saved physical experiment."
            )
        if not np.array_equal(
            saved.boundary_dofs, destination.boundary_dofs
        ) or not np.array_equal(saved.boundary_values, destination.boundary_values):
            raise ValueError(
                "Destination boundary temperatures differ from the saved physical experiment."
            )
        return forward.apply_conductivity_field(values, coordinates=self.coordinates)

    def close(self):
        """Release the saved calculation handle; repeated calls are harmless."""
        self._store.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
