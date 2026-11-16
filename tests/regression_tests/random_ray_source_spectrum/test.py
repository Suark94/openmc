from pathlib import Path

import numpy as np
import openmc
from openmc.examples import slab_mg

from tests.testing_harness import TolerantPyAPITestHarness


class SourceSpectrumTestHarness(TolerantPyAPITestHarness):
    def _get_results(self):
        with openmc.StatePoint(self._sp_name) as sp:
            absorption = sp.get_tally(name='total').mean.ravel()
            # Reflective boundaries and pure absorption mean each group's
            # absorption rate equals its share of the total source strength.
            np.testing.assert_allclose(absorption, [0.2, 0.8], rtol=1e-5)
        return super()._get_results()

    def _cleanup(self):
        super()._cleanup()
        Path('2g.h5').unlink(missing_ok=True)


def test_random_ray_source_spectrum():
    model = slab_mg()
    groups = openmc.mgxs.EnergyGroups([0.0, 1.0e4, 2.0e6])
    xs = openmc.XSdata('mat_1', groups)
    xs.order = 0
    xs.set_total([2.0, 1.0])
    xs.set_absorption([2.0, 1.0])
    xs.set_scatter_matrix(np.zeros((2, 2, 1)))
    library = openmc.MGXSLibrary(groups)
    library.add_xsdata(xs)
    library.export_to_hdf5('2g.h5')

    cell = next(iter(model.geometry.get_all_cells().values()))
    box = openmc.model.RectangularParallelepiped(
        0.0, 10.0, 0.0, 10.0, 0.0, 10.0, boundary_type='reflective')
    cell.region = -box

    settings = model.settings
    settings.run_mode = 'fixed source'
    settings.particles = 50
    space = openmc.stats.Box((0.0, 0.0, 0.0), (10.0, 10.0, 10.0))
    settings.random_ray = {
        'distance_active': 50.0,
        'distance_inactive': 20.0,
        'ray_source': openmc.IndependentSource(space=space),
        'volume_estimator': 'naive',
    }
    settings.source = openmc.IndependentSource(
        space=space,
        energy=openmc.stats.Discrete([1.0e3, 1.0e6], [2.0, 8.0]),
        constraints={'domains': [cell]})

    tally = openmc.Tally(name='total')
    tally.filters = [openmc.EnergyFilter(groups.group_edges)]
    # All collisions are absorptions; random ray supports the total score.
    tally.scores = ['total']
    model.tallies = openmc.Tallies([tally])

    harness = SourceSpectrumTestHarness('statepoint.10.h5', model)
    harness.main()
