# Copyright (c) 2011, Enthought, Ltd.
# Author: Pietro Berkes <pberkes@enthought.com>
# License: Modified BSD license (2-clause)

from traits.has_traits import HasTraits, on_trait_change
from traits.trait_numeric import Array
from traits.trait_types import Enum, Button, Instance, Int, Str, Float
from traitsui.group import VGroup, HGroup
from traitsui.item import Item, Spring
from traitsui.message import message
from traitsui.view import View

import numpy as np
import pyanno.measures as measures
from pyanno.plots.matrix_plot import MatrixPlot

import logging
logger = logging.getLogger(__name__)


# statistical measures for pairs of annotators
from pyanno.util import PyannoValueError


PAIRWISE_STATS = {
    "Scott's Pi": measures.scotts_pi,
    "Cohen's Kappa": measures.cohens_kappa,
    "Cohen's weighted Kappa": measures.cohens_weighted_kappa,
    "Pearson's Rho": measures.pearsons_rho,
    "Spearman's Rho": measures.spearmans_rho
}

# statistical measures for all annotators
GLOBAL_STATS = {
    "Fleiss' Kappa": measures.fleiss_kappa,
    "Krippendorff's Kappa": measures.krippendorffs_alpha,
    "Cronbach's Alpha": measures.cronbachs_alpha
}

ALL_STATS_NAMES = list(PAIRWISE_STATS.keys()) + list(GLOBAL_STATS.keys())


class _SingleStatView(HasTraits):
    value = Float
    name = Str
    txt = Str

    @on_trait_change('value')
    def _update_txt(self):
        txt = "{name} = {val:.4}".format(
            name = self.name,
            val = self.value
        )
        self.txt = txt

    traits_view = View(
        Item('txt', style='readonly', show_label=False)
    )


class AnnotationsStatisticsView(HasTraits):

    statistics_name = Enum(ALL_STATS_NAMES)

    annotations = Array

    nclasses = Int

    info_button = Button("Info...")

    stats_view = Instance(HasTraits)


    def _info_button_fired(self):
        """Open dialog with description of statistics."""
        if self.statistics_name in GLOBAL_STATS:
            stat_func = GLOBAL_STATS[self.statistics_name]
        else:
            stat_func = PAIRWISE_STATS[self.statistics_name]
        message(message = stat_func.__doc__, title='Statistics info')


    @on_trait_change('statistics_name')
    def _update_stats_view(self):
        if self.statistics_name in GLOBAL_STATS:
            stat_func = GLOBAL_STATS[self.statistics_name]

            try:
                res = stat_func(self.annotations, nclasses=self.nclasses)
            except PyannoValueError as e:
                logger.info(e)
                res = np.nan

            self.stats_view = _SingleStatView(value=res,
                                              name=self.statistics_name)

        else:
            stat_func = PAIRWISE_STATS[self.statistics_name]

            try:
                res = measures.pairwise_matrix(stat_func, self.annotations,
                                               nclasses=self.nclasses)
            except PyannoValueError as e:
                logger.info(e)
                nannotators = self.annotations.shape[1]
                res = np.empty((nannotators, nannotators))
                res.fill(np.nan)

            self.stats_view = MatrixPlot(matrix=res,
                                              colormap_low=-1.0,
                                              colormap_high=1.0,
                                              title=self.statistics_name)


    def traits_view(self):
        self.statistics_name = "Cohen's Kappa"
        traits_view = View(
            VGroup(
                HGroup(
                    Item("statistics_name", show_label=False),
                    Item("info_button", show_label=False)
                ),
                Spring(),
                HGroup(
                    Spring(),
                    Item("stats_view",
                         style="custom",
                         show_label=False,
                         width=300,
                         resizable=False),
                    Spring()
                ),
            ),
            width=400,
            resizable = True
        )
        return traits_view


#### Testing and debugging ####################################################

def main():
    """ Entry point for standalone testing/debugging. """

    from pyanno.modelBt_loopdesign import ModelBtLoopDesign
    model = ModelBtLoopDesign.create_initial_state(5)
    annotations = model.generate_annotations(20)

    stats_view = AnnotationsStatisticsView(annotations=annotations, nclasses=5)
    stats_view.configure_traits()
    return model, annotations, stats_view


if __name__ == '__main__':
    m, a, sv = main()
