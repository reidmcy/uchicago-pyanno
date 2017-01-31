# Copyright (c) 2011, Enthought, Ltd.
# Author: Pietro Berkes <pberkes@enthought.com>
# License: Modified BSD license (2-clause)

"""TraitsUI view of the Theta tensor in Model B, and its samples."""

from chaco.array_plot_data import ArrayPlotData
from chaco.data_range_2d import DataRange2D
from chaco.legend import Legend
from chaco.plot import Plot
from chaco.plot_containers import HPlotContainer, VPlotContainer
from chaco.scales.scales import FixedScale
from chaco.scales_tick_generator import ScalesTickGenerator
from chaco.tools.legend_tool import LegendTool

from enable.component_editor import ComponentEditor

from traits.trait_numeric import Array
from traits.trait_types import Instance, Str, Range, Button, Int, Any
from traitsui.item import Item
from pyanno.plots.plot_tools import get_class_color

from pyanno.plots.plots_superclass import PyannoPlotContainer


import numpy as np
from pyanno.ui.appbase.wx_utils import is_display_small


def sigmoid(x):
    return 1./(1.+np.exp(-x))


class ThetaTensorPlot(PyannoPlotContainer):

    # reference to the theta tensor for one annotator
    theta = Array

    # reference to an array of samples for theta for one annotator
    theta_samples = Any

    # index of the annotator
    annotator_idx = Int

    # chaco plot of the tensor
    theta_plot = Any


    def _label_name(self, k):
        """Return a name for the data with index `k`."""
        nclasses = self.theta.shape[0]
        ndigits = int(np.ceil(np.log10(nclasses)))
        format_str = 'theta[{{}},{{:{}d}},:]'.format(ndigits)

        return format_str.format(self.annotator_idx,k)


    def _plot_samples(self, plot, plot_data):
        nclasses = self.theta.shape[0]
        nsamples = self.theta_samples.shape[0]

        for k in range(nclasses):
            samples = np.sort(self.theta_samples[:,k,:], axis=0)
            perc5 = samples[int(nsamples*0.05),:]
            perc95 = samples[int(nsamples*0.95),:]
            avg = samples.mean(0)

            # build polygon
            index_name = self._label_name(k) + '_confint_index'
            value_name = self._label_name(k) + '_confint_value'
            index_coord = []
            value_coord = []
            # bottom part
            for i in range(nclasses):
                index_coord.append(i)
                value_coord.append(perc5[i])
            # top part
            for i in range(nclasses-1, -1, -1):
                index_coord.append(i)
                value_coord.append(perc95[i])

            plot_data.set_data(index_name, np.array(index_coord,
                                                    dtype=float))
            plot_data.set_data(value_name, np.array(value_coord,
                                                    dtype=float))

            # make color lighter and more transparent
            color = get_class_color(k)
            for i in range(3):
                color[i] = min(1.0, sigmoid(color[i]*5.))
            color[-1] = 0.3

            plot.plot(
                (index_name, value_name),
                type = 'polygon',
                face_color = color,
                edge_color = 'black',
                edge_width = 0.5
            )

            # add average
            avg_name = self._label_name(k) + '_avg_value'
            plot_data.set_data(avg_name, avg)
            plot.plot(
                ('classes', avg_name),
                color = get_class_color(k),
                line_style = 'dash'
            )


    def _plot_theta_values(self, plot, plot_data):
        theta = self.theta
        nclasses = theta.shape[0]

        data_names = ['classes']
        for k in range(nclasses):
            name = self._label_name(k)
            plot_data.set_data(name, theta[k,:])
            data_names.append(name)

        plots = {}
        for k in range(nclasses):
            name = self._label_name(k)
            line_plot = plot.plot(
                ['classes', name],
                line_width=2.,
                color = get_class_color(k),
                name=name
            )
            plots[name] = line_plot

        return plots


    def _theta_plot_default(self):

        theta = self.theta
        nclasses = theta.shape[0]

        # create a plot data object and give it this data
        plot_data = ArrayPlotData()

        plot_data.set_data('classes', list(range(nclasses)))

        # create the plot
        plot = Plot(plot_data)

        # --- plot theta samples
        if self.theta_samples is not None:
            self._plot_samples(plot, plot_data)

        # --- plot values of theta
        plots = self._plot_theta_values(plot, plot_data)

        # --- adjust plot appearance

        plot.aspect_ratio = 1.6 if is_display_small() else 1.7

        # adjust axis bounds
        y_high = theta.max()
        if self.theta_samples is not None:
            y_high = max(y_high, self.theta_samples.max())

        plot.range2d = DataRange2D(
            low  = (-0.2, 0.0),
            high = (nclasses-1+0.2, y_high*1.1)
        )

        # create new horizontal axis
        label_axis = self._create_increment_one_axis(
            plot, 0., nclasses, 'bottom')
        label_axis.title = 'True classes'
        self._add_index_axis(plot, label_axis)

        # label vertical axis
        plot.value_axis.title = 'Probability'

        # add legend
        legend = Legend(component=plot, plots=plots,
                        align="ur", border_padding=10)
        legend.tools.append(LegendTool(legend, drag_button="left"))
        legend.padding_right = -100
        plot.overlays.append(legend)

        container = VPlotContainer(width=plot.width + 100, halign='left')
        plot.padding_bottom = 50
        plot.padding_top = 10
        plot.padding_left = 0
        container.add(plot)
        container.bgcolor = 0xFFFFFF

        self.decorate_plot(container, theta)

        return container


    #### View definition #####################################################

    resizable_plot_item = Item(
        'theta_plot',
        editor=ComponentEditor(),
        resizable=True,
        show_label=False,
        height=-250,
        width=-500
        )

    traits_plot_item = Instance(Item)

    def _traits_plot_item_default(self):
        height = -200 if is_display_small() else -250
        return Item(
                    'theta_plot',
                    editor=ComponentEditor(),
                    resizable=False,
                    show_label=False,
                    height=height,
                    )


def plot_theta_tensor(modelB, annotator_idx, theta_samples=None, **kwargs):
    """Display a plot of model B's accuracy tensor, theta.

    The tensor theta[annotator_idx,:,:] is shown for one annotator as a
    set of line plots, each depicting the distribution
    theta[annotator_idx,k,:] = P(annotator_idx outputs : | real class is k).

    Arguments
    ---------
    modelB : ModelB instance
        An instance of ModelB.

    annotator_idx : int
        Index of the annotator for which the parameters are displayed.

    theta_samples : ndarray, shape = (n_samples x n_annotators x n_classes x n_classes)
        Array of samples over the posterior of theta.

    kwargs : dictionary
        Additional keyword arguments passed to the plot. The argument 'title'
        sets the title of the plot.

    Returns
    -------
    theta_view : ThetaTensorPlot instance
        Reference to the plot.
    """
    samples = None
    if theta_samples is not None:
        samples = theta_samples[:,annotator_idx,:,:]

    theta_view = ThetaTensorPlot(
        theta = modelB.theta[annotator_idx,:,:],
        annotator_idx = annotator_idx,
        theta_samples = samples
    )
    theta_view.configure_traits(view='resizable_view')
    return theta_view


#### Testing and debugging ####################################################

def main():
    """ Entry point for standalone testing/debugging. """

    from pyanno.models import ModelB

    model = ModelB.create_initial_state(4, 5)
    anno = model.generate_annotations(100)
    samples = model.sample_posterior_over_accuracy(anno, 10,
                                                   return_all_samples=False)

    theta_view = plot_theta_tensor(model, 2, samples,
                                   title='Debug plot_theta_parameters')

    return model, theta_view


if __name__ == '__main__':
    model, theta_view = main()
