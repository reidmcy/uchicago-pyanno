# Copyright (c) 2011, Enthought, Ltd.
# Author: Pietro Berkes <pberkes@enthought.com>
# License: Modified BSD license (2-clause)

"""TraitsUI view of the Theta parameters, and their samples."""

from chaco.array_plot_data import ArrayPlotData
from chaco.data_range_2d import DataRange2D
from chaco.label_axis import LabelAxis
from chaco.legend import Legend
from chaco.plot import Plot
from chaco.plot_containers import VPlotContainer
from chaco.scales.scales import FixedScale
from chaco.scales_tick_generator import ScalesTickGenerator
from chaco.tools.legend_tool import LegendTool
from enable.component_editor import ComponentEditor

from traits.has_traits import on_trait_change
from traits.trait_numeric import Array
from traits.trait_types import Instance, Bool, Event, Str, DictStrAny, Any
from traitsui.handler import ModelView
from traitsui.item import Item

import numpy as np
from pyanno.plots.plot_tools import get_annotator_color
from pyanno.plots.plots_superclass import PyannoPlotContainer
from pyanno.ui.appbase.wx_utils import is_display_small


def _w_idx(str_, idx):
    """Append number to string. Used to generate PlotData labels"""
    return str_ + str(idx)


class ThetaScatterPlot(ModelView, PyannoPlotContainer):
    """Defines a view of the annotator accuracy parameters, theta.

    The view consists in a Chaco plot that displays the theta parameter for
    each annotator, and samples from the posterior distribution over theta
    with a combination of a scatter plot and a candle plot.
    """

    #### Traits definition ####################################################

    theta_samples_valid = Bool(False)
    theta_samples = Array(dtype=float, shape=(None, None))

    # return value for "Copy" action on plot
    data = DictStrAny

    def _data_default(self):
        return {'theta': self.model.theta, 'theta_samples': None}

    @on_trait_change('redraw,theta_samples,theta_samples_valid')
    def _update_data(self):
        if self.theta_samples_valid:
            theta_samples = self.theta_samples
        else:
            theta_samples = None

        self.data['theta'] = self.model.theta
        self.data['theta_samples'] = theta_samples

    #### plot-related traits
    title = Str('Accuracy (theta)')

    theta_plot_data = Instance(ArrayPlotData)
    theta_plot = Any

    redraw = Event


    ### Plot definition #######################################################

    def _compute_range2d(self):
        low = min(0.6, self.model.theta.min()-0.05)
        if self.theta_samples_valid:
            low = min(low, self.theta_samples.min()-0.05)
        range2d = DataRange2D(low=(0., low),
                              high=(self.model.theta.shape[0]+1, 1.))
        return range2d


    @on_trait_change('redraw', post_init=True)
    def _update_range2d(self):
        self.theta_plot.range2d = self._compute_range2d()


    def _theta_plot_default(self):
        """Create plot of theta parameters."""

        # We plot both the thetas and the samples from the posterior; if the
        # latter are not defined, the corresponding ArrayPlotData names
        # should be set to an empty list, so that they are not displayed
        theta = self.model.theta
        theta_len = theta.shape[0]

        # create the plot data
        if not self.theta_plot_data:
            self.theta_plot_data = ArrayPlotData()
            self._update_plot_data()

        # create the plot
        theta_plot = Plot(self.theta_plot_data)

        for idx in range(theta_len):
            # candle plot summarizing samples over the posterior
            theta_plot.candle_plot((_w_idx('index', idx),
                                    _w_idx('min', idx),
                                    _w_idx('barmin', idx),
                                    _w_idx('avg', idx),
                                    _w_idx('barmax', idx),
                                    _w_idx('max', idx)),
                                    color = get_annotator_color(idx),
                                    bar_line_color = "black",
                                    stem_color = "blue",
                                    center_color = "red",
                                    center_width = 2)

            # plot of raw samples
            theta_plot.plot((_w_idx('ysamples', idx),
                             _w_idx('xsamples', idx)),
                            type='scatter',
                            color='black',
                            marker='dot',
                            line_width=0.5,
                            marker_size=1)

            # plot current parameters
            theta_plot.plot((_w_idx('y', idx), _w_idx('x', idx)),
                            type='scatter',
                            color=get_annotator_color(idx),
                            marker='plus',
                            marker_size=8,
                            line_width=2)

        # adjust axis bounds
        theta_plot.range2d = self._compute_range2d()

        # remove horizontal grid and axis
        theta_plot.underlays = [theta_plot.x_grid, theta_plot.y_axis]

        # create new horizontal axis
        label_list = [str(i) for i in range(theta_len)]

        label_axis = LabelAxis(
            theta_plot,
            orientation = 'bottom',
            positions = list(range(1, theta_len+1)),
            labels = label_list,
            label_rotation = 0
        )
        # use a FixedScale tick generator with a resolution of 1
        label_axis.tick_generator = ScalesTickGenerator(scale=FixedScale(1.))

        theta_plot.index_axis = label_axis
        theta_plot.underlays.append(label_axis)
        theta_plot.padding = 25
        theta_plot.padding_left = 40
        theta_plot.aspect_ratio = 1.0

        container = VPlotContainer()
        container.add(theta_plot)
        container.bgcolor = 0xFFFFFF

        self.decorate_plot(container, theta)
        self._set_title(theta_plot)

        return container


    ### Handle plot data ######################################################

    def _samples_names_and_values(self, idx):
        """Return a list of names and values for the samples PlotData."""

        # In the following code, we rely on lazy evaluation of the
        # X if CONDITION else Y statements to return a default value if the
        # theta samples are not currently defined, or the real value if they
        # are.

        invalid = not self.theta_samples_valid
        samples = [] if invalid else np.sort(self.theta_samples[:,idx])
        nsamples = None if invalid else samples.shape[0]
        perc5 = None if invalid else samples[int(nsamples*0.05)]
        perc95 = None if invalid else samples[int(nsamples*0.95)]

        data_dict = {
            'xsamples':
                [] if invalid else samples,
            'ysamples':
                [] if invalid else (
                    np.random.random(size=(nsamples,))*0.1-0.05 + idx + 1.2
                    ),
            'min':
                [] if invalid else [perc5],
            'max':
                [] if invalid else [perc95],
            'barmin':
                [] if invalid else [samples.mean() - samples.std()],
            'barmax':
                [] if invalid else [samples.mean() + samples.std()],
            'avg':
                [] if invalid else [samples.mean()],
            'index':
                [] if invalid else [idx + 0.8]
        }

        name_value = [(_w_idx(name, idx), value)
                      for name, value in list(data_dict.items())]
        return name_value

    @on_trait_change('theta_plot_data,theta_samples_valid,redraw')
    def _update_plot_data(self):
        """Updates PlotData on changes."""
        theta = self.model.theta

        plot_data = self.theta_plot_data

        if plot_data is not None:
            for idx, th in enumerate(theta):
                plot_data.set_data('x%d' % idx, [th])
                plot_data.set_data('y%d' % idx, [idx+1.2])

                for name_value in self._samples_names_and_values(idx):
                    name, value = name_value
                    plot_data.set_data(name, value)


    #### View definition #####################################################

    resizable_plot_item = Item(
        'theta_plot',
        editor=ComponentEditor(),
        resizable=True,
        show_label=False,
        width=600,
        height=400
        )

    traits_plot_item = Instance(Item)

    def _traits_plot_item_default(self):
        height = -220 if is_display_small() else -280
        return Item('theta_plot',
                    editor=ComponentEditor(),
                    resizable=False,
                    show_label=False,
                    height=height
                    )


class ThetaDistrPlot(PyannoPlotContainer):
    """Defines a view of the annotator accuracy parameters, theta.

    The view consists in a Chaco plot that displays the theta parameter for
    each annotator, and samples from the posterior distribution over theta
    as a discretized distribution over theta.
    """

    # reference to the theta tensor for one annotator
    theta = Array

    # reference to an array of samples for theta for one annotator
    theta_samples = Any

    # chaco plot of the tensor
    theta_plot = Any

    def _theta_name(self, k):
        nannotators = self.theta.shape[0]
        ndigits = int(np.ceil(np.log10(nannotators)))
        format_str = 'theta[{{:{}d}}]'.format(ndigits)
        return format_str.format(k)

    def _theta_plot_default(self):
        theta = self.theta
        nannotators = theta.shape[0]
        samples = self.theta_samples

        # plot data object
        plot_data = ArrayPlotData()

        # create the plot
        plot = Plot(plot_data)

        # --- plot theta as vertical dashed lines
        # add vertical lines extremes
        plot_data.set_data('line_extr', [0., 1.])

        for k in range(nannotators):
            name = self._theta_name(k)
            plot_data.set_data(name, [theta[k], theta[k]])

        plots = {}
        for k in range(nannotators):
            name = self._theta_name(k)
            line_plot = plot.plot(
                (name, 'line_extr'),
                line_width = 2.,
                color = get_annotator_color(k),
                line_style = 'dash',
                name = name
            )
            plots[name] = line_plot

        # --- plot samples as distributions
        if samples is not None:
            bins = np.linspace(0., 1., 100)
            max_hist = 0.
            for k in range(nannotators):
                name = self._theta_name(k) + '_distr_'
                hist, x = np.histogram(samples[:,k], bins=bins)
                hist = hist / float(hist.sum())
                max_hist = max(max_hist, hist.max())

                # make "bars" out of histogram values
                y = np.concatenate(([0], np.repeat(hist, 2), [0]))
                plot_data.set_data(name+'x', np.repeat(x, 2))
                plot_data.set_data(name+'y', y)

            for k in range(nannotators):
                name = self._theta_name(k) + '_distr_'
                plot.plot((name+'x', name+'y'),
                          line_width = 2.,
                          color = get_annotator_color(k)
                          )

        # --- adjust plot appearance

        plot.aspect_ratio = 1.6 if is_display_small() else 1.7
        plot.padding = [20,0,10,40]

        # adjust axis bounds
        x_low, x_high = theta.min(), theta.max()
        y_low, y_high = 0., 1.
        if samples is not None:
            x_high = max(x_high, samples.max())
            x_low = min(x_low, samples.min())
            y_high = max_hist

        plot.range2d = DataRange2D(
            low  = (max(x_low-0.05, 0.), y_low),
            high = (min(x_high*1.1, 1.), min(y_high*1.1, 1.))
        )

        # label axes
        plot.value_axis.title = 'Probability'
        plot.index_axis.title = 'Theta'

        # add legend
        legend = Legend(component=plot, plots=plots,
                        align="ul", padding=5)
        legend.tools.append(LegendTool(legend, drag_button="left"))
        plot.overlays.append(legend)

        container = VPlotContainer()
        container.add(plot)
        container.bgcolor = 0xFFFFFF

        self.decorate_plot(container, theta)
        self._set_title(plot)

        return container


    #### View definition #####################################################

    resizable_plot_item = Item(
        'theta_plot',
        editor=ComponentEditor(),
        resizable=True,
        show_label=False,
        width = 600
        )

    traits_plot_item = Instance(Item)

    def _traits_plot_item_default(self):
        height = -220 if is_display_small() else -280
        return Item(
                    'theta_plot',
                    editor=ComponentEditor(),
                    resizable=False,
                    show_label=False,
                    height=height,
                    )


def plot_theta_parameters(model, theta_samples=None,
                          type='distr', **kwargs):
    """Display a plot of the annotator accuracy parameters, theta.

    This class gives a graphical representation of the `theta` accuracy
    parameters for :class:`~pyanno.modelA.ModelA`,
    :class:`~pyanno.modelBt.ModelBt`, and
    :class:`~pyanno.modelBt_loopdesign.ModelBtLoopDesign`.

    Arguments
    ---------
    model : instance
        an instance of :class:`~pyanno.modelA.ModelA`,
        :class:`~pyanno.modelBt.ModelBt`, or
        :class:`~pyanno.modelBt_loopdesign.ModelBtLoopDesignModelBt`

    theta_samples : ndarray, shape = (n_items, n_annotators)
        Samples from the posterior over theta,
        as returned by the method `model.sample_posterior_over_accuracy`.
        If given, they are displayed as a probability distribution superimposed
        to the values or `model.theta`.

    type : string
        Either 'scatter' or 'distr'. Parametrizes two different kind of plots.

    kwargs : dictionary
        Additional keyword arguments passed to the plot. The argument 'title'
        sets the title of the plot.
    """

    if type == 'distr':
        theta_view = ThetaDistrPlot(theta = model.theta,
                                    theta_samples = theta_samples)
    else:
        theta_view = ThetaScatterPlot(model=model, **kwargs)
        if theta_samples is not None:
            theta_view.theta_samples = theta_samples
            theta_view.theta_samples_valid = True

    theta_view.configure_traits(view='resizable_view')
    return theta_view


#### Testing and debugging ####################################################

def main():
    """ Entry point for standalone testing/debugging. """

    from pyanno.modelBt_loopdesign import ModelBtLoopDesign

    model = ModelBtLoopDesign.create_initial_state(5)
    annotations = model.generate_annotations(100)
    theta_samples = model.sample_posterior_over_accuracy(
        annotations, 100,
        step_optimization_nsamples = 3
    )

    theta_view = plot_theta_parameters(model, theta_samples,
                                       type='distr',
                                       title='Debug plot_theta_parameters')

    return model, theta_view


if __name__ == '__main__':
    model, theta_view = main()
