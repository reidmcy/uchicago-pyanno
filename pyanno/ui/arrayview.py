# Copyright (c) 2011, Enthought, Ltd.
# Author: Pietro Berkes <pberkes@enthought.com>
# License: Modified BSD license (2-clause)

from traits.api import HasTraits, Property, Array
from traits.trait_types import List, Int, Bool, Str
from traitsui.api import View, Item, TabularEditor
from traitsui.group import VGroup
from traitsui.tabular_adapter import TabularAdapter
from traitsui.menu import NoButtons


class Array2DAdapter(TabularAdapter):
    columns = List
    show_index = Bool(True)
    count_from_one = Bool(True)
    ncolumns = Int

    data_format = Str('%s')

    font = 'Courier 10'
    alignment = 'right'
    format = data_format
    index_text = Property
    index_alignment = Property
    width = 60


    def _get_index_text(self):
        return '- {} -'.format(self.row)


    def _get_index_alignment(self):
        return 'left'


    def _columns_default(self):
        if self.count_from_one:
            columns = [('%d' % (i+1), i) for i in range(self.ncolumns)]
        else:
            columns = [('%d' % i, i) for i in range(self.ncolumns)]

        if self.show_index:
            columns.insert(0, ('items', 'index'))
        return columns


#### Testing and debugging ####################################################

def main():
    """Entry point for standalone testing/debugging."""

    class TestShowArray(HasTraits):

        data = Array

        view = View(
            Item(
                'data',
                editor=TabularEditor
                         (
                         adapter=Array2DAdapter(ncolumns=2,
                                                format='%s',
                                                show_index=False)),
                show_label=False
            ),
            title     = 'Array2D editor',
            width     = 0.3,
            height    = 0.8,
            resizable = True,
            buttons   = NoButtons
        )

        VGroup(Item('data',
                     editor=TabularEditor
                         (
                         adapter=Array2DAdapter(ncolumns=2,
                                                format='%d',
                                                show_index=False)),
                     show_label=False)),

    data = [['a', 'b'], [1, 2]]
    blah = TestShowArray(data=data)
    blah.data = data
    print(blah.data)
    blah.configure_traits()


if __name__ == '__main__':
    main()
