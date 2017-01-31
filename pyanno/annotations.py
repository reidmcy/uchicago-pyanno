# Copyright (c) 2011, Enthought, Ltd.
# Author: Pietro Berkes <pberkes@enthought.com>
# License: Modified BSD license (2-clause)

"""Defines objects to create and manipulate raw annotations."""

from traits.has_traits import HasStrictTraits, cached_property
from traits.trait_numeric import Array
from traits.trait_types import Str, List, Int
from traits.traits import Property

import numpy as np
from pyanno.util import MISSING_VALUE, PyannoValueError


def _robust_isnan(x):
    res = False

    # workaround for the fact that np.isnan is not defined for non-numerical
    # type, e.g. strings
    try:
        res = np.isnan(x)
    except NotImplementedError:
        pass

    return res


def _is_nan_in_list(lst):
    return np.any([_robust_isnan(el) for el in lst])


class AnnotationsContainer(HasStrictTraits):
    """Translate from general annotations files and arrays to pyAnno's format.

    This class exposes a few methods to import data from files and arrays, and
    converts them to pyAnno's format:

    * annotations are 2D integer arrays; rows index items, and columns
      annotators

    * label classes are numbered 0 to :attr:`nclasses`-1 . The attribute
      :attr:`labels` defines a mapping from label tokens to label classes

    * missing values are defined as :attr:`pyanno.util.MISSING_VALUE`. The
      attribute :attr:`missing_values` contains the missing values tokens
      found in the original, raw data

    The converted data can be accessed through the :attr:`annotations` property.

    The `AnnotationsContainer` is also used as the format to store annotations
    in :class:`~pyanno.database.PyannoDatabase` objects.
    """

    DEFAULT_MISSING_VALUES_STR = ['-1', 'NA', 'None', '*']
    DEFAULT_MISSING_VALUES_NUM = [-1, np.nan, None]
    DEFAULT_MISSING_VALUES_ALL = (DEFAULT_MISSING_VALUES_STR +
                                  DEFAULT_MISSING_VALUES_NUM)

    #: raw annotations, as they are imported from file or array
    raw_annotations = List(List)

    #: name of file or array from which the annotations were imported
    name = Str

    #: list of all labels found in file/array
    labels = List

    #: labels corresponding to a missing value
    missing_values = List

    #: number of classes found in the annotations
    nclasses = Property(Int, depends_on='labels')
    def _get_nclasses(self):
        return len(self.labels)

    #: number of annotators
    nannotators = Property(Int, depends_on='raw_annotations')
    def _get_nannotators(self):
        return len(self.raw_annotations[0])

    #: number of annotations
    nitems = Property(Int, depends_on='raw_annotations')
    def _get_nitems(self):
        return len(self.raw_annotations)

    #: annotations in pyAnno format
    annotations = Property(Array, depends_on='raw_annotations')

    @cached_property
    def _get_annotations(self):
        nitems, nannotators = len(self.raw_annotations), self.nannotators
        anno = np.empty((nitems, nannotators), dtype=int)

        # build map from labels and missing values to annotation values
        raw2val = dict(list(zip(self.labels, list(range(self.nclasses)))))
        raw2val.update([(mv, MISSING_VALUE) for mv in self.missing_values])

        # translate
        nan_in_missing_values = _is_nan_in_list(self.missing_values)
        for i, row in enumerate(self.raw_annotations):
            for j, lbl in enumerate(row):
                if nan_in_missing_values and _robust_isnan(lbl):
                    # workaround for the fact that np.nan cannot be used as
                    # the key to a dictionary, since np.nan != np.nan
                    anno[i,j] = MISSING_VALUE
                else:
                    anno[i,j] = raw2val[lbl]

        return anno


    @staticmethod
    def _from_generator(rows_generator, missing_values, name=''):

        missing_set = set(missing_values)
        labels_set = set()

        raw_annotations = []
        nannotators = None
        for n, row in enumerate(rows_generator):

            # verify that number of lines is consistent in the whole file
            if nannotators is None: nannotators = len(row)
            else:
                if len(row) != nannotators:
                    raise PyannoValueError(
                        'File has inconsistent number of entries '
                        'on separate lines (line {})'.format(n))

            raw_annotations.append(row)
            labels_set.update(row)

        # remove missing values from set of labels
        all_labels = sorted(list(labels_set - missing_set))
        missing_values = sorted(list(missing_set & labels_set))

        # workaround for np.nan != np.nan, so intersection does not work
        if _is_nan_in_list(all_labels):
            # uses fact that np.nan < x, for every x
            all_labels = all_labels[1:]
            missing_values.insert(0, np.nan)

        # create annotations object
        anno = AnnotationsContainer(
            raw_annotations = raw_annotations,
            labels = all_labels,
            missing_values = missing_values,
            name = name
        )

        return anno

    @staticmethod
    def _from_file_object(fobj, missing_values=None, name=''):
        """Useful for testing, as it can be called using a StringIO object.
        """

        if missing_values is None:
            missing_values = AnnotationsContainer.DEFAULT_MISSING_VALUES_STR

        # generator for rows of file-like object
        def file_row_generator():
            for line in fobj.readlines():
                # remove commas and split in individual tokens
                line = line.strip().replace(',', ' ')

                # ignore empty lines
                if len(line) == 0: continue

                labels = line.split()
                yield labels

        return AnnotationsContainer._from_generator(file_row_generator(),
                                           missing_values,
                                           name=name)


    @staticmethod
    def from_file(filename, missing_values=None):
        """Load annotations from a file.

        The file is a text file with a columns separated by spaces and/or
        commas, and rows on different lines.

        Arguments
        ---------
        filename : string
            File name

        missing_values : list
            List of labels that are considered missing values.
            Default is :attr:`DEFAULT_MISSING_VALUES_STR`
        """

        if missing_values is None:
            missing_values = AnnotationsContainer.DEFAULT_MISSING_VALUES_STR

        with open(filename) as fh:
            anno = AnnotationsContainer._from_file_object(fh,
                                                 missing_values=missing_values,
                                                 name=filename)

        return anno


    @staticmethod
    def from_array(x, missing_values=None, name=''):
        """Create an annotations object from an array or list-of-lists.

        Arguments
        ---------
        x : ndarray or list-of-lists
            Array or list-of-lists containing numerical or string annotations

        missing_values : list
            List of values that are considered missing values.
            Default is :attr:`DEFAULT_MISSING_VALUES_ALL`

        name : string
            Name of the annotations (for user interaction and used as key in
            databases).
        """

        if missing_values is None:
            missing_values = AnnotationsContainer.DEFAULT_MISSING_VALUES_ALL

        # generator for array objects
        def array_rows_generator():
            for row in x:
                yield list(row)

        return AnnotationsContainer._from_generator(array_rows_generator(),
                                           missing_values, name=name)


    def save_to(self, filename, set_name=False):
        """Save raw annotations to file.

        Arguments
        ---------
        filename : string
            File name

        set_name : bool
            Set the :attr:`name` of the annotation container to the file name
        """
        if set_name:
            self.name = filename
        with open(filename, 'w') as f:
            f.writelines(
                (' '.join(map(str, row))+'\n'
                 for row in self.raw_annotations)
            )


def load_annotations(filename, missing_values=None):
    """Load annotations from file.

    The file is a text file with a columns separated by spaces and/or
    commas, and rows on different lines.

    Arguments
    ---------
    filename : string
        File name

    missing_values : list
       List of labels that are considered missing values.
       Default is
       :attr:`~pyanno.AnnotationsContainer.DEFAULT_MISSING_VALUES_STR`

    """
    anno = AnnotationsContainer.from_file(filename,
                                          missing_values=missing_values)
    return anno.annotations
