# Copyright (c) 2011, Enthought, Ltd.
# Author: Pietro Berkes <pberkes@enthought.com>
# License: Modified BSD license (2-clause)

"""Defines a database object to store model results."""

import shelve
from traits.has_traits import HasStrictTraits
from traits.trait_types import Float, Instance
from pyanno.abstract_model import AbstractModel
from pyanno.annotations import AnnotationsContainer
from pyanno.util import PyannoValueError
import numpy as np

class PyannoResult(HasStrictTraits):
    """Class for database entries
    """

    #: :class:`~pyanno.annotations.AnnotationsContainer` object
    anno_container = Instance(AnnotationsContainer)

    #: pyAnno model (subclass of :class:`~pyanno.abstract_model.AbstractModel`)
    model = Instance(AbstractModel)

    #: value of the model performance (usually the log likelihood)
    value = Float


class PyannoDatabase(object):
    """Database to store model results.

    The database is based on :mod:`shelve`. Keys are strings that uniquely
    identify data sets. Values are lists of :class:`PyannoResult` objects,
    which contain a copy of the annotations, the pyanno model that has
    been applied on them, and the value of the log likelihood of the
    annotations given the model.
    """

    def __init__(self, filename):
        self.db_filename = filename

        #: `shelve` database storing the models
        self.database = shelve.open(filename, flag='c', protocol=2)

        #: True if the database is closed
        self.closed = False


    def store_result(self, data_id, anno_container, model, value):
        """Store a pyAnno result in the database.

        The `data_id` must be a **unique** identifier for an annotations
        set.

        Arguments
        ---------
        data_id : string
            Readable **unique** identifier for the annotations set (e.g.,
            the file name where the annotations are stored).

        anno_container : AnnotationsContainer
            An annotations container
            (see :class:`~pyanno.annotations.AnnotationsContainer`).

        model : object
            pyAnno model object instance
            (subclass of :class:`~pyanno.abstract_model.AbstractModel`

        value : float
            Value of the objective function for the model-annotations pair,
            typically the log likelihood of the annotations given the model
        """

        entry = PyannoResult(anno_container=anno_container,
                             model=model, value=value)
        self._check_consistency(data_id, anno_container)

        # NOTE shelves to not automatically handle changing mutable values,
        # we need to take care of it manually
        if data_id not in self.database:
            temp = []
        else:
            temp = self.database[data_id]

        temp.append(entry)

        self.database[data_id] = temp


    def retrieve_id(self, data_id):
        """Return all entries with given data ID.

        Arguments
        ---------
        data_id : string
            Readable **unique** identifier for the annotations set
        """
        return self.database[data_id]


    def remove(self, data_id, idx):
        """Remove entry from database.

        Arguments
        ---------
        data_id : string
            Readable **unique** identifier for the annotations set

        idx : int
            Index in the list of entries with id `data_id`
        """
        temp = self.database[data_id]
        del temp[idx]
        self.database[data_id] = temp


    def close(self):
        """Close database."""
        self.database.close()
        self.closed = True


    def get_available_id(self):
        """Return an data ID that has is not present in the database.

        The returned IDs have the form "<new_data_N>", where N is an integer
        number.
        """
        n = 0
        while True:
            id = '<name_{}>'.format(n)
            if id not in self.database:
                break
            n += 1
        return id


    def _check_consistency(self, data_id, anno_container):
        """Make sure that all entries with same ID have the same annotations.
        """
        if data_id in self.database:
            previous = self.database[data_id]
            if len(previous) > 0:
                # check if the new annotations are the same as the previous
                if not np.all(previous[0].anno_container.annotations ==
                              anno_container.annotations):
                    msg = ('Conflicting annotations with same ID. Please '
                           'rename the new entry.')
                    raise PyannoValueError(msg)
