# Copyright (c) 2011, Enthought, Ltd.
# Author: Pietro Berkes <pberkes@enthought.com>
# License: Modified BSD license (2-clause)
from contextlib import closing
import os
from shelve import Shelf
import shelve
from tempfile import TemporaryFile, tempdir, mktemp, gettempdir

import unittest
from pyanno.models import ModelA, ModelB
from pyanno.annotations import AnnotationsContainer
from pyanno.database import PyannoDatabase, PyannoResult
from pyanno.util import PyannoValueError
import numpy as np


class TestDatabase(unittest.TestCase):

    def setUp(self):
        self.tmp_filename = mktemp(prefix='tmp_pyanno_db_')

        # fixtures
        self.model1 = ModelA.create_initial_state(4)
        self.annotations1 = self.model1.generate_annotations(100)
        self.value1 = self.model1.log_likelihood(self.annotations1)
        self.anno_container1 = AnnotationsContainer.from_array(
            self.annotations1)
        self.data_id1 = 'bogus.txt'

        self.model2 = ModelB.create_initial_state(4, 8)
        self.annotations2 = self.model2.generate_annotations(100)
        self.value2 = self.model2.log_likelihood(self.annotations2)
        self.anno_container2 = AnnotationsContainer.from_array(
            self.annotations2)
        self.data_id2 = 'bogus2.txt'



    def tearDown(self):
        # remove temp file
        try:
            os.remove(self.tmp_filename)
        except OSError:
            # ignore, the file was not created by the test
            pass


    def test_database_create_and_close(self):
        with closing(PyannoDatabase(self.tmp_filename)) as db:
            self.assertEqual(len(db.database), 0)
            self.assertEqual(db.db_filename, self.tmp_filename)
            self.assertFalse(db.closed)

        self.assertTrue(db.closed)

        # make sure database is closed
        with self.assertRaisesRegex(ValueError, 'closed shelf'):
            db.database['a'] = 2


    def test_storage(self):
        with closing(PyannoDatabase(self.tmp_filename)) as db:
            db.store_result(self.data_id1, self.anno_container1,
                            self.model1, self.value1)
            self.assertEqual(len(db.database), 1)

            results = db.retrieve_id(self.data_id1)
            self.assertTrue(isinstance(results, list))
            self.assertEqual(len(results), 1)
            self.assertTrue(isinstance(results[0], PyannoResult))

            np.testing.assert_equal(results[0].anno_container.annotations,
                                    self.anno_container1.annotations)
            self.assertEqual(results[0].model.nclasses, self.model1.nclasses)
            self.assertEqual(results[0].value, self.value1)
            self.assertTrue(isinstance(results[0].model, self.model1.__class__))

            # store new entry for different annotations
            db.store_result(self.data_id2, self.anno_container2,
                            self.model2, self.value2)
            self.assertEqual(len(db.database), 2)

            # add new result for same annotations
            db.store_result(self.data_id1, self.anno_container1,
                            self.model2, self.value2)
            results = db.retrieve_id(self.data_id1)
            self.assertEqual(len(results), 2)
            self.assertEqual(results[1].value, self.value2)


    def test_conflict(self):
        # behavior: one cannot store two different annotations with same id
        with closing(PyannoDatabase(self.tmp_filename)) as db:
            db.store_result(self.data_id1, self.anno_container1,
                            self.model1, self.value1)

            with self.assertRaises(PyannoValueError):
                db.store_result(self.data_id1, self.anno_container2,
                                self.model1, self.value1)


    def test_persistence(self):
        with closing(PyannoDatabase(self.tmp_filename)) as db:
            db.store_result(self.data_id1, self.anno_container1,
                            self.model1, self.value1)

        with closing(PyannoDatabase(self.tmp_filename)) as db:
            results = db.retrieve_id(self.data_id1)
            np.testing.assert_equal(results[0].anno_container.annotations,
                                    self.anno_container1.annotations)


    def test_remove(self):
        # behavior: removing one item makes the item disappear,
        # the rest of the database is intact
        # deletion should be indexed by data id and index in the list of
        # entries with the same if

        with closing(PyannoDatabase(self.tmp_filename)) as db:
            db.store_result(self.data_id1, self.anno_container1,
                            self.model1, self.value1)
            db.store_result(self.data_id1, self.anno_container1,
                            self.model2, self.value2)
            db.store_result(self.data_id2, self.anno_container2,
                            self.model2, self.value2)

            db.remove(self.data_id1, 1)

            self.assertEqual(len(db.database), 2)
            self.assertEqual(len(db.database[self.data_id1]), 1)
            self.assertEqual(len(db.database[self.data_id2]), 1)

            self.assertTrue(isinstance(db.database[self.data_id1][0].model,
                                       self.model1.__class__))


    def test_tmp_dataname(self):
        # behavior: the method get_available_id should return a new ID of the
        # form <name_N>, that is not yet present in the database

        with closing(PyannoDatabase(self.tmp_filename)) as db:
            id1 = db.get_available_id()
            self.assertEqual(id1, '<name_0>')
            self.assertTrue(id1 not in db.database)
            db.store_result(id1, self.anno_container1,
                            self.model1, self.value1)

            id2 = db.get_available_id()
            self.assertNotEqual(id1, id2)
            self.assertTrue(id2 not in db.database)


if __name__ == '__main__':
    unittest.main()