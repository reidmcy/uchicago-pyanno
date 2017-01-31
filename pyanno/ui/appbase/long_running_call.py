# Copyright (c) 2011, Enthought, Ltd.
# Author: Mark Dickinson Martin Chilvers, and Ioannis Tziakos
# License: Modified BSD license (2-clause)


""" A long running job (aka operation) that requires a progress bar. """


from pyface.api import GUI
from traits.api import Any, Callable, Dict, HasTraits, Tuple, Unicode

from pyanno.ui.appbase.pulse_progress_dialog import PulseProgressDialog


class LongRunningCall(HasTraits):
    """ A long running call (aka operation) that requires a progress bar.

    fixme: wx specific!

    """

    #### 'LongRunningCall' protocol ###########################################

    # Positional arguments to pass to the callable.
    args = Tuple

    # The calllable that takes a long time!
    callable = Callable

    # Keyword arguments to pass to the callable.
    kw = Dict

    # The message shown in the body of the progress dialog.
    message = Unicode
    
    # A callable that is invoked if the long running job fails (i.e. raises
    # an exception). This callable must take a single argument which is the
    # exception that was raised.
    on_failure = Callable

    # A callable that is invoked if the long running job succeeds (i.e does
    # not raise an exception). This callable must take a single argument
    # which is the result returned by 'self.job'.
    on_success = Callable

    # The toolkit-specific control that is used as the parent of the progress
    # dialog (can be None).
    parent = Any
    
    # The title of the progress dialog.
    title = Unicode

    #### 'object' interface ###################################################

    def __call__(self):
        """ Make the call! """

        # Display the progress dialog (we do it here and NOT in the result
        # producer because the producer is NOT in the GUI thread!).
        self._open_progress_dialog()

        import wx.lib.delayedresult as delayedresult
        delayedresult.startWorker(self._result_consumer, self._result_producer)

        return

    #### Private protocol #####################################################
    
    def _open_progress_dialog(self):
        """ Open the progress dialog. """

        if self.parent is not None:
            # In wx the progress dialog isn't really modal, so we just disable
            # its parent!
            self.parent.Enable(False)

        progress_dialog_class = PulseProgressDialog
        self.progress_dialog = progress_dialog_class(
            parent=self.parent, title=self.title, message=self.message
        )
        self.progress_dialog.open()
        
        return
    
    def _close_progress_dialog(self):
        """ Close the progress dialog. """
        
        self.progress_dialog.close()

        if self.parent is not None:
            self.parent.Enable(True)
        
        return

    def _result_consumer(self, delayed_result):
        """ The delayed result consumer. """
        
        GUI.invoke_later(self._close_progress_dialog)
        try:
            result = delayed_result.get()
            GUI.invoke_later(self.on_success, result)
            
        except Exception as exc:
            GUI.invoke_later(self.on_failure, exc)
                
        return

    def _result_producer(self):
        """ The delayed result producer. """
        
        return self.callable(*self.args, **self.kw)


def main():
    # Slooow division! Divide by zero to see it fail!
    def divide(x, y): import time; time.sleep(5); return x / y
    def divide_succeeded(result): print('divide succeeded', result)
    def divide_failed(exc): print('divide failed', exc)
    
    from pyface.api import GUI

    gui = GUI()

    call = LongRunningCall(
        parent     = None,
        title      = 'Test',
        message    = 'Dividing...',
        callable   = divide,
        args       = (42, 6),
        on_success = divide_succeeded,
        on_failure = divide_failed,
    )
    call()

    gui.start_event_loop()

    return


if __name__ == '__main__':
    main()
    
#### EOF ######################################################################
