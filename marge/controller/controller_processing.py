from marge.widgets.widget_processing import ProcessingWidget


class ProcessingController(ProcessingWidget):
    """
    Controller class for the TabWidget.

    Inherits from TabWidget to provide additional functionality for managing tabs.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the TabController.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super(ProcessingController, self).__init__(*args, **kwargs)
