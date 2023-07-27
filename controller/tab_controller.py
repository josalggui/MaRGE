from widgets.tab_widget import TabWidget


class TabController(TabWidget):
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
        super(TabController, self).__init__(*args, **kwargs)
