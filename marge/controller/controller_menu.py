"""
:author:    J.M. Algarín
:email:     josalggui@i3m.upv.es
:affiliation: MRILab, i3M, CSIC, Valencia, Spain

"""


class MenuController:
    """
    Menu controller class.

    This class is responsible for controlling the menus in the application. It adds menus to the main window and
    connects actions to them.

    Methods:
        __init__(self, main): Initialize the MenuController instance.

    Attributes:
        main: The main window instance.

    """
    def __init__(self, main):
        """
        Initialize the MenuController instance.

        This method initializes the MenuController instance by setting the `main` attribute to the provided `main`
        window instance. It adds menus to the main window and connects actions to them.

        Args:
            main: The main window instance.

        Returns:
            None
        """
        self.main = main

        # Add menus
        self.menu_scanner = self.main.menu.addMenu("Scanner")
        self.menu_protocols = self.main.menu.addMenu("Protocols")
        self.menu_sequences = self.main.menu.addMenu("Sequences")
        self.menu_session = self.main.menu.addMenu("Session")

        # Protocol menu
        self.menu_protocols.addAction(self.main.toolbar_protocols.action_new_protocol)
        self.menu_protocols.addAction(self.main.toolbar_protocols.action_del_protocol)
        self.menu_protocols.addAction(self.main.toolbar_protocols.action_new_sequence)
        self.menu_protocols.addAction(self.main.toolbar_protocols.action_del_sequence)

        # Scanner menu
        self.menu_scanner.addAction(self.main.toolbar_marcos.action_marcos_install)
        self.menu_scanner.addAction(self.main.toolbar_marcos.action_copybitstream)
        self.menu_scanner.addAction(self.main.toolbar_marcos.action_server)
        self.menu_scanner.addAction(self.main.toolbar_marcos.action_gpa_init)

        # Sequences menu
        self.menu_sequences.addAction(self.main.toolbar_sequences.action_load_parameters)
        self.menu_sequences.addAction(self.main.toolbar_sequences.action_save_parameters)
        self.menu_sequences.addAction(self.main.toolbar_sequences.action_save_parameters_cal)
        self.menu_sequences.addAction(self.main.toolbar_sequences.action_add_to_list)
        self.menu_sequences.addAction(self.main.toolbar_sequences.action_acquire)
        self.menu_sequences.addAction(self.main.toolbar_sequences.action_bender)
        self.menu_sequences.addAction(self.main.toolbar_sequences.action_view_sequence)
