import unittest
import tkinter as tk
from unittest.mock import Mock
from gui_components import DensityCalculatorGUI

class TestDensityCalculatorGUI(unittest.TestCase):
    def test_get_input_data_does_not_return_durations(self):
        root = tk.Tk()
        root.withdraw()
        callbacks = {
            'calculate': Mock(),
            'clear': Mock(),
            'generate_report': Mock(),
            'execute_validation': Mock()
        }
        app = DensityCalculatorGUI(root, callbacks)
        data = app.get_input_data()
        self.assertNotIn('durations', data)
        root.destroy()

if __name__ == '__main__':
    unittest.main()
