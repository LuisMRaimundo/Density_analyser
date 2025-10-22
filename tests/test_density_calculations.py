import unittest
from density_calculations import analisar_densidade_completa

class TestDensityCalculations(unittest.TestCase):
    def test_analisar_densidade_completa_com_ponderacao(self):
        """
        Tests that analisar_densidade_completa runs without error when perceptual weighting is enabled.
        """
        c_major = [60, 64, 67]  # C4, E4, G4
        try:
            analisar_densidade_completa(c_major, usar_ponderacao_perceptual=True)
        except NameError:
            self.fail("analisar_densidade_completa raised NameError unexpectedly!")
