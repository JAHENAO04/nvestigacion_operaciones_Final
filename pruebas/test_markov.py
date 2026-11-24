"""Pruebas unitarias para el motor de Cadenas de Markov (modulos/markov.py)."""

import sys
import unittest
from pathlib import Path

import numpy as np

# Permite ejecutar el archivo directamente desde cualquier carpeta
RAIZ_PROYECTO = Path(__file__).resolve().parent.parent
if str(RAIZ_PROYECTO) not in sys.path:
    sys.path.insert(0, str(RAIZ_PROYECTO))

from modulos.markov import analizar_cadena


class TestMotorMarkov(unittest.TestCase):
    """Casos de verificación definidos en el plan de trabajo."""

    def setUp(self) -> None:
        self.matriz_base = [[0.9, 0.1], [0.5, 0.5]]
        self.estado_estable_esperado = (5 / 6, 1 / 6)

    def test_1_proyeccion_paso_a_paso(self) -> None:
        """Verifica pi_1 = (0.9, 0.1) y pi_2 = (0.86, 0.14) desde pi_0 = (1, 0)."""
        resultado = analizar_cadena(self.matriz_base, [1.0, 0.0], pasos=2)

        self.assertTrue(resultado.valido)
        self.assertIsNone(resultado.error)
        self.assertEqual(len(resultado.historial), 3)

        paso_1, pi_1 = resultado.historial[1]
        paso_2, pi_2 = resultado.historial[2]

        self.assertEqual(paso_1, 1)
        self.assertEqual(paso_2, 2)
        np.testing.assert_allclose(pi_1, (0.9, 0.1), atol=1e-6)
        np.testing.assert_allclose(pi_2, (0.86, 0.14), atol=1e-6)
        np.testing.assert_allclose(resultado.pi_n, (0.86, 0.14), atol=1e-6)

    def test_2_largo_plazo_estado_estable(self) -> None:
        """Verifica que el estado estable sea (5/6, 1/6) ≈ (0.8333, 0.1667)."""
        resultado = analizar_cadena(self.matriz_base, [1.0, 0.0], pasos=5)

        self.assertTrue(resultado.valido)
        self.assertTrue(resultado.es_regular)
        self.assertEqual(resultado.potencia_regular, 1)
        np.testing.assert_allclose(
            resultado.estado_estable, self.estado_estable_esperado, atol=1e-6
        )

    def test_3_independencia_del_inicio(self) -> None:
        """Cambiar pi_0 a (0, 1) debe producir el mismo estado estable."""
        resultado_a = analizar_cadena(self.matriz_base, [1.0, 0.0], pasos=5)
        resultado_b = analizar_cadena(self.matriz_base, [0.0, 1.0], pasos=5)

        self.assertTrue(resultado_a.valido)
        self.assertTrue(resultado_b.valido)
        np.testing.assert_allclose(
            resultado_a.estado_estable, resultado_b.estado_estable, atol=1e-6
        )
        np.testing.assert_allclose(
            resultado_b.estado_estable, self.estado_estable_esperado, atol=1e-6
        )

    def test_4_validacion_fila_invalida(self) -> None:
        """Una fila que suma 0.9 debe devolver error y no calcular resultados."""
        matriz_invalida = [[0.8, 0.1], [0.5, 0.5]]
        resultado = analizar_cadena(matriz_invalida, [1.0, 0.0], pasos=2)

        self.assertFalse(resultado.valido)
        self.assertIsNotNone(resultado.error)
        self.assertIn("sumar 1", resultado.error.lower())
        self.assertEqual(resultado.historial, ())
        self.assertIsNone(resultado.pi_n)
        self.assertIsNone(resultado.estado_estable)

    def test_5_no_ergodicidad_matriz_identidad(self) -> None:
        """La matriz identidad 2x2 no es regular/ergódica y debe avisarlo."""
        identidad = [[1.0, 0.0], [0.0, 1.0]]
        resultado = analizar_cadena(identidad, [0.5, 0.5], pasos=3)

        self.assertTrue(resultado.valido)
        self.assertFalse(resultado.es_regular)
        self.assertIsNone(resultado.potencia_regular)
        self.assertIn("no es regular", resultado.mensaje.lower())


if __name__ == "__main__":
    unittest.main(verbosity=2)
