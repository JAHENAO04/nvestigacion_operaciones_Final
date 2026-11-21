"""Motor puro para proyecciones y análisis de cadenas de Markov finitas."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
TOLERANCIA_VALIDACION = 1e-6


@dataclass(frozen=True)
class ResultadoMarkov:
    """Validación, proyección, regularidad y estado estable de una cadena."""

    valido: bool
    error: str | None
    historial: tuple[tuple[int, tuple[float, ...]], ...]
    pi_n: tuple[float, ...] | None
    es_regular: bool | None
    potencia_regular: int | None
    estado_estable: tuple[float, ...] | None
    diferencia_final: tuple[float, ...] | None
    diferencia_maxima: float | None
    mensaje: str


def _respuesta_error(mensaje: str) -> ResultadoMarkov:
    return ResultadoMarkov(
        valido=False,
        error=mensaje,
        historial=(),
        pi_n=None,
        es_regular=None,
        potencia_regular=None,
        estado_estable=None,
        diferencia_final=None,
        diferencia_maxima=None,
        mensaje="La cadena no se calculó porque los datos no son válidos.",
    )


def _validar_entrada(
    matriz: object, distribucion_inicial: object, pasos: int
) -> tuple[FloatArray, FloatArray] | ResultadoMarkov:
    try:
        transicion = np.asarray(matriz, dtype=float)
        pi_0 = np.asarray(distribucion_inicial, dtype=float)
    except (TypeError, ValueError):
        return _respuesta_error("La matriz y la distribución deben contener números.")

    if transicion.ndim != 2 or transicion.shape[0] == 0:
        return _respuesta_error("La matriz de transición debe ser cuadrada y no vacía.")
    if transicion.shape[0] != transicion.shape[1]:
        return _respuesta_error("La matriz de transición debe ser cuadrada.")
    if not np.all(np.isfinite(transicion)):
        return _respuesta_error("La matriz de transición debe contener valores finitos.")
    if np.any(transicion < 0.0) or np.any(transicion > 1.0):
        return _respuesta_error("Cada probabilidad de transición debe estar entre 0 y 1.")
    if not np.allclose(
        transicion.sum(axis=1), 1.0, atol=TOLERANCIA_VALIDACION, rtol=0.0
    ):
        return _respuesta_error("Cada fila de la matriz de transición debe sumar 1.")
    if pi_0.ndim != 1 or pi_0.shape[0] != transicion.shape[0]:
        return _respuesta_error(
            "La distribución inicial debe tener una probabilidad por cada estado."
        )
    if not np.all(np.isfinite(pi_0)):
        return _respuesta_error("La distribución inicial debe contener valores finitos.")
    if np.any(pi_0 < 0.0) or np.any(pi_0 > 1.0):
        return _respuesta_error(
            "Cada probabilidad de la distribución inicial debe estar entre 0 y 1."
        )
    if not np.isclose(pi_0.sum(), 1.0, atol=TOLERANCIA_VALIDACION, rtol=0.0):
        return _respuesta_error("La distribución inicial debe sumar 1.")
    if isinstance(pasos, bool) or not isinstance(pasos, int) or pasos < 0:
        return _respuesta_error("El número de pasos debe ser un entero no negativo.")
    return transicion, pi_0


def _buscar_regularidad(matriz: FloatArray) -> int | None:
    """Retorna el primer exponente dentro del límite teórico que da P^k > 0."""
    cantidad_estados = matriz.shape[0]
    limite = (cantidad_estados - 1) ** 2 + 1
    potencia = np.array(matriz, dtype=float, copy=True)
    for exponente in range(1, limite + 1):
        if np.all(potencia > 0.0):
            return exponente
        potencia = potencia @ matriz
    return None


def _calcular_estado_estable(matriz: FloatArray) -> FloatArray:
    """Resuelve las ecuaciones estacionarias junto con la suma unitaria."""
    cantidad_estados = matriz.shape[0]
    sistema = np.vstack((matriz.T - np.eye(cantidad_estados), np.ones(cantidad_estados)))
    termino = np.concatenate((np.zeros(cantidad_estados), np.ones(1)))
    solucion, _, _, _ = np.linalg.lstsq(sistema, termino, rcond=None)

    if np.min(solucion) < -TOLERANCIA_VALIDACION:
        raise ArithmeticError("No se obtuvo una distribución estacionaria válida.")
    solucion = np.maximum(solucion, 0.0)
    suma = float(solucion.sum())
    if suma <= 0.0:
        raise ArithmeticError("No se obtuvo una distribución estacionaria válida.")
    solucion /= suma
    return solucion


def analizar_cadena(
    matriz: object, distribucion_inicial: object, pasos: int
) -> ResultadoMarkov:
    """Valida una cadena, proyecta ``pi_0`` y estima su distribución estable.

    La regularidad se verifica buscando una potencia estrictamente positiva
    desde ``k = 1`` hasta ``(m - 1)^2 + 1``.
    """
    validacion = _validar_entrada(matriz, distribucion_inicial, pasos)
    if isinstance(validacion, ResultadoMarkov):
        return validacion
    transicion, pi_0 = validacion

    historial: list[tuple[int, tuple[float, ...]]] = [
        (0, tuple(float(valor) for valor in pi_0))
    ]
    pi_actual = pi_0.copy()
    for indice in range(1, pasos + 1):
        pi_actual = pi_actual @ transicion
        historial.append((indice, tuple(float(valor) for valor in pi_actual)))

    exponente_regular = _buscar_regularidad(transicion)
    try:
        estable = _calcular_estado_estable(transicion)
    except np.linalg.LinAlgError:
        return _respuesta_error(
            "No fue posible resolver numéricamente el estado estacionario."
        )
    except ArithmeticError as error:
        return _respuesta_error(str(error))

    diferencia = pi_actual - estable
    es_regular = exponente_regular is not None
    mensaje = (
        f"La cadena es regular: P^{exponente_regular} tiene todas sus entradas positivas."
        if es_regular
        else (
            "La cadena no es regular dentro del límite analizado. "
            "Una cadena finita sí tiene al menos un estado estacionario, pero "
            "puede no ser único y la proyección puede no converger a él."
        )
    )
    return ResultadoMarkov(
        valido=True,
        error=None,
        historial=tuple(historial),
        pi_n=tuple(float(valor) for valor in pi_actual),
        es_regular=es_regular,
        potencia_regular=exponente_regular,
        estado_estable=tuple(float(valor) for valor in estable),
        diferencia_final=tuple(float(valor) for valor in diferencia),
        diferencia_maxima=float(np.max(np.abs(diferencia))),
        mensaje=mensaje,
    )
