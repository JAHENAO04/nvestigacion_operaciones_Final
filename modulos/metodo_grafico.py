"""Motor puro para problemas de programación lineal de dos variables."""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, isfinite, pi, sin
from typing import Iterable, Literal

Signo = Literal["<=", ">=", "="]
Restriccion = tuple[float, float, Signo, float]
TOLERANCIA = 1e-7


@dataclass(frozen=True)
class EvaluacionVertice:
    """Coordenadas y valor de la función objetivo en un punto factible."""

    x1: float
    x2: float
    z: float


@dataclass(frozen=True)
class SegmentoRecta:
    """Puntos que permiten dibujar una recta en el plano."""

    etiqueta: str
    x: tuple[float, float]
    y: tuple[float, float]


@dataclass(frozen=True)
class ResultadoMetodoGrafico:
    """Resultado completo del análisis gráfico y su estado de resolución."""

    estado: str
    objetivo: str
    c1: float
    c2: float
    restricciones: tuple[Restriccion, ...]
    no_negatividad: bool
    vertices: tuple[EvaluacionVertice, ...]
    region_ordenada: tuple[tuple[float, float], ...]
    segmentos_rectas: tuple[SegmentoRecta, ...]
    limites_grafico: tuple[float, float, float, float]
    region_acotada: bool
    punto_optimo: EvaluacionVertice | None
    valor_optimo: float | None
    error: str | None
    mensaje: str


def _crear_resultado_error(
    objetivo: str,
    c1: float,
    c2: float,
    restricciones: tuple[Restriccion, ...],
    no_negatividad: bool,
    mensaje: str,
) -> ResultadoMetodoGrafico:
    """Construye una respuesta tipada cuando los datos de entrada no son válidos."""
    return ResultadoMetodoGrafico(
        estado="error",
        objetivo=objetivo,
        c1=c1,
        c2=c2,
        restricciones=restricciones,
        no_negatividad=no_negatividad,
        vertices=(),
        region_ordenada=(),
        segmentos_rectas=(),
        limites_grafico=(-10.0, 10.0, -10.0, 10.0),
        region_acotada=False,
        punto_optimo=None,
        valor_optimo=None,
        error=mensaje,
        mensaje="No se pudo analizar el modelo.",
    )


def _es_factible(
    x1: float,
    x2: float,
    restricciones: tuple[Restriccion, ...],
    no_negatividad: bool,
) -> bool:
    if no_negatividad and (x1 < -TOLERANCIA or x2 < -TOLERANCIA):
        return False
    for a1, a2, signo, b in restricciones:
        diferencia = a1 * x1 + a2 * x2 - b
        if signo == "<=" and diferencia > TOLERANCIA:
            return False
        if signo == ">=" and diferencia < -TOLERANCIA:
            return False
        if signo == "=" and abs(diferencia) > TOLERANCIA:
            return False
    return True


def _agregar_punto(
    puntos: list[tuple[float, float]], punto: tuple[float, float]
) -> None:
    if not all(isfinite(coordenada) for coordenada in punto):
        return
    if not any(
        abs(punto[0] - existente[0]) <= TOLERANCIA
        and abs(punto[1] - existente[1]) <= TOLERANCIA
        for existente in puntos
    ):
        puntos.append(punto)


def _interseccion(
    primera: tuple[float, float, float],
    segunda: tuple[float, float, float],
) -> tuple[float, float] | None:
    a1, b1, c1 = primera
    a2, b2, c2 = segunda
    determinante = a1 * b2 - a2 * b1
    if abs(determinante) <= TOLERANCIA:
        return None
    return (
        (c1 * b2 - c2 * b1) / determinante,
        (a1 * c2 - a2 * c1) / determinante,
    )


def _puntos_factibles(
    restricciones: tuple[Restriccion, ...], no_negatividad: bool
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    rectas = [(a1, a2, b) for a1, a2, _, b in restricciones]
    if no_negatividad:
        rectas.extend(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)))

    vertices: list[tuple[float, float]] = []
    for indice, primera in enumerate(rectas):
        for segunda in rectas[indice + 1 :]:
            punto = _interseccion(primera, segunda)
            if punto is not None and _es_factible(
                punto[0], punto[1], restricciones, no_negatividad
            ):
                _agregar_punto(vertices, punto)

    # Las rectas paralelas y las regiones sin vértices aún pueden ser factibles.
    pruebas = [(0.0, 0.0)]
    for a1, a2, _, b in restricciones:
        norma_cuadrada = a1 * a1 + a2 * a2
        if norma_cuadrada > TOLERANCIA:
            pie = (a1 * b / norma_cuadrada, a2 * b / norma_cuadrada)
            pruebas.append(pie)
            tangente = (-a2, a1)
            for escala in (1.0, 10.0, 100.0):
                pruebas.extend(
                    (
                        (pie[0] + tangente[0] * escala, pie[1] + tangente[1] * escala),
                        (pie[0] - tangente[0] * escala, pie[1] - tangente[1] * escala),
                    )
                )
    factibles = vertices.copy()
    for punto in pruebas:
        if _es_factible(punto[0], punto[1], restricciones, no_negatividad):
            _agregar_punto(factibles, punto)
    return vertices, factibles


def _direcciones_recesion(
    restricciones: tuple[Restriccion, ...], no_negatividad: bool
) -> list[tuple[float, float]]:
    """Enumera rayos factibles usando los sectores definidos por sus fronteras."""
    desigualdades: list[tuple[float, float]] = []
    for a1, a2, signo, _ in restricciones:
        if signo in ("<=", "="):
            desigualdades.append((a1, a2))
        if signo in (">=", "="):
            desigualdades.append((-a1, -a2))
    if no_negatividad:
        desigualdades.extend(((-1.0, 0.0), (0.0, -1.0)))

    angulos = [0.0, pi / 2, pi, 3 * pi / 2]
    for a1, a2 in desigualdades:
        if abs(a1) + abs(a2) > TOLERANCIA:
            angulo = atan2(-a1, a2)
            angulos.extend((angulo % (2 * pi), (angulo + pi) % (2 * pi)))
    angulos = sorted(set(round(angulo, 12) for angulo in angulos))

    candidatos = list(angulos)
    for indice, angulo in enumerate(angulos):
        siguiente = angulos[(indice + 1) % len(angulos)]
        if indice == len(angulos) - 1:
            siguiente += 2 * pi
        candidatos.append((angulo + siguiente) / 2 % (2 * pi))

    direcciones: list[tuple[float, float]] = []
    for angulo in candidatos:
        direccion = (cos(angulo), sin(angulo))
        if all(
            a1 * direccion[0] + a2 * direccion[1] <= TOLERANCIA
            for a1, a2 in desigualdades
        ):
            _agregar_punto(direcciones, direccion)
    return direcciones


def _calcular_limites(
    restricciones: tuple[Restriccion, ...],
    puntos: list[tuple[float, float]],
    no_negatividad: bool,
) -> tuple[float, float, float, float]:
    valores_x = [punto[0] for punto in puntos]
    valores_y = [punto[1] for punto in puntos]
    for a1, a2, _, b in restricciones:
        if abs(a1) > TOLERANCIA:
            valores_x.append(b / a1)
        if abs(a2) > TOLERANCIA:
            valores_y.append(b / a2)

    if no_negatividad:
        xmax = max((valor for valor in valores_x if valor > 0), default=8.0)
        ymax = max((valor for valor in valores_y if valor > 0), default=8.0)
        return (0.0, max(10.0, xmax * 1.2), 0.0, max(10.0, ymax * 1.2))

    def limites(valores: list[float]) -> tuple[float, float]:
        if not valores:
            return (-10.0, 10.0)
        minimo, maximo = min(valores), max(valores)
        margen = max((maximo - minimo) * 0.15, 2.0)
        return (minimo - margen, maximo + margen)

    xmin, xmax = limites(valores_x)
    ymin, ymax = limites(valores_y)
    return (xmin, xmax, ymin, ymax)


def _crear_segmentos(
    restricciones: tuple[Restriccion, ...],
    limites: tuple[float, float, float, float],
) -> tuple[SegmentoRecta, ...]:
    xmin, xmax, ymin, ymax = limites
    segmentos: list[SegmentoRecta] = []
    for indice, (a1, a2, signo, b) in enumerate(restricciones, start=1):
        if abs(a2) > TOLERANCIA:
            xs = (xmin, xmax)
            ys = ((b - a1 * xmin) / a2, (b - a1 * xmax) / a2)
        elif abs(a1) > TOLERANCIA:
            x = b / a1
            xs = (x, x)
            ys = (ymin, ymax)
        else:
            continue
        segmentos.append(
            SegmentoRecta(f"R{indice}: {signo}", xs, ys)
        )
    return tuple(segmentos)


def _region_recortada(
    restricciones: tuple[Restriccion, ...],
    limites: tuple[float, float, float, float],
    no_negatividad: bool,
) -> tuple[tuple[float, float], ...]:
    """Calcula el polígono factible visible, recortado al rectángulo del gráfico."""
    xmin, xmax, ymin, ymax = limites
    fronteras = [(a1, a2, b) for a1, a2, _, b in restricciones]
    fronteras.extend(
        (
            (1.0, 0.0, xmin),
            (1.0, 0.0, xmax),
            (0.0, 1.0, ymin),
            (0.0, 1.0, ymax),
        )
    )
    puntos: list[tuple[float, float]] = []
    for indice, primera in enumerate(fronteras):
        for segunda in fronteras[indice + 1 :]:
            punto = _interseccion(primera, segunda)
            if punto is None:
                continue
            x1, x2 = punto
            dentro_del_grafico = (
                xmin - TOLERANCIA <= x1 <= xmax + TOLERANCIA
                and ymin - TOLERANCIA <= x2 <= ymax + TOLERANCIA
            )
            if dentro_del_grafico and _es_factible(
                x1, x2, restricciones, no_negatividad
            ):
                _agregar_punto(puntos, punto)
    if len(puntos) < 3:
        return tuple(puntos)
    centro_x = sum(punto[0] for punto in puntos) / len(puntos)
    centro_y = sum(punto[1] for punto in puntos) / len(puntos)
    return tuple(
        sorted(
            puntos,
            key=lambda punto: atan2(punto[1] - centro_y, punto[0] - centro_x),
        )
    )


def resolver_programacion_lineal(
    objetivo: str,
    c1: float,
    c2: float,
    restricciones: Iterable[Restriccion],
    no_negatividad: bool = True,
) -> ResultadoMetodoGrafico:
    """Resuelve gráficamente un modelo lineal con dos variables.

    Cada restricción se expresa como ``(a1, a2, signo, b)`` para
    ``a1*x1 + a2*x2 signo b``. El resultado distingue errores de entrada,
    ausencia de región factible y objetivos no acotados.
    """
    try:
        coeficiente_1, coeficiente_2 = float(c1), float(c2)
        lista_restricciones = tuple(
            (float(a1), float(a2), signo, float(b))
            for a1, a2, signo, b in restricciones
        )
    except (TypeError, ValueError):
        return _crear_resultado_error(
            objetivo, 0.0, 0.0, (), no_negatividad, "Los coeficientes deben ser numéricos."
        )

    if objetivo not in ("maximizar", "minimizar"):
        return _crear_resultado_error(
            objetivo,
            coeficiente_1,
            coeficiente_2,
            lista_restricciones,
            no_negatividad,
            "El objetivo debe ser 'maximizar' o 'minimizar'.",
        )
    if not all(
        isfinite(valor)
        for valor in (
            coeficiente_1,
            coeficiente_2,
            *(valor for restriccion in lista_restricciones for valor in (restriccion[0], restriccion[1], restriccion[3])),
        )
    ):
        return _crear_resultado_error(
            objetivo,
            coeficiente_1,
            coeficiente_2,
            lista_restricciones,
            no_negatividad,
            "Todos los coeficientes deben ser finitos.",
        )
    if any(signo not in ("<=", ">=", "=") for _, _, signo, _ in lista_restricciones):
        return _crear_resultado_error(
            objetivo,
            coeficiente_1,
            coeficiente_2,
            lista_restricciones,
            no_negatividad,
            "Los signos permitidos son '<=', '>=' y '='.",
        )

    vertices, puntos_factibles = _puntos_factibles(
        lista_restricciones, no_negatividad
    )
    if not puntos_factibles:
        limites = _calcular_limites(lista_restricciones, [], no_negatividad)
        return ResultadoMetodoGrafico(
            estado="sin_solucion",
            objetivo=objetivo,
            c1=coeficiente_1,
            c2=coeficiente_2,
            restricciones=lista_restricciones,
            no_negatividad=no_negatividad,
            vertices=(),
            region_ordenada=(),
            segmentos_rectas=_crear_segmentos(lista_restricciones, limites),
            limites_grafico=limites,
            region_acotada=False,
            punto_optimo=None,
            valor_optimo=None,
            error=None,
            mensaje="Las restricciones son incompatibles: no existe región factible.",
        )

    direcciones = _direcciones_recesion(lista_restricciones, no_negatividad)
    region_acotada = not direcciones
    signo_mejora = 1.0 if objetivo == "maximizar" else -1.0
    no_acotado = any(
        signo_mejora * (coeficiente_1 * dx + coeficiente_2 * dy) > TOLERANCIA
        for dx, dy in direcciones
    )

    evaluaciones_vertices = tuple(
        EvaluacionVertice(x1, x2, coeficiente_1 * x1 + coeficiente_2 * x2)
        for x1, x2 in vertices
    )
    criterio = lambda punto: punto.z if objetivo == "maximizar" else -punto.z
    opciones = evaluaciones_vertices or tuple(
        EvaluacionVertice(x1, x2, coeficiente_1 * x1 + coeficiente_2 * x2)
        for x1, x2 in puntos_factibles
    )
    mejor = max(opciones, key=criterio)
    estado = "no_acotado" if no_acotado else "optimo"
    mensaje = (
        "La región factible existe, pero la función objetivo puede mejorar sin límite."
        if no_acotado
        else "Se encontró el valor óptimo entre los puntos factibles."
    )

    limites = _calcular_limites(
        lista_restricciones, puntos_factibles, no_negatividad
    )
    region_ordenada = _region_recortada(
        lista_restricciones, limites, no_negatividad
    )
    return ResultadoMetodoGrafico(
        estado=estado,
        objetivo=objetivo,
        c1=coeficiente_1,
        c2=coeficiente_2,
        restricciones=lista_restricciones,
        no_negatividad=no_negatividad,
        vertices=evaluaciones_vertices,
        region_ordenada=region_ordenada,
        segmentos_rectas=_crear_segmentos(lista_restricciones, limites),
        limites_grafico=limites,
        region_acotada=region_acotada,
        punto_optimo=mejor,
        valor_optimo=mejor.z,
        error=None,
        mensaje=mensaje,
    )
