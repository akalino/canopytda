"""Importable finite complexes, filtrations, and order fixtures.

The L complex and its named simplices follow Examples 2.17 and 2.19 of
Giunti and Munch, *Canopies* (arXiv:2606.23859). The two disjoint edges
follow Example 3.16. Other small complexes exercise basic homology cases.

Every filtration is one face-monotone function on a fixed complex. The
complex and filtration objects can be imported directly by tests or demos.
"""

from canopytda.complex import SimplicialComplex
from canopytda.filtration import Filtration


__all__ = (
    "SINGLE_EDGE_COMPLEX", "SINGLE_EDGE_FILTRATION",
    "SINGLE_EDGE_ORDER_ONE", "SINGLE_EDGE_ORDER_TWO",
    "SINGLE_EDGE_PAIR_ONE", "SINGLE_EDGE_PAIR_TWO",
    "FILLED_TRIANGLE_COMPLEX", "FILLED_TRIANGLE_FILTRATION",
    "LOOP_KILLED_BY_TRIANGLE_COMPLEX", "LOOP_KILLED_BY_TRIANGLE_FILTRATION",
    "ESSENTIAL_LOOP_COMPLEX", "ESSENTIAL_LOOP_FILTRATION",
    "ESSENTIAL_LOOP_BIRTH_SIMPLEX",
    "L_COMPLEX", "A_SIMPLEX", "B_SIMPLEX", "C_SIMPLEX", "D_SIMPLEX",
    "L_NAMED_SIMPLICES", "L_SIMPLEX_NAMES", "L_EARLY_SIMPLICES",
    "L_F_FILTRATION", "L_G_FILTRATION",
    "L_G_ORDER_ONE", "L_G_ORDER_TWO",
    "L_G_PAIRS_ONE", "L_G_PAIRS_TWO",
    "PAPER_3_7_PARAMETERS", "PAPER_3_7_NAMED_VALUES",
    "PAPER_3_7_FILTRATIONS",
    "D_DIAGONAL_COMPLEX", "D_DIAGONAL_PARAMETERS",
    "D_DIAGONAL_FILTRATIONS", "D_FINITE_PAIR", "D_ALWAYS_DIAGONAL_PAIR",
    "TWO_DISJOINT_EDGES_COMPLEX", "TWO_DISJOINT_EDGES_FILTRATION",
    "TWO_DISJOINT_EDGES_PAIRS",
    "DISCONNECTED_VERTICES_COMPLEX", "DISCONNECTED_VERTICES_FILTRATION",
    "SQUARE_LOOP_COMPLEX", "SQUARE_LOOP_FILTRATION",
    "TETRAHEDRON_BOUNDARY_COMPLEX", "TETRAHEDRON_BOUNDARY_FILTRATION",
    "FILLED_TETRAHEDRON_COMPLEX", "FILLED_TETRAHEDRON_FILTRATION",
    "PAPER_COMPLEXES", "PAPER_FILTRATIONS",
    "ADDITIONAL_COMPLEXES", "ADDITIONAL_FILTRATIONS",
)


# One edge; equal vertex values allow two consistent simplex orders.
SINGLE_EDGE_COMPLEX = SimplicialComplex(((0, 1),))
SINGLE_EDGE_FILTRATION = Filtration(
    SINGLE_EDGE_COMPLEX,
    {(0,): 0.0, (1,): 0.0, (0, 1): 1.0},
)
SINGLE_EDGE_ORDER_ONE = ((0,), (1,), (0, 1))
SINGLE_EDGE_ORDER_TWO = ((1,), (0,), (0, 1))
SINGLE_EDGE_PAIR_ONE = ((1,), (0, 1))
SINGLE_EDGE_PAIR_TWO = ((0,), (0, 1))


# Filling at the same value as the edges gives a zero-length H1 interval.
FILLED_TRIANGLE_COMPLEX = SimplicialComplex(((0, 1, 2),))
FILLED_TRIANGLE_FILTRATION = Filtration(
    FILLED_TRIANGLE_COMPLEX,
    {
        (0,): 0.0, (1,): 0.0, (2,): 0.0,
        (0, 1): 1.0, (0, 2): 1.0, (1, 2): 1.0,
        (0, 1, 2): 1.0,
    },
)


# The same complex with a visible H1 interval from 1 to 2.
LOOP_KILLED_BY_TRIANGLE_COMPLEX = FILLED_TRIANGLE_COMPLEX
LOOP_KILLED_BY_TRIANGLE_FILTRATION = Filtration(
    LOOP_KILLED_BY_TRIANGLE_COMPLEX,
    {
        (0,): 0.0, (1,): 0.0, (2,): 0.0,
        (0, 1): 1.0, (0, 2): 1.0, (1, 2): 1.0,
        (0, 1, 2): 2.0,
    },
)


# The triangle boundary has an essential H1 class (paired with star).
ESSENTIAL_LOOP_COMPLEX = SimplicialComplex(
    ((0, 1), (0, 2), (1, 2))
)
ESSENTIAL_LOOP_FILTRATION = Filtration(
    ESSENTIAL_LOOP_COMPLEX,
    {
        (0,): 0.0, (1,): 0.0, (2,): 0.0,
        (0, 1): 1.0, (0, 2): 1.0, (1, 2): 1.0,
    },
)
ESSENTIAL_LOOP_BIRTH_SIMPLEX = (1, 2)


# The L complex from the paper: a and b are edges; c and d are triangles.
# SimplicialComplex automatically includes the vertices and missing faces.
A_SIMPLEX = (0, 2)
B_SIMPLEX = (1, 2)
C_SIMPLEX = (0, 1, 2)
D_SIMPLEX = (1, 2, 3)
L_NAMED_SIMPLICES = (A_SIMPLEX, B_SIMPLEX, C_SIMPLEX, D_SIMPLEX)
L_SIMPLEX_NAMES = {
    A_SIMPLEX: "a", B_SIMPLEX: "b",
    C_SIMPLEX: "c", D_SIMPLEX: "d",
}
L_COMPLEX = SimplicialComplex((C_SIMPLEX, D_SIMPLEX))
L_EARLY_SIMPLICES = tuple(
    _simplex for _simplex in L_COMPLEX.simplices
    if _simplex not in L_NAMED_SIMPLICES
)

# The paper's f and g in Figure 1: every other vertex and edge has value 0.
L_F_FILTRATION = Filtration(
    L_COMPLEX,
    {
        **{_simplex: 0.0 for _simplex in L_EARLY_SIMPLICES},
        A_SIMPLEX: 1.0, B_SIMPLEX: 1.0,
        C_SIMPLEX: 2.0, D_SIMPLEX: 2.0,
    },
)
L_G_FILTRATION = Filtration(
    L_COMPLEX,
    {
        **{_simplex: 0.0 for _simplex in L_EARLY_SIMPLICES},
        A_SIMPLEX: 1.0, B_SIMPLEX: 2.0,
        C_SIMPLEX: 3.0, D_SIMPLEX: 3.0,
    },
)

# Both total orders refine g. Example 2.19 gives different simplex pairings.
L_G_ORDER_ONE = L_EARLY_SIMPLICES + (
    A_SIMPLEX, B_SIMPLEX, C_SIMPLEX, D_SIMPLEX
)
L_G_ORDER_TWO = L_EARLY_SIMPLICES + (
    A_SIMPLEX, B_SIMPLEX, D_SIMPLEX, C_SIMPLEX
)
L_G_PAIRS_ONE = ((A_SIMPLEX, D_SIMPLEX), (B_SIMPLEX, C_SIMPLEX))
L_G_PAIRS_TWO = ((A_SIMPLEX, C_SIMPLEX), (B_SIMPLEX, D_SIMPLEX))


# The piecewise-linear L example of Figures 4--6 / Examples 3.7 and 3.12.
# All unnamed simplices stay earlier than the named edges and triangles.
PAPER_3_7_PARAMETERS = (0.0, 1.0 / 3.0, 3.0 / 4.0, 1.0)
PAPER_3_7_NAMED_VALUES = (
    {A_SIMPLEX: 0.0, B_SIMPLEX: 1.0,
     C_SIMPLEX: 2.0, D_SIMPLEX: 3.0},
    {A_SIMPLEX: 2.0 / 3.0, B_SIMPLEX: 2.0 / 3.0,
     C_SIMPLEX: 7.0 / 3.0, D_SIMPLEX: 7.0 / 3.0},
    {A_SIMPLEX: 3.0 / 2.0, B_SIMPLEX: 1.0 / 4.0,
     C_SIMPLEX: 11.0 / 4.0, D_SIMPLEX: 3.0 / 2.0},
    {A_SIMPLEX: 2.0, B_SIMPLEX: 0.0,
     C_SIMPLEX: 3.0, D_SIMPLEX: 1.0},
)
PAPER_3_7_FILTRATIONS = tuple(
    Filtration(
        L_COMPLEX,
        {**{_simplex: -1.0 for _simplex in L_EARLY_SIMPLICES},
         **_named_values},
    )
    for _named_values in PAPER_3_7_NAMED_VALUES
)


# D-canopy check: one H0 finite pair touches the diagonal at t=1/2;
# another stays on the diagonal and disappears from the D-total space.
# All three filtrations share the same simplex order and fixed complex.
D_DIAGONAL_COMPLEX = SimplicialComplex(((0, 1), (2, 3)))
D_DIAGONAL_PARAMETERS = (0.0, 0.5, 1.0)
D_DIAGONAL_FILTRATIONS = tuple(
    Filtration(D_DIAGONAL_COMPLEX, {
        (0,): 0.0, (1,): 0.0, (0, 1): _gap,
        (2,): 2.0, (3,): 2.0, (2, 3): 2.0,
    })
    for _gap in (1.0, 0.0, 1.0)
)
D_FINITE_PAIR = ((1,), (0, 1))
D_ALWAYS_DIAGONAL_PAIR = ((3,), (2, 3))


# Example 3.16: two unrelated H0 pairs with identical (birth, death) values.
TWO_DISJOINT_EDGES_COMPLEX = SimplicialComplex(((0, 1), (2, 3)))
TWO_DISJOINT_EDGES_FILTRATION = Filtration(
    TWO_DISJOINT_EDGES_COMPLEX,
    {
        (0,): 0.0, (1,): 1.0, (0, 1): 2.0,
        (2,): 0.0, (3,): 1.0, (2, 3): 2.0,
    },
)
TWO_DISJOINT_EDGES_PAIRS = (
    ((1,), (0, 1)),
    ((3,), (2, 3)),
)


# Additional examples: disconnected components, a longer cycle, and H2.
DISCONNECTED_VERTICES_COMPLEX = SimplicialComplex(((0,), (1,), (2,)))
DISCONNECTED_VERTICES_FILTRATION = Filtration(
    DISCONNECTED_VERTICES_COMPLEX,
    {(0,): 0.0, (1,): 1.0, (2,): 2.0},
)

SQUARE_LOOP_COMPLEX = SimplicialComplex(
    ((0, 1), (1, 2), (2, 3), (0, 3))
)
SQUARE_LOOP_FILTRATION = Filtration(
    SQUARE_LOOP_COMPLEX,
    {
        (0,): 0.0, (1,): 0.0, (2,): 0.0, (3,): 0.0,
        (0, 1): 1.0, (1, 2): 1.0,
        (2, 3): 1.0, (0, 3): 1.0,
    },
)

TETRAHEDRON_BOUNDARY_COMPLEX = SimplicialComplex(
    ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))
)
TETRAHEDRON_BOUNDARY_FILTRATION = Filtration(
    TETRAHEDRON_BOUNDARY_COMPLEX,
    {_simplex: float(len(_simplex) - 1)
     for _simplex in TETRAHEDRON_BOUNDARY_COMPLEX.simplices},
)

FILLED_TETRAHEDRON_COMPLEX = SimplicialComplex(((0, 1, 2, 3),))
FILLED_TETRAHEDRON_FILTRATION = Filtration(
    FILLED_TETRAHEDRON_COMPLEX,
    {_simplex: float(len(_simplex) - 1)
     for _simplex in FILLED_TETRAHEDRON_COMPLEX.simplices},
)


PAPER_COMPLEXES = {
    "single_edge": SINGLE_EDGE_COMPLEX,
    "filled_triangle": FILLED_TRIANGLE_COMPLEX,
    "loop_killed_by_triangle": LOOP_KILLED_BY_TRIANGLE_COMPLEX,
    "essential_class_paired_with_star": ESSENTIAL_LOOP_COMPLEX,
    "l_complex": L_COMPLEX,
    "two_disjoint_edges": TWO_DISJOINT_EDGES_COMPLEX,
}
PAPER_FILTRATIONS = {
    "single_edge": SINGLE_EDGE_FILTRATION,
    "filled_triangle": FILLED_TRIANGLE_FILTRATION,
    "loop_killed_by_triangle": LOOP_KILLED_BY_TRIANGLE_FILTRATION,
    "essential_class_paired_with_star": ESSENTIAL_LOOP_FILTRATION,
    "l_f": L_F_FILTRATION,
    "l_g": L_G_FILTRATION,
    "two_disjoint_edges": TWO_DISJOINT_EDGES_FILTRATION,
}
ADDITIONAL_COMPLEXES = {
    "disconnected_vertices": DISCONNECTED_VERTICES_COMPLEX,
    "square_loop": SQUARE_LOOP_COMPLEX,
    "tetrahedron_boundary": TETRAHEDRON_BOUNDARY_COMPLEX,
    "filled_tetrahedron": FILLED_TETRAHEDRON_COMPLEX,
}
ADDITIONAL_FILTRATIONS = {
    "disconnected_vertices": DISCONNECTED_VERTICES_FILTRATION,
    "square_loop": SQUARE_LOOP_FILTRATION,
    "tetrahedron_boundary": TETRAHEDRON_BOUNDARY_FILTRATION,
    "filled_tetrahedron": FILLED_TETRAHEDRON_FILTRATION,
}
