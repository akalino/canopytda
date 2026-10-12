"""Importable finite complexes, filtrations, and order fixtures.

The L complex and named simplices follow Examples 2.17 and 2.19 of Canopies.
The two disjoint edges follow Example 3.16.
Other small complexes exercise basic homology cases.

Every filtration is one face-monotone function on a fixed complex. The
complex and filtration objects can be imported directly by tests or demos.
"""
# pylint: disable=import-error
from canopytda.complex import SimplicialComplex
from canopytda.filtration import Filtration


# One edge and equal vertex values allow two consistent simplex orders.
SINGLE_EDGE_COMPLEX = SimplicialComplex(((0, 1),))
SINGLE_EDGE_FILTRATION = Filtration(
    SINGLE_EDGE_COMPLEX,
    {(0,): 0.0, (1,): 0.0, (0, 1): 1.0}
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
        (0, 1, 2): 1.0
    }
)

# The same complex with a visible H1 interval from 1 to 2.
LOOP_KILLED_BY_TRIANGLE_COMPLEX = FILLED_TRIANGLE_COMPLEX
LOOP_KILLED_BY_TRIANGLE_FILTRATION = Filtration(
    LOOP_KILLED_BY_TRIANGLE_COMPLEX,
    {
        (0,): 0.0, (1,): 0.0, (2,): 0.0,
        (0, 1): 1.0, (0, 2): 1.0, (1, 2): 1.0,
        (0, 1, 2): 2.0
    }
)


# The triangle boundary has an essential H1 class (paired with star).
ESSENTIAL_LOOP_COMPLEX = SimplicialComplex(
    ((0, 1), (0, 2), (1, 2))
)
ESSENTIAL_LOOP_FILTRATION = Filtration(
    ESSENTIAL_LOOP_COMPLEX,
    {
        (0,): 0.0, (1,): 0.0, (2,): 0.0,
        (0, 1): 1.0, (0, 2): 1.0, (1, 2): 1.0
    }
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
    C_SIMPLEX: "c", D_SIMPLEX: "d"
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
        C_SIMPLEX: 2.0, D_SIMPLEX: 2.0
    }
)
L_G_FILTRATION = Filtration(
    L_COMPLEX,
    {
        **{_simplex: 0.0 for _simplex in L_EARLY_SIMPLICES},
        A_SIMPLEX: 1.0, B_SIMPLEX: 2.0,
        C_SIMPLEX: 3.0, D_SIMPLEX: 3.0
    }
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

# Example 3.16: two unrelated H0 pairs with identical (birth, death) values.
TWO_DISJOINT_EDGES_COMPLEX = SimplicialComplex(((0, 1), (2, 3)))
TWO_DISJOINT_EDGES_FILTRATION = Filtration(
    TWO_DISJOINT_EDGES_COMPLEX,
    {
        (0,): 0.0, (1,): 1.0, (0, 1): 2.0,
        (2,): 0.0, (3,): 1.0, (2, 3): 2.0
    }
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
    "two_disjoint_edges": TWO_DISJOINT_EDGES_COMPLEX
}
PAPER_FILTRATIONS = {
    "single_edge": SINGLE_EDGE_FILTRATION,
    "filled_triangle": FILLED_TRIANGLE_FILTRATION,
    "loop_killed_by_triangle": LOOP_KILLED_BY_TRIANGLE_FILTRATION,
    "essential_class_paired_with_star": ESSENTIAL_LOOP_FILTRATION,
    "l_f": L_F_FILTRATION,
    "l_g": L_G_FILTRATION,
    "two_disjoint_edges": TWO_DISJOINT_EDGES_FILTRATION
}
ADDITIONAL_COMPLEXES = {
    "disconnected_vertices": DISCONNECTED_VERTICES_COMPLEX,
    "square_loop": SQUARE_LOOP_COMPLEX,
    "tetrahedron_boundary": TETRAHEDRON_BOUNDARY_COMPLEX,
    "filled_tetrahedron": FILLED_TETRAHEDRON_COMPLEX
}
ADDITIONAL_FILTRATIONS = {
    "disconnected_vertices": DISCONNECTED_VERTICES_FILTRATION,
    "square_loop": SQUARE_LOOP_FILTRATION,
    "tetrahedron_boundary": TETRAHEDRON_BOUNDARY_FILTRATION,
    "filled_tetrahedron": FILLED_TETRAHEDRON_FILTRATION
}
