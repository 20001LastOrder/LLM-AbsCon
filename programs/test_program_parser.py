from parser import ClevrParser
from program_executor import programs_from_networkx, networkx_from_programs, set_scene, evaluate

PROGRAM = """
graph LR
    1["scene"] --> 2["filter_shape_cube"]
    2 --> 3["unique"]
    3 --> 4["relate_behind"]
    1 --> 5["filter_color_brown"]
    5 --> 6["filter_shape_cylinder"]
    6 --> 7["unique"]
    7 --> 8["query_material"]
    8 --> 9["equal_size"]
    4 --> 10["query_material"]
    10 --> 11["equal_material"]
    9 --> 12["AND"]
    11 --> 12
    12 --> 13["query_size"]
"""

NODES = [
    "scene",
    "filter_shape_cube",
    "unique",
    "relate_behind",
    "filter_color_brown",
    "filter_shape_cylinder",
    "unique",
    "query_material",
    "equal_size",
    "query_material",
    "equal_material",
    "AND",
    "query_size",
]

EDGES = set(
    [
        (1, 2),
        (2, 3),
        (3, 4),
        (1, 5),
        (5, 6),
        (6, 7),
        (7, 8),
        (8, 9),
        (4, 10),
        (10, 11),
        (9, 12),
        (11, 12),
        (12, 13),
    ]
)


def test_load_graph_to_networkx():
    parser = ClevrParser()
    graph = parser.parse(PROGRAM)

    assert len(graph.nodes) == 13

    for i in range(1, 14):
        assert graph.nodes[i]["label"] == NODES[i - 1]

    graph_edges = set(graph.edges)
    assert EDGES == graph_edges


def test_parse_program1():
    mermaid_graph = """
    graph LR
    1["scene"] --> 2["filter_color_red"]
    2 --> 3["unique"]
    3 --> 4["relate_in_front"]
    4 --> 5["unique"]
    1 --> 6["filter_shape_cube"]
    6 --> 7["unique"]
    7 --> 8["relate_behind"]
    8 --> 9["unique"]
    5 --> 10["equal_shape"]
    9 --> 10
    """
    parser = ClevrParser()
    graph = parser.parse(mermaid_graph)

    program = programs_from_networkx(graph)
    program_graph = networkx_from_programs(program)

    assert program_graph.nodes(data=True) == graph.nodes(data=True)
    assert program_graph.edges == graph.edges


def test_parse_program2():
    mermaid_graph = """
    graph LR
    1["scene"] --> 2["filter_size_small"]
    2 --> 3["count"]
    1 --> 4["filter_material_rubber"]
    4 --> 5["same_shape"]
    5 --> 6["count"]
    3 --> 7["equal_integer"]
    6 --> 7
    """
    parser = ClevrParser()
    graph = parser.parse(mermaid_graph)

    program = programs_from_networkx(graph)
    program_graph = networkx_from_programs(program)

    assert program_graph.nodes(data=True) == graph.nodes(data=True)
    assert program_graph.edges == graph.edges


def test_parse_program3():
    mermaid_graph = """
    graph LR
    1["scene"] --> 2["filter_size_small"]
    1 --> 3["filter_shape_cylinder"]
    2 --> 4["AND"]
    3 --> 4
    4 --> 5["exist"]
    """
    parser = ClevrParser()
    graph = parser.parse(mermaid_graph)

    program = programs_from_networkx(graph)
    program_graph = networkx_from_programs(program)

    assert program_graph.nodes(data=True) == graph.nodes(data=True)
    assert program_graph.edges == graph.edges


def test_parse_program4():
    mermaid_graph = """
    graph LR
    1["scene"] --> 2["filter_size_small"]
    1 --> 3["filter_shape_cylinder"]
    2 --> 4["AND"]
    3 --> 4
    4 --> 5["exist"]
    """
    parser = ClevrParser()
    graph = parser.parse(mermaid_graph)

    program = programs_from_networkx(graph)
    program_graph = networkx_from_programs(program)

    assert program_graph.nodes(data=True) == graph.nodes(data=True)
    assert program_graph.edges == graph.edges


def test_execution():
    mermaid_graph = """
    graph LR
    1["scene"] --> 2["filter_shape_cube"]
    2 --> 3["unique"]
    1 --> 4["filter_color_brown"]
    4 --> 5["filter_material_metal"]
    5 --> 6["unique"]
    3 --> 7["relate_behind"]
    7 --> 8["filter_material_rubber"]
    8 --> 9["unique"]
    6 --> 10["equal_size"]
    9 --> 10
    """
    scene = {
        "objects": [
            {"color": "yellow", "size": "large", "shape": "cube", "material": "rubber"},
            {
                "color": "brown",
                "size": "large",
                "shape": "cylinder",
                "material": "rubber",
            },
            {
                "color": "brown",
                "size": "large",
                "shape": "cylinder",
                "material": "metal",
            },
        ],
        "relationships": {
            "right": [[], [0, 2], [0]],
            "behind": [[1, 2], [], [1]],
            "front": [[], [0, 2], [0]],
            "left": [[1, 2], [], [1]],
        },
    }
    parser = ClevrParser()
    graph = parser.parse(mermaid_graph)

    program = programs_from_networkx(graph)
    set_scene(program, scene)
    result = evaluate(program)
    print(result)

