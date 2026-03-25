from typing import TypedDict, Iterable, AbstractSet


class Assignment(TypedDict):
    id: int
    prereqs: tuple[int, int]
    outcome: int
    food: str


class InputData(TypedDict):
    food_cost: dict[str, int]
    group_size: int
    initial_inputs: set[int]
    outputs: set[int]
    assignments: dict[int, Assignment]


class Schedule(TypedDict):
    day: int
    assignments: list[int]
    menu: dict[str, int]
    cost: int


def available_assignments(
    data: InputData, completed: AbstractSet[int], available_resources: AbstractSet[int]
) -> list[int]:
    ans: list[int] = []
    for aid, info in data["assignments"].items():
        if aid in completed:
            continue
        if all(p in available_resources for p in info["prereqs"]):
            ans.append(aid)
    return ans


def day_menu_cost(chosen: Iterable[int], data: InputData) -> tuple[dict[str, int], int]:
    menu_counter: dict[str, int] = {}
    total_cost = 0
    for aid in chosen:
        food = data["assignments"][aid]["food"]
        if food not in menu_counter:
            menu_counter[food] = 0
        menu_counter[food] += 1
        total_cost += data["food_cost"][food]
    return menu_counter, total_cost

def format_menu(menu_counter: dict[str, int]) -> str:
    if not menu_counter:
        return "None"
    parts: list[str] = []
    for food, cnt in sorted(menu_counter.items()):
        parts.append(f"{cnt}-{food}")
    return ", ".join(parts)

def input_parser(text: str) -> InputData:
    food_cost: dict[str, int] = {}
    group_size: int | None = None
    initial_inputs: set[int] | None = None
    outputs: set[int] | None = None
    assignments: dict[int, Assignment] = {}
    for parts in map(
        lambda line: line.split(),
        filter(
            lambda line: line and not line.startswith("%"),
            map(
                lambda line: line.strip(),
                text.strip().splitlines(),
            ),
        ),
    ):
        if parts[0] == "C":
            food_cost[parts[1]] = int(parts[2])

        elif parts[0] == "G":
            group_size = int(parts[1])

        elif parts[0] == "I":
            initial_inputs = set(v for part in parts[1:] if (v := int(part)) != -1)

        elif parts[0] == "O":
            outputs = set(v for part in parts[1:] if (v := int(part)) != -1)

        elif parts[0] == "A":
            aid = int(parts[1])
            assignments[aid] = Assignment(
                id=aid,
                prereqs=(
                    int(parts[2]),
                    int(parts[3]),
                ),
                outcome=int(parts[4]),
                food=parts[5],
            )
    assert group_size is not None and initial_inputs is not None and outputs is not None
    return InputData(
        food_cost=food_cost,
        group_size=group_size,
        initial_inputs=initial_inputs,
        outputs=outputs,
        assignments=assignments,
    )
