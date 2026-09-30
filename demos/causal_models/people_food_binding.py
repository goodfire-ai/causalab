"""A compact entity-binding model: find the queried person, return their food.

An additional hypothesis illustrating None and reusable namespaced submodels.
The shipped entity_binding task retains its original positional retrieval graph.
"""

from causalab.causal import Dom, FamilyDom, V, family, mechanism, submodel
from causalab.causal.model import CausalModel


def unique_true_index(flags):
    positions = tuple(i for i, flag in enumerate(flags) if flag is True)
    return positions[0] if len(positions) == 1 else None


def make_people_food_binding(names, foods, groups=2):
    Name = Dom(tuple(names))
    Food = Dom(tuple(foods))

    @submodel
    def locate(people, query, active_groups):
        matches = family(size=groups)
        for g in range(groups):
            if g < active_groups:
                matches[g] = people[g] == query
            else:
                matches[g] = None
        position = V(unique_true_index(matches), domain=Dom([None, *range(groups)]))
        return position

    @mechanism
    def people_food_binding(
        people: FamilyDom(Name, size=groups),
        foods: FamilyDom(Food, size=groups),
        query: Name,
        active_groups: Dom(range(groups + 1)),
    ):
        location = locate(people, query, active_groups)
        answer = V(None if location is None else foods[location])
        raw_input = V(f"{people}:{foods}:{query}:{active_groups}", domain=Dom(str))  # noqa: F841
        raw_output = V("UNKNOWN" if answer is None else answer, domain=Dom(str))  # noqa: F841
        return answer

    return CausalModel(people_food_binding, id="people_food_binding")
