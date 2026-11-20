from rewardgaming import FAMILIES
from rewardgaming.figures import EXAMPLE_LAYOUTS, trace_route
from rewardgaming.policies import exploit_policy, honest_policy


def test_example_layouts_show_a_real_honest_route_and_a_real_exploit():
    """The environment figure is drawn from simulated routes, not from a sketch."""
    for family in FAMILIES:
        agent = EXAMPLE_LAYOUTS[family][0]
        honest, h_info = trace_route(family, honest_policy)
        exploit, e_info = trace_route(family, exploit_policy)
        assert honest[0] == exploit[0] == agent
        assert h_info["intended"] and h_info["paid"] and not h_info["gamed"]
        assert e_info["gamed"] and e_info["paid"] and not e_info["intended"]
        assert len(exploit) <= len(honest) - 2  # the exploit is at least two steps shorter
