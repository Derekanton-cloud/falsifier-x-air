import numpy as np

class ActiveExperimentSelector:
    def __init__(self):
        self.costs = {
            "disable_aircraft_rotation": 1.0,
            "increase_capacity": 1.5,
            "disable_resource_dependency": 2.0,
        }

    def experiment_for(self, mechanism):
        return {
            "AIRCRAFT_ROTATION": "disable_aircraft_rotation",
            "AIRPORT_CAPACITY": "increase_capacity",
            "RESOURCE_DEPENDENCY": "disable_resource_dependency",
        }[mechanism.name]

    def select(self, candidates):
        signatures = {
            "AIRCRAFT_ROTATION": {
                "disable_aircraft_rotation": .60,
                "increase_capacity": .15,
                "disable_resource_dependency": .03},
            "AIRPORT_CAPACITY": {
                "disable_aircraft_rotation": .05,
                "increase_capacity": .45,
                "disable_resource_dependency": .03},
            "RESOURCE_DEPENDENCY": {
                "disable_aircraft_rotation": .02,
                "increase_capacity": .05,
                "disable_resource_dependency": .30},
        }
        best, best_score = None, -np.inf
        for exp, cost in self.costs.items():
            vals = [signatures[c.name][exp] for c in candidates]
            score = float(np.std(vals))/cost
            if score > best_score:
                best, best_score = exp, score
        return best

def run_experiment(twin, intervention, weather=1.0, capacity=.7):
    before = twin.total_delay(weather=weather, capacity=capacity)
    after = twin.total_delay(
        weather=weather, capacity=capacity,
        interventions={intervention: True}
    )
    return before, after
