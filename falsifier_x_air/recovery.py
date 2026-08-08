def compare_recovery(twin, weather=1.0, capacity=.7):
    base = twin.total_delay(weather=weather, capacity=capacity)
    actions = {
        "NO_ACTION": {},
        "INCREASE_CAPACITY": {"increase_capacity": True},
        "DISABLE_AIRCRAFT_ROTATION": {"disable_aircraft_rotation": True},
    }
    results = {}
    for name, intervention in actions.items():
        delay = twin.total_delay(
            weather=weather, capacity=capacity,
            interventions=intervention
        )
        results[name] = {
            "total_delay": delay,
            "improvement": base-delay
        }
    best = min(results, key=lambda x: results[x]["total_delay"])
    return {"baseline": base, "actions": results, "best_action": best}
