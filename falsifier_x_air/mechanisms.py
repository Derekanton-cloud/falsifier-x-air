from .schema import Mechanism

MECHANISM_ONTOLOGY = [
    Mechanism("AIRCRAFT_ROTATION","flight_delay","AIRCRAFT_ROTATION",
              "next_flight_delay","positive",
              "Delay propagates through an aircraft's next rotation."),
    Mechanism("AIRPORT_CAPACITY","airport_capacity","CAPACITY_CONSTRAINT",
              "flight_delay","positive",
              "Reduced airport capacity increases delay."),
    Mechanism("RESOURCE_DEPENDENCY","resource_state","RESOURCE_DEPENDENCY",
              "flight_delay","positive",
              "Shared operational resource constraints increase delay."),
]

def generate_candidates(local_context, top_k=3):
    candidates = []
    if local_context.get("has_aircraft_rotation"):
        candidates.append(MECHANISM_ONTOLOGY[0])
    candidates.extend([MECHANISM_ONTOLOGY[1], MECHANISM_ONTOLOGY[2]])
    return candidates[:top_k]
