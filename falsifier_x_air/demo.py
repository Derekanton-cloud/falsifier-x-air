import numpy as np
from .schema import Flight
from .twin import AviationDigitalTwin
from .predictor import BaselinePredictor, CrossModelAgreement
from .falsifier import FalsifierXAir
from .recovery import compare_recovery

def make_flights():
    return [
        Flight("F1","A","B","AC1",0), Flight("F2","B","C","AC1",1),
        Flight("F3","C","D","AC1",2), Flight("F4","A","C","AC2",0),
        Flight("F5","C","D","AC2",1), Flight("F6","D","A","AC2",2),
    ]

def main():
    rng = np.random.default_rng(7)
    n = 400
    weather = rng.uniform(0,1,n)
    capacity = rng.uniform(.6,1,n)
    X = np.column_stack([weather, capacity])
    y = 8*weather + 12*(1-capacity) + rng.normal(0,1.5,n)

    predictor = BaselinePredictor().fit(X,y)
    cross = CrossModelAgreement().fit(X,y)

    flights = make_flights()
    twin = AviationDigitalTwin(
        flights, hidden_mechanisms={"AIRCRAFT_ROTATION"}, seed=21
    )
    state = twin.run(weather=1.0, capacity=.7)
    obs = np.array(list(state.delays.values()))
    Xobs = np.tile([1.0,.7], (len(obs),1))
    pred = predictor.predict(Xobs)
    cross_pred = cross.predict(Xobs)

    system = FalsifierXAir()
    result = system.investigate(
        twin=twin, y_true=obs, prediction=pred,
        ood_score=float(np.mean(predictor.feature_distance(Xobs))),
        persistence_count=3, graph_concentration=.95,
        cross_model_gap=float(np.mean(np.abs(pred.mean-cross_pred))/20),
        local_context={"has_aircraft_rotation": True}
    )

    print("\n=== FALSIFIER-X AIR v0.1 ===")
    print("State:", result.adequacy.state)
    print("MIS:", round(result.adequacy.mis_score,3))
    print("Reason:", result.adequacy.reason)

    for name, r in result.experiment_results.items():
        print(f"{name:24s} | {r['intervention']:30s} | effect={r['effect']:.1f}")

    print("Selected mechanism:", result.selected_mechanism)
    print("Selected experiment:", result.selected_experiment)

    recovery = compare_recovery(twin)
    print("\nRecovery:")
    for name, r in recovery["actions"].items():
        print(f"{name:30s} delay={r['total_delay']:.1f} improvement={r['improvement']:.1f}")
    print("Best action:", recovery["best_action"])

if __name__ == "__main__":
    main()
