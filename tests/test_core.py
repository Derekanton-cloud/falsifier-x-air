import numpy as np
from falsifier_x_air.schema import Flight
from falsifier_x_air.twin import AviationDigitalTwin
from falsifier_x_air.predictor import BaselinePredictor
from falsifier_x_air.adequacy import StructuralAdequacyDetector

def test_hidden_rotation_changes_delay():
    flights=[Flight("F1","A","B","AC1",0),Flight("F2","B","C","AC1",1)]
    twin=AviationDigitalTwin(flights,{"AIRCRAFT_ROTATION"},seed=1)
    a=twin.run(weather=1,capacity=.7)
    b=twin.run(weather=1,capacity=.7,
               interventions={"disable_aircraft_rotation":True})
    assert sum(a.delays.values()) > sum(b.delays.values())

def test_interval():
    rng=np.random.default_rng(1)
    X=rng.random((50,2))
    y=8*X[:,0]+12*(1-X[:,1])
    m=BaselinePredictor().fit(X,y)
    p=m.predict(X[:3])
    assert len(p.mean)==3 and np.all(p.upper>=p.lower)

def test_ood_gate():
    d=StructuralAdequacyDetector()
    class P:
        mean=np.array([20.])
        lower=np.array([10.])
        upper=np.array([30.])
    r=d.evaluate(np.array([100.]),P(),5,10,1,1)
    assert r.state=="ADEQUATE"
