# Flower Client

Flower ist ein Federated Learning Framework (siehe: [Flower](https://github.com/adap/flower)). <br>
Flower kann bei Tensorflow-, PyTorch- und Raw-Numpy Modellen genutzt werden.<br>
Hier ist das Release 0.13.0 umgesetzt.

Flower setzt bei der Kommunikation zwischen Server und Clients auf gRPC (Google Remote Procedure Call) welches wiederum auf Protobuf (auch von Google, Binäre Serialisierung) setzt. Da das Format zur Serialisierung vorgegebenen werden muss, sind Aenderungen im Code nur bedingt vornehmbar.

## Installation und Test

- Conda Environment mit Python>=3.7
- Clonen der Repo: git clone --recursive <repository> (recursive da ein Submodul (flower-common) benoetigt wird)
- Installieren der requirements: `pip install -r requirements.txt`

## Branches
Abhaengig ob ein Tensorflow/Keras oder ein Pytorch Modell trainiert wird, bitte den entsprechenden Branch auswaehlen. Am Server Code ist im Prinzip nichts anders außer der Gewichte im Beispiel, welches der Server fuer Federated LEarning zu Beginn laedt. Wird noch gaendert!

## Beispiel

Die `examples/client.py` zeigt ein Beispiel.<br>
- Modell ist hier festgelegt (Mask R-CNN)
- Trainings- und Testdaten werden hier festgelegt (Cifar10)
- **--ip** Server IP Adresse
- **--port** des Servers, Default: 8080
- **--init** lädt die vortrainierten Coco Modelparameter und speichert diese in `weights/0000.weights` ab. Diese müssen dann zum Server in `federated/history/data/0000.weights` kopiert werden, damit diese dann Federated Learning durchgeführt werden kann,

Gewichte für den Server initialsieren:
```
python client.py --ip 10.122.198.57 --port 8080 --init
```
Starten des Clients:
```
python client.py --ip 10.122.198.57 --port 8080 --init
```

Die durch Federated Learning gemittelten Modellparameter werden binaer in `weights/0000.weights` gespeichtert und koennen fuer Inference wie folgt geladen werden:
```
weights = pickle.load(open(weights/0000.weights, 'rb'))
model.m_set_weights(weights) bzw. model.keras_model.set.weights(weights)
```

Jeder Client legt eine logfile in `federated/logfiles/<datetime>.log` an, in der alle Federated Learning Events des Clients festgehalten werden.

## Note

- Die maximale Laenge der gRPC Messages ist auf 512MB festgelegt (definiert in `flwr/common/__init__.py: GRPC_MAX_MESSAGE_LENGTH, default: 536_870_912, this equals 512MB`).<br> Fuer das Trainieren groeßere Modelle muss diese Groeße eventuell angepasst werden. <br> The Flower server needs to be started with the same value (see `flwr.server.start_server`).

## ToDo


