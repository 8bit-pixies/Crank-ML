# 🔧 Crank ML

A selection of single-file machine learning recipes for `pytorch`. The provided implementations are intended to be self-contained and reusable in different contexts and problems. The goals of this project is:

* Single-file implementation. Each component and single script only depends on `pytorch` and is placed in a single file. As such there may be duplicated code in places, but this is _intentional_. 
* Provide recipes via examples for machine learning use cases in a manner that is comparable to `scikit-learn`. 

## Design Principles

- the only dependencies should be pytorch - no other preprocessing library is required when performing _inference_ workflows. All items can be exported to `onnx` by default
- API naming should follow the broad patterns established in `scikit-learn`, however the actual interfaces will remain as `pytorch`-style inputs/outputs
- To support best practises, we will encourage using lightning
- Parameters which are not differentiable are updated stochastically via polyak averaging (e.g. `KBinsDiscretizer`)

# Implementations

| Implementation | Description |
| ----------- | ----------- |
| SGD | `linear_model/sgd_classifier.py`, `linear_model/sgd_regressor.py` |
| KBinsDiscretizer | `preprocessing/kbins_discretizer.py` |
| StandardScaler | `preprocessing/standard_scaler.py` |
| FactorizationMachine | `factorization_machine/factorization_machine_classifier.py` |
| FieldawareFactorizationMachine | `factorization_machine/fieldaware_factorization_machine_classifier.py` this variation uses random n latent variables |
| NeuralDecisionForest | `tree/neural_decision_forest_classifier.py` this variation uses smoothstep instead of logistic function for the soft routing. See: https://arxiv.org/abs/2002.07772 |
| NeuralDecisionBoosting | `tree/neural_decision_boosting_classifier.py` this neural decision forest with gentleboost for the boosting variation |
| KMeans | `cluster/kmeans.py` this is not a differentiable variation |
| PCA | `decomposition/pca.py` |
| TabNet | `tabnet/tabnet_classifier.py` tabnet implementation without the pre-training step, based on the dreamquark-ai implementation but now ONNX exportable |
| TabNet | `tabnet/tabnet_regressor.py` tabnet implementation without the pre-training step, based on the dreamquark-ai implementation but now ONNX exportable|
| TabNetPretraining | `impute/tabnet_pretraining` tabnet pretraining for imputation using encoder/decoder architecture |
