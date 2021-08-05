# Mask R-CNN for Object Detection and Segmentation

Dies ist eine Implementierung von [Mask R-CNN](https://github.com/leekunhee/Mask_RCNN), speziell für Tensorflow >= 2.0.

## Änderungen

`- mrcnn/model.py:`

```
    def eval(self, val_dataset):
        # Data generators
        val_generator = DataGenerator(val_dataset, self.config, shuffle=True)
        
        # Compile, needed for custom callbacks
        #self.compile(learning_rate, self.config.LEARNING_MOMENTUM)
        
        return self.keras_model.evaluate(val_generator)


    def m_get_weights(self):
        return self.keras_model.get_weights()

        
    def m_set_weights(self, weights):
        self.keras_model.set_weights(weights)
```

Damit es auch unter Linux laeuft, in der `self.keras_model.fit`:

- `workers=workers,` -> `workers=0,`
- `use_multiprocessing=workers > 1,` `use_multiprocessing=False,` (im Prinzip das gleiche)

