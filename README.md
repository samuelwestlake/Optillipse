# Optillipse

The pramatisation of an ellipse to encapsulate a given shape.

Example usage:
```python
from optillipse.model import Model

model = Model()
model.load_image("examples/01.png", pad=100)
model.fit()
```

![Alt Text](examples/example-03.gif)

## Model

```python
Model(
    image=None, 
    steps=100,
    fn_max=5, 
    threshold_1=1e-2, 
    threshold_2=1e-4, 
    period_1=10, 
    period_2=100,   
    lr_steps=25,
    verbose_steps=10, 
    vis_steps=1, 
    video_filename=None
)
```

### Methods

#### fit

```python
model.fit(steps=None)
```
TODO

#### load_image

```python
model.load_image(
    filename, 
    pad=0, 
    invert=False, 
    threshold=None, 
    read_flag=cv2.IMREAD_UNCHANGED
)
 ```
TODO

#### get_lr()

```python
model.get_lr()
```
TODO

#### print_parameters

```python
model.print_parameters()
```
TODO

#### print_status

```python
model.print_status()
```
TODO

#### reset

```python
model.reset()
```
TODO

#### reset_lr_scheduler

```python
model.reset_lr_scheduler()
```
TODO

#### reset_optimizer

```python
model.reset_optimizer()
```
TODO

#### set_lr_scheduler

```python
model.set_lr_scheduler(scheduler, **kwargs)
```
TODO

#### set_optimizer

```python
model.set_optimizer(optimizer, **kwargs)
```
TODO

#### set_weights

```python
model.set_weights(weights)
```
TODO

#### show_image

```python
model.show_image()
```
TODO

### Properties

#### device
TODO

#### fn_weight
TODO

#### image
TODO

#### metrics
TODO

#### loss
TODO

#### loss_as_float
TODO

#### lr_scheduler
TODO

#### lr_scheduler_args
TODO

#### optimizer
TODO

#### optimizer_args
TODO

#### period_1
TODO

#### period_2
TODO

#### threshold_1
TODO

#### threshold_2
TODO

#### weights
TODO

## How does it work?
TODO

### Phase 1

### Phase 2

### Phase 3

### Phase 4
