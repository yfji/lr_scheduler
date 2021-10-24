# lr_scheduler
A numpy implementation of several learning rate schedulers

This code implements several learning rate schedulers

## Learning rate decaying strategies
* WarmupLR </br>
This scheduler increases lr during warming up after each iteration. So you should use * iterations instead of epochs in warmup.

* StepLR </br>
This scheduler manages lr during training after each epoch. In fact, lr may not decay as the current epoch is not in the milestones.

* LinearLR
This scheduler decays lr during training after each epoch in a linear manner. Once the milestone encountered, lr decays by gamma.

* ExponentialLR
This scheduler decays lr during training after each epoch in a exponential manner. Once the milestone encountered, lr decays by gamma.

* CosineLR
This scheduler decays lr during training after each epoch in a cosine manner. Once the milestone encountered, lr decays by gamma. Notice that the curve in which lr decays by gamma is shorter than a period.

* CosineLRv2
This scheduler decays lr during training after each epoch in a cosine manner. Once the milestone encountered, lr decays by gamma. Notice that the curve in which lr decays by gamma is half a period.

* DampeningLR
This scheduler decays lr during training after each epoch in a dampening manner. Once the milestone encountered, lr decays by gamma. Now it decays using a linear manner and dampens after each epoch, while you can extend this method to decay by other curves

## Learning rate resume
All the classes except WarmupLR provide resume function to recover the learning rate from a given epoch.

## Sample
![LR schedulers](https://github.com/yfji/lr_scheduler/blob/main/LR%20schedulers.png)
