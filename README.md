Author: Kenan Suljić


# pydmcSR

Adaptded Version of pydmc: Python implementation of the diffusion process model (Diffusion Model for Conflict Tasks, DMC) presented in Automatic and controlled stimulus processing in conflict tasks: Superimposed diffusion processes and delta functions (<https://www.sciencedirect.com/science/article/pii/S0010028515000195>).

Instead of 1 Diffusion Process, there are 2: One sensory diffusion process & 1 response diffusion process. The reason is to enable that both can be either congruent or incongurent sepereately.

## Installation


git clone [<https://github.com/KSuljic/pydmcSR.git>](https://github.com/KSuljic/pydmcSR.git)

pip install -e pydmc

## Basic Examples

```python
import pydmc

dat = pydmc.Sim(full_data=True)
dat.plot.summary()
```

![alt text](/figures/figure1.png)

```python
import pydmc

dat = pydmc.Sim(pydmc.Prms(tau=150), full_data=True)
dat.plot.summary()
```

![alt text](/figures/figure2.png)

```python
from pydmc import Sim, Prms

dat = Sim(Prms(tau=150))
dat.plot.summary()

```
