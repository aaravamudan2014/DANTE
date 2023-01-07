#!/usr/bin/python
'''Memory Kernel base and sub-classes'''
from abc import ABC, abstractmethod
import numpy as np
from scipy.interpolate import UnivariateSpline, CubicSpline
from scipy import optimize

################################################################################
#
# MemoryKernel abstract class definition & implementation
#
# An abstract class designed to encapsulate memory kernel information and functionality.
# Such memory kernels are used to define point processes.
# A memory kernel of a random variable $T > 0$ w.p. 1 representing time is a
# left-continuous, non-negative function $\phi(t)$, which is zero for $t <= 0$.
# Its integrated kernel is a continuous, non-negative function defined as
# $\psi(t) \triangleq [t >= 0] \int_{0}^t \phi(\tau) d\tau$.
# The random variable's PDF, Survival Function and CDF are given as
# $f(t) = \phi(t) e^{-\psi(t)}$, $S(t) = e^{-\psi(t)}$ and
# $F(t) = 1 - S(t)$ respectively.
# This class allows for memory kernels of random variables of the form $T' \trianglew T + t_o$,
# so that $T > t_o$ w.p. 1. In that case the previous quantities (memory kernel, PDF, etc.)
# for $T'$ are obtained by offsetting $t \to t - t_o$.
#
#
# PUBLIC ATTRIBUTES:
#
#   desc:       string; description of memory kernel.
#   timeOffset: real scalar; offset $t_o$ of the (integrated) kernel on the time axis.
#
# PUBLIC INTERFACE:
#
#   MemoryKernel(): object; constructor of abstract class; not callable.
#   phi():          numpy.ndarray; values  (or right limits) of the memory kernel $\phi(\cdot)$ at specified time instances; abstract method.
#   phiUB():        real scalar; upper bound for $\phi(\cdot)$ over a given time interval; abstract method.
#   psi():          numpy.ndarray; values of the integrated memory kernel $\psi(\cdot)$ at specified time instances; abstract method.
#   psiInv():       numpy.ndatrrayl values of the inverse integrated memory kernel $\psi(\cdot)$ at specified values $z$; abstract method.
#   pdf():          numpy.ndarray; values of the implied PDF at specified time instances.
#   sf():           numpy.ndarray; values of the implied survival function (1-CDF) at specified time instances.
#   cdf():          numpy.ndarray; values of the implied PDF (1-SF) at specified time instances.
#   __str__():      string; string representation of object; defaults to the description of memory kernel.
#   print():        None; print string representation of object
#
# USAGE EXAMPLES:
#
#   This abstract class is only intended to be sub-classed.
#
# DEPENDENCIES:
#   imports: abc.ABC, abc.abstractmethod, numpy as np
#
# NOTES: Objects of this class cannot be instantiated.
##
################################################################################


class MemoryKernel(ABC):
    '''Abstract base class for memory kernels'''

    # Instance variables
    #
    # desc:       string; name of the memory kernel
    # timeOffset: real scalar; memory kernel's offset in time

    # Construct a base class object (not callable)
    def __init__(self, desc='base memory kernel', timeOffset=0.0):
        self.desc = desc
        self.timeOffset = timeOffset
        super().__init__()

    # Create a string representing the event
    # Usage example: print(mk)
    def __str__(self):
        return self.desc

    # Print out the string supplied by self.__str()__
    # Usage example: print(mk)
    def print(self):
        print(self)

    # Value of memory kernel at time t (not callable)
    @abstractmethod
    def phi(self, t, rightLimit=False):
        pass

    # Upper-bound of the memory kernel's value in the interval (t1,t2) (not callable)
    @abstractmethod
    def phiUB(self, t1, t2):
        pass

    # Value of integrated memory kernel at time t (not callable)
    @abstractmethod
    def psi(self, t):
        pass

    # Value of inverse integrated memory kernel at value z (not calable)
    @abstractmethod
    def psiInv(self, z):
        pass

    # PDF value from memory kernel & integrated memory kernel at time t
    def pdf(self, t):
        return self.phi(t) * np.exp(- self.psi(t))

    # Survival function value from integrated memory kernel at time t
    def sf(self, t):
        return np.exp(- self.psi(t))

    # CDF value from memory kernel & integrated memory kernel at time t
    def cdf(self, t):
        return 1.0 - np.exp(- self.psi(t))


################################################################################
#
# Definitions & implementations of the following MemoryKernel sub-classes:
#
#   ConstantMemoryKernel: memory kernel for a unit-scale exponential distirbution.
#   RayleighMemoryKernel: memory kernel for a unit-scale Rayleigh distribution.
#   WeibullMemoryKernel: memory kernel for a unit-scale Weibull distribution.
#   PowerLawMemoryKernel: memory kernel for a unit-shape Power Law distribution.
#   HawkesPseudoMemoryKernel: pseudo- memory kernel typically used in Hawkes processes.
#   GompertzMemoryKernel: memory kernel for a special case of the Gompertz distribution.
#   GammaGompertzMemoryKernel: memory kernel for a special case of the Gamma-Gompertz distribution.
#
# PUBLIC ATTRIBUTES:
#
#   desc:       string; description of the memory kernel; inherited from MemoryKernel base class.
#   timeOffset: real scalar; memory kernel's offset in time; inherited from MemoryKernel base class.
#   beta, gamma, etc.: real scalars; memory kernel parameters; their existence depends on the particular sub-class.
#
# PUBLIC INTERFACE:
#
#   <name>MemoryKernel(): object; constructor of the corresponding sub-class; must be implemented.
#
#   All methods listed below are inhereted from the MemoryKernel base class:
#
#   phi():          numpy.ndarray; values of the memory kernel $\phi(\cdot)$ at specified time instances; must be implemented.
#   psi():          numpy.ndarray; values of the integrated memory kernel $\psi(\cdot)$ at specified time instances; must be implemented.
#   pdf():          numpy.ndarray; values of the implied PDF at specified time instances.
#   sf():           numpy.ndarray; values of the implied survival function (1-CDF) at specified time instances.
#   cdf():          numpy.ndarray; values of the implied PDF (1-SF) at specified time instances.
#   __str__():      string; string representation of object; defaults to the description of memory kernel.
#   print():        None; print string representation of object
#
# USAGE EXAMPLES:
#
#   mk = WeibullMemoryKernel(0.5, 0.1)   # construct a Weibull kernel with gamma=0.5 time offset 0.1
#   t = np.linspace(0.0, 10.0, 100)
#   pdfValues = mk.pdf(t)   # compute the associated PDF for a given time range
#   print(mk)   # print out description of the kernel
#
# DEPENDENCIES:
#   imports: numpy as np
#   classes: MemoryKernel
#
# NOTES:
#   1) The numpy.ndarray returned by phi(), psi(), pdf(), sf() and cdf() matches the dimensions of t,
#      which is assumed to be a scalar or a numpy.ndarray.
#   2) The HawkesPseudoMemoryKernel is not a proper kernel, as its associate integrated kernel does not diverge as t
#      grows. It is provided, so that it can be combined with other memory kernels (e.g. with a constant memory kernel).
#
# AUTHOR: Georgios C. Anagnostopoulos, January 2020 (version 1.1)
#
################################################################################


################################################################################
# Memory kernel for $t_o=0$: $phi(t) = [t > 0]$.
# Integrated memory kernel for $t_o=0$: $\psi(t) = t [t \geq 0]$
#
class ConstantMemoryKernel(MemoryKernel):
    '''Constant memory kernel class'''

    # Instance variables
    #
    # desc:       string; description of the memory kernel; inherited from MemoryKernel base class.
    # timeOffset: real scalar; memory kernel's offset in time; inherited from MemoryKernel base class.

    def __init__(self, timeOffset=0.0):
        #desc = 'constant memory kernel: offset={:1}'.format(timeOffset)
        desc = 'Constant({:1})'.format(timeOffset)
        super().__init__(desc, timeOffset)

    # t: real scalar or arbitrary-dimensional numpy.ndarray of reals
    def phi(self, t, rightLimit=False):
        if type(t) is not list:
            relative_t = np.array(t - self.timeOffset)
        else:
            relative_t = np.array(t - self.timeOffset*np.ones(len(t)))
        if rightLimit is False:
            isNotZero = np.array(relative_t > 0.0).astype(int)
        else:
            isNotZero = np.array(relative_t >= 0.0).astype(int)

        return isNotZero

    # t1, t2: real scalars with t1<=t2
    def phiUB(self, t1, t2):
        if t2 < t1:  # not a necessary check; just precautionary
            raise ValueError(
                'ConstantMemoryKernel.phiUB(): 1st argument (t1) expected to be less or equal to 2nd argument (t2)!')
        ub = max(self.phi(t1, rightLimit=True), self.phi(t2, rightLimit=True))
        return ub

    # t: real scalar or arbitrary-dimensional numpy.ndarray of reals
    def psi(self, t):
        if type(t) is not list:
            relative_t = np.array(t - self.timeOffset)
        else:
            relative_t = np.array(t - self.timeOffset*np.ones(len(t)))
        isNotZero = np.array(relative_t >= 0.0).astype(int)
        value = relative_t * isNotZero
        return np.abs(value)  # to avoid -0 values

    # z: scalar or or arbitrary-dimensional numpy.ndarray of non-negative reals
    def psiInv(self, z):
        if np.any(z < 0.0):  # not a necessary check; just precautionary
            raise ValueError(
                'ConstantMemoryKernel.psiInv(): 1st argument (z) expected to be non=negative!')
        return z


################################################################################
# Memory kernel for $t_o=0$: $phi(t) = t [t > 0]$.
# Integrated memory kernel for $t_o=0$: $\psi(t) = \frac{t^2}{2} [t \geq 0]$
#
class RayleighMemoryKernel(MemoryKernel):
    '''Rayleigh memory kernel class'''

    # Instance variables
    #
    # desc:       string; description of the memory kernel; inherited from MemoryKernel base class.
    # timeOffset: real scalar; memory kernel's offset in time; inherited from MemoryKernel base class.

    def __init__(self, timeOffset=0.0):
        #desc = 'Rayleigh memory kernel: offset={:1}'.format(timeOffset)
        desc = 'Rayleigh({:1})'.format(timeOffset)
        super().__init__(desc, timeOffset)

    # t: real scalar or arbitrary-dimensional numpy.ndarray of reals
    def phi(self, t, rightLimit=False):
        if type(t) is not list:
            relative_t = np.array(t - self.timeOffset)
        else:
            relative_t = np.array(t - self.timeOffset*np.ones(len(t)))
        if rightLimit is False:
            isNotZero = np.array(relative_t > 0.0).astype(int)
        else:
            isNotZero = np.array(relative_t >= 0.0).astype(int)
        value = relative_t * isNotZero
        return np.abs(value)  # to avoid -0 values

    # t1, t2: real scalars with t1<=t2
    def phiUB(self, t1, t2):
        if t2 < t1:  # not a necessary check; just precautionary
            raise ValueError(
                'RayleighMemoryKernel.phiUB(): 1st argument (t1) expected to be less or equal to 2nd argument (t2)!')
        # This kernel has a continuous phi(), so no need for right limits.
        ub = max(self.phi(t1), self.phi(t2))
        return ub

    # t: real scalar or arbitrary-dimensional numpy.ndarray of reals
    def psi(self, t):
        if type(t) is not list:
            relative_t = np.array(t - self.timeOffset)
        else:
            relative_t = np.array(t - self.timeOffset*np.ones(len(t)))
        isNotZero = np.array(relative_t >= 0.0).astype(int)
        value = 0.5 * (relative_t ** 2) * isNotZero
        return np.abs(value)  # to avoid -0 values

    # z: scalar or or arbitrary-dimensional numpy.ndarray of non-negative reals
    def psiInv(self, z):
        if np.any(z < 0.0):
            raise ValueError(
                'RayleighMemoryKernel.psiInv(): 1st argument (z) expected to be non=negative!')
        value = np.sqrt(2.0 * z)
        return value

################################################################################
# Memory kernel for $t_o=0$: $phi(t) = \gamma t^{\gamma - 1} [t > 0]$.
# Integrated memory kernel for $t_o=0$: $\psi(t) = t^{\gamma} [t \geq 0]$
# Constraints: $\gamma > 0$
#
# Notes:
#   When $\gamma=1$, it becomes the constant kernel.
#   When $\gamma=2$, it becomes a scaled Rayleigh kernel.
#


class WeibullMemoryKernel(MemoryKernel):
    '''Weibull memory kernel class'''

    # Instance variables
    #
    # desc:       string; description of the memory kernel; inherited from MemoryKernel base class.
    # timeOffset: real scalar; memory kernel's offset in time; inherited from MemoryKernel base class.
    # gamma:      positive real scalar; memory kernel parameter.

    # Must have gamma > 0
    def __init__(self, gamma, timeOffset=0.0):
        if gamma <= 0.0:
            raise ValueError(
                'WeibullMemoryKernel(): 2nd argument (gamma) must be strictly positive!')
        #desc = 'Weibull memory kernel: offset={:1}, gamma={:2}'.format(timeOffset, gamma)
        desc = 'Weibull({:1}, {:2})'.format(timeOffset, gamma)
        super().__init__(desc, timeOffset)
        self.gamma = gamma

    # t: real scalar or arbitrary-dimensional numpy.ndarray of reals
    def phi(self, t, rightLimit=False):
        if type(t) is not list:
            relative_t = np.array(t - self.timeOffset)
        else:
            relative_t = np.array(t - self.timeOffset*np.ones(len(t)))
        if rightLimit is False:
            isNotZero = np.array(relative_t > 0.0).astype(int)
        else:
            isNotZero = np.array(relative_t >= 0.0).astype(int)
        # to avoid negative numbers in the base of a power function
        abs_relative_t = np.abs(relative_t)
        value = self.gamma * (abs_relative_t ** (self.gamma - 1.0)) * isNotZero
        return value

     # t1, t2: real scalars with t1<=t2
    def phiUB(self, t1, t2):
        if t2 < t1:  # not a necessary check; just precautionary
            raise ValueError(
                'WeibullMemoryKernel.phiUB(): 1st argument (t1) expected to be less or equal to 2nd argument (t2)!')
        if self.gamma < 1.0 and t1 * t2 == 0.0:
            ub = float("inf")
        else:
            # Right limits are needed when gamma=0
            ub = max(self.phi(t1, rightLimit=True),
                     self.phi(t2, rightLimit=True))
        return ub

    # t: real scalar or arbitrary-dimensional numpy.ndarray of reals
    def psi(self, t):
        if type(t) is not list:
            relative_t = np.array(t - self.timeOffset)
        else:
            relative_t = np.array(t - self.timeOffset*np.ones(len(t)))
        isNotZero = np.array(relative_t >= 0.0).astype(int)
        # to avoid negative numbers in the base of a power function
        abs_relative_t = np.abs(relative_t)
        value = (abs_relative_t ** self.gamma) * isNotZero
        return value

    # z: scalar or or arbitrary-dimensional numpy.ndarray of non-negative reals
    def psiInv(self, z):
        if np.any(z < 0.0):
            raise ValueError(
                'WeibullMemoryKernel.psiInv(): 1st argument (z) expected to be non=negative!')
        value = np.power(z, 1.0 / self.gamma)
        return value


################################################################################
# Memory kernel for $t_o=0$: $phi(t) = \frac{1}{t} [t > \beta]$.
# Integrated memory kernel for $t_o=0$: $\psi(t) = \ln\left( \frac{t}{\beta} \right) [t \geq \beta]$
# Constraints: $\beta > 0$
#
class PowerLawMemoryKernel(MemoryKernel):
    '''Power Law memory kernel class'''

    # Instance variables
    #
    # desc:       string; description of the memory kernel; inherited from MemoryKernel base class.
    # timeOffset: real scalar; memory kernel's offset in time; inherited from MemoryKernel base class.
    # beta:       positive real scalar; memory kernel parameter.

    # beta > 0
    def __init__(self, beta, timeOffset=0.0):
        if beta <= 0.0:
            raise ValueError(
                'PowerLawMemoryKernel(): 2nd argument (beta) must be strictly positive!')
        #desc = 'Power Law memory kernel: offset={:1}, beta={:2}'.format(timeOffset, beta)
        desc = 'PowerLaw({:1}, {:2})'.format(timeOffset, beta)
        super().__init__(desc, timeOffset)
        self.beta = beta

    # beta = 0.05
    # delta_t = 10e-7
    # t: real scalar or arbitrary-dimensional numpy.ndarray of reals   ####
    # relative_t = [10e-7]
    # isGreaterThanBeta = [0]
    # relative_t = [1]
    # value = [0]
    def phi(self, t, rightLimit=False):
        if type(t) is not list:
            relative_t = np.array(t - self.timeOffset)
        else:
            relative_t = np.array(t - self.timeOffset*np.ones(len(t)))
        if rightLimit is False:
            isGreaterThanBeta = np.array(relative_t > self.beta)
        else:
            isGreaterThanBeta = np.array(relative_t >= self.beta)
        # assign an appropriate dummy value to out-of-range relative times
        relative_t[np.logical_not(isGreaterThanBeta)] = 1.0
        value = isGreaterThanBeta.astype(int) / relative_t
        return value

     # t1, t2: real scalars with t1<=t2
    def phiUB(self, t1, t2):
        if t2 < t1:  # not a necessary check; just precautionary
            raise ValueError(
                'PowerLawMemoryKernel.phiUB(): 1st argument (t1) expected to be less or equal to 2nd argument (t2)!')
        if t1 < self.beta and self.beta < t2:
            ub = 1.0 / self.beta
        else:
            ub = max(self.phi(t1, rightLimit=True),
                     self.phi(t2, rightLimit=True))
        return ub

    # t: real scalar or arbitrary-dimensional numpy.ndarray of reals
    def psi(self, t):
        if type(t) is not list:
            relative_t = np.array(t - self.timeOffset)
        else:
            relative_t = np.array(t - self.timeOffset*np.ones(len(t)))
        # .astype(int) does not work and gives wrong results!?
        isLessThanBeta = np.array(relative_t < self.beta)
        # assign an appropriate dummy value to out-of-range relative times
        relative_t[isLessThanBeta] = self.beta
        value = np.log(relative_t / self.beta)
        return value

    # z: scalar or or arbitrary-dimensional numpy.ndarray of non-negative reals
    def psiInv(self, z):
        if np.any(z < 0.0):
            raise ValueError(
                'PowerLawMemoryKernel.psiInv(): 1st argument (z) expected to be non=negative!')
        value = self.beta * np.exp(z)
        return value


################################################################################
# Pseudo- memory kernel for $t_o=0$: $phi(t) = e^{-\gamma t} [t > 0]$.
# Integrated memory kernel for $t_o=0$: $\psi(t) =  \frac{1}{\gamma}\left( 1 - e^{-\gamma t} \right) [t \geq 0]$
# Constraints: $\gamma > 0$
#
# Notes:
#   When $\gamma \to 0$, it becomes the constant kernel, but this is not implemented.
#
class ExponentialPseudoMemoryKernel(MemoryKernel):
    '''Hawkes pseudo- memory kernel class'''

    # Instance variables
    #
    # desc:       string; description of the memory kernel; inherited from MemoryKernel base class.
    # timeOffset: real scalar; memory kernel's offset in time; inherited from MemoryKernel base class.
    # beta:      positive real scalar; memory kernel parameter

    # beta > 0
    def __init__(self, beta, timeOffset=0.0):
        if beta <= 0.0:
            raise ValueError(
                'ExponentialPseudoMemoryKernel(): 2nd argument (beta) must be strictly positive!')
        #desc = 'Hawkes pseudo- memory kernel: offset={:1}, gamma={:2}'.format(timeOffset, gamma)
        desc = 'ExponentialPseudo({:1}, {:2})'.format(timeOffset, beta)
        super().__init__(desc, timeOffset)
        self.beta = beta

    # t: real scalar or arbitrary-dimensional numpy.ndarray of reals
    def phi(self, t, rightLimit=False):
        if type(t) is not list:
            relative_t = np.array(t - self.timeOffset)
        else:
            relative_t = np.array(t - self.timeOffset*np.ones(len(t)))
        if rightLimit is False:
            isNotZero = np.array(relative_t > 0.0).astype(int)
        else:
            isNotZero = np.array(relative_t >= 0.0).astype(int)
        value = np.exp(- self.beta * relative_t) * isNotZero
        return value

     # t1, t2: real scalars with t1<=t2
    def phiUB(self, t1, t2):
        if t2 < t1:  # not a necessary check; just precautionary
            raise ValueError(
                'ExponentialPseudoMemoryKernel.phiUB(): 1st argument (t1) expected to be less or equal to 2nd argument (t2)!')
        if t1 < 0.0 < t2:
            ub = 1.0
        else:
            ub = max(self.phi(t1, rightLimit=True),
                     self.phi(t2, rightLimit=True))
        return ub

    # t: real scalar or arbitrary-dimensional numpy.ndarray of reals
    def psi(self, t):
        if type(t) is not list:
            relative_t = np.array(t - self.timeOffset)
        else:
            relative_t = np.array(t - self.timeOffset*np.ones(len(t)))
        isNotZero = np.array(relative_t >= 0.0).astype(int)
        value = (1.0 - np.exp(- self.beta * relative_t)) * \
            isNotZero / self.beta
        return np.abs(value)  # to avoid -0 values

    # z: scalar or or arbitrary-dimensional numpy.ndarray of non-negative reals less than 1/gamma
    def psiInv(self, z):
        if np.any(z < 0.0):
            raise ValueError(
                'HawkesPseudoMemoryKernel.psiInv(): 1st argument (z) expected to be non=negative!')
        if np.any(z >= 1.0 / self.beta):
            raise ValueError(
                'HawkesPseudoMemoryKernel.psiInv(): 1st argument (z) expected to be less than 1/gamma!')
        value = np.log(1.0 - self.beta * z) / self.beta
        return value


################################################################################
# Pseudo- memory kernel for $t_o=0$: $phi(t) = e^{-\gamma t} [t > 0]$.
# Integrated memory kernel for $t_o=0$: $\psi(t) =  \frac{1}{\gamma}\left( 1 - e^{-\gamma t} \right) [t \geq 0]$
# Constraints: $\gamma > 0$
#
# Notes:
#   When $\gamma \to 0$, it becomes the constant kernel, but this is not implemented.
#
class HawkesPseudoMemoryKernel(MemoryKernel):
    '''Hawkes pseudo- memory kernel class'''

    # Instance variables
    #
    # desc:       string; description of the memory kernel; inherited from MemoryKernel base class.
    # timeOffset: real scalar; memory kernel's offset in time; inherited from MemoryKernel base class.
    # gamma:      positive real scalar; memory kernel parameter

    # gamma > 0
    def __init__(self, gamma, timeOffset=0.0):
        if gamma <= 0.0:
            raise ValueError(
                'HawkesPseudoMemoryKernel(): 2nd argument (gamma) must be strictly positive!')
        #desc = 'Hawkes pseudo- memory kernel: offset={:1}, gamma={:2}'.format(timeOffset, gamma)
        desc = 'Hawkes({:1}, {:2})'.format(timeOffset, gamma)
        super().__init__(desc, timeOffset)
        self.gamma = gamma

    # t: real scalar or arbitrary-dimensional numpy.ndarray of reals
    def phi(self, t, rightLimit=False):
        relative_t = t - self.timeOffset
        if rightLimit is False:
            isNotZero = np.array(relative_t > 0.0).astype(int)
        else:
            isNotZero = np.array(relative_t >= 0.0).astype(int)
        value = np.exp(- self.gamma * relative_t) * isNotZero
        return value

     # t1, t2: real scalars with t1<=t2
    def phiUB(self, t1, t2):
        if t2 < t1:  # not a necessary check; just precautionary
            raise ValueError(
                'HawkesPseudoMemoryKernel.phiUB(): 1st argument (t1) expected to be less or equal to 2nd argument (t2)!')
        if t1 < 0.0 < t2:
            ub = 1.0
        else:
            ub = max(self.phi(t1, rightLimit=True),
                     self.phi(t2, rightLimit=True))
        return ub

    # t: real scalar or arbitrary-dimensional numpy.ndarray of reals
    def psi(self, t):
        relative_t = t - self.timeOffset
        isNotZero = np.array(relative_t >= 0.0).astype(int)
        value = (1.0 - np.exp(- self.gamma * relative_t)) * \
            isNotZero / self.gamma
        return np.abs(value)  # to avoid -0 values

    # z: scalar or or arbitrary-dimensional numpy.ndarray of non-negative reals less than 1/gamma
    def psiInv(self, z):
        if np.any(z < 0.0):
            raise ValueError(
                'HawkesPseudoMemoryKernel.psiInv(): 1st argument (z) expected to be non=negative!')
        if np.any(z >= 1.0 / self.gamma):
            raise ValueError(
                'HawkesPseudoMemoryKernel.psiInv(): 1st argument (z) expected to be less than 1/gamma!')
        value = np.log(1.0 - self.gamma * z) / self.gamma
        return value


################################################################################
# Gompertz memory kernel for $t_o=0$: $phi(t) = e^{\gamma t} [t > 0]$.
# Integrated memory kernel for $t_o=0$: $\psi(t) =  \frac{1}{\gamma}\left( e^{\gamma t} - 1 \right) [t \geq 0]$
# Constraints: $\gamma > 0$
#
# Notes:
#   When $\gamma \to 0$, it becomes the constant kernel, but this is not implemented.
#
class GompertzMemoryKernel(MemoryKernel):
    '''Gompertz memory kernel class'''

    # Instance variables
    #
    # desc:       string; description of the memory kernel; inherited from MemoryKernel base class.
    # timeOffset: real scalar; memory kernel's offset in time; inherited from MemoryKernel base class.
    # gamma:      positive real scalar; memory kernel parameter

    # gamma > 0
    def __init__(self, gamma, timeOffset=0.0):
        if gamma <= 0.0:
            raise ValueError(
                'GompertzMemoryKernel(): 2nd argument (gamma) must be strictly positive!')
        #desc = 'Gompertz memory kernel: offset={:1}, gamma={:2}'.format(timeOffset, gamma)
        desc = 'Gompertz({:1}, {:2})'.format(timeOffset, gamma)
        super().__init__(desc, timeOffset)
        self.gamma = gamma

    # t: real scalar or arbitrary-dimensional numpy.ndarray of reals
    def phi(self, t, rightLimit=False):
        if type(t) is not list:
            relative_t = np.array(t - self.timeOffset)
        else:
            relative_t = np.array(t - self.timeOffset*np.ones(len(t)))
        if rightLimit is False:
            isNotZero = np.array(relative_t > 0.0).astype(int)
        else:
            isNotZero = np.array(relative_t >= 0.0).astype(int)
        value = np.exp(self.gamma * relative_t) * isNotZero
        return value

    # t1, t2: real scalars with t1<=t2
    def phiUB(self, t1, t2):
        if t2 < t1:  # not a necessary check; just precautionary
            raise ValueError(
                'GompertzMemoryKernel.phiUB(): 1st argument (t1) expected to be less or equal to 2nd argument (t2)!')
        ub = max(self.phi(t1, rightLimit=True), self.phi(t2, rightLimit=True))
        return ub

    # t: real scalar or arbitrary-dimensional numpy.ndarray of reals
    def psi(self, t):
        if type(t) is not list:
            relative_t = np.array(t - self.timeOffset)
        else:
            relative_t = np.array(t - self.timeOffset*np.ones(len(t)))
        isNotZero = np.array(relative_t >= 0.0).astype(int)
        value = (np.exp(self.gamma * relative_t) - 1.0) * \
            isNotZero / self.gamma
        return np.abs(value)  # to avoid -0 values

    # z: scalar or or arbitrary-dimensional numpy.ndarray of non-negative reals
    def psiInv(self, z):
        if np.any(z < 0.0):
            raise ValueError(
                'GompertzMemoryKernel.psiInv(): 1st argument (z) expected to be non=negative!')
        value = np.log(self.gamma * z + 1.0) / self.gamma
        return value


################################################################################
# Gamma-Gompertz memory kernel for $t_o=0$: $phi(t) = \frac{\beta}{1 + (\gamma - 1) e^{-\beta t}} [t > 0]$.
# Integrated memory kernel for $t_o=0$: $\psi(t) =  \ln\left( 1 + \frac{e^{\beta t} - 1}{\gamma} \right) [t \geq 0]$
# Constraints: $\beta > 0$ and $\gamma > 0$
#
# Notes:
#   When $\gamma = 1$, it becomes a scaled constant kernel.
#
class GammaGompertzMemoryKernel(MemoryKernel):
    '''Gamma-Gompertz memory kernel class'''

    # Instance variables
    #
    # desc:       string; description of the memory kernel; inherited from MemoryKernel base class.
    # timeOffset: real scalar; memory kernel's offset in time; inherited from MemoryKernel base class.
    # beta:      positive real scalar; memory kernel parameter
    # gamma:      positive real scalar; memory kernel parameter

    def __init__(self, beta, gamma, timeOffset=0.0):
        if beta <= 0.0:
            raise ValueError(
                'GammaGompertzMemoryKernel(): 2nd argument (beta) must be strictly positive!')
        if gamma <= 0.0:
            raise ValueError(
                'GammaGompertzMemoryKernel(): 3rd argument (gamma) must be strictly positive!')
        #desc = 'Gamma-Gompertz memory kernel: offest={:1}, beta={:2}, gamma={:3}'.format(timeOffset, beta, gamma)
        desc = 'GammaGompertz({:1}, {:2}, {:3})'.format(
            timeOffset, beta, gamma)
        super().__init__(desc, timeOffset)
        self.beta = beta
        self.gamma = gamma

    # t: real scalar or arbitrary-dimensional numpy.ndarray of reals
    def phi(self, t, rightLimit=False):
        relative_t = t - self.timeOffset
        if rightLimit is False:
            isNotZero = np.array(relative_t > 0.0).astype(int)
        else:
            isNotZero = np.array(relative_t >= 0.0).astype(int)
        value = (self.beta / (1.0 + (self.gamma - 1.0) *
                              np.exp(- self.beta * relative_t))) * isNotZero
        return value

    # t1, t2: real scalars with t1<=t2
    def phiUB(self, t1, t2):
        if t2 < t1:  # not a necessary check; just precautionary
            raise ValueError(
                'GammaGompertzMemoryKernel.phiUB(): 1st argument (t1) expected to be less or equal to 2nd argument (t2)!')
        if t1 < 0.0 and 0.0 < t2:
            return self.beta / self.gamma
        else:
            ub = max(self.phi(t1, rightLimit=True),
                     self.phi(t2, rightLimit=True))
        return ub

    # t: real scalar or arbitrary-dimensional numpy.ndarray of reals
    def psi(self, t):
        relative_t = np.array(t - self.timeOffset)
        isLessThanZero = (relative_t < 0.0)
        relative_t[isLessThanZero] = 0.0
        value = np.log(1.0 + (np.exp(self.beta * relative_t) - 1) / self.gamma)
        return value

    # z: scalar or or arbitrary-dimensional numpy.ndarray of non-negative reals
    def psiInv(self, z):
        if np.any(z < 0.0):
            raise ValueError(
                'GammaGompertzMemoryKernel.psiInv(): 1st argument (z) expected to be non=negative!')
        value = np.log(self.gamma * (np.exp(z) - 1.0) + 1.0) / self.beta
        return value


################################################################################
#
# CompositeMemoryKernel class definition & implementation.
#
# Allows for creating composite memory kernels (CMKs) by conically combining a set of fixed memory kernels.
# Given the combination's coeffieicents, it can compute various quantities of the CMK, such as
# overall (integrated) memory kernel, PDF, SF, etc. values.
#
# PUBLIC ATTRIBUTES:
#
#   desc:         string; description of the memory kernel; inherited from MemoryKernel base class.
#   timeOffset:   real scalar; the smallest time offset among the kernels that comprise the CMK; inherited from MemoryKernel base class.
#   KernelList:   list; a list of memory kernels (objects from MemoryKernel's sub-classes) that comprise the CMK object.
#   Alphas:       numpy.ndarray of shape (M,), where M is the number of kernels comprising the CMK object.
#
# PUBLIC INTERFACE:
#
#   CompositeMemoryKernel(): object; CMK object constructor.
#   addKernels():            None; adds one or more memory kernels to the CMK object.
#   numKernels():            non-negative integer; returns the number M of kernels comprising the CMK object.
#   isEmpty():               boolean; determines whether the CMK object is comprised of any kernels or not.
#
#   All methods listed below are inhereted from the MemoryKernel base class:
#
#   phi():          numpy.ndarray; CMK object's memory kernel $\phi(\cdot)$ values at specified time instances; is overidden.
#   phiUB():        non-negative scalar; upper bound of CMK object's memory kernel $\phi(\cdot)$ values in a given interval; is overriden.
#   psi():          numpy.ndarray; CMK object's integrated memory kernel $\psi(\cdot)$ values at specified time instances; is overidden.
#   psiInv():       numpy.ndarray; CMK object's inverse integrated memory kernel $\psi^{-1}(\cdot)$; is overriden.
#                   Not implemented yet for CMK object's consisting of more than one memory kernel.
#   pdf():          numpy.ndarray; CMK object's PDF values at specified time instances.
#   sf():           numpy.ndarray; CMK object's survival function (1-CDF) values at specified time instances.
#   cdf():          numpy.ndarray; CMK object's CDF values at specified time instances.
#   __str__():      string; string representation of CMK object.
#   print():        None; print string representation of CMK object.
#
# PRIVATE INTERFACE:
#
#   __chkAlphas():  None; currently, checking if the Alphas array has M (number of kernels in CMK object) elements.
#
# USAGE EXAMPLES:
#
#   mk1 = ConstantMemoryKernel()            # construct a constant kernel with time offset 0.0.
#   mk2 = WeibullMemoryKernel(0.5, 0.2)     # construct a Weibull kernel with gamma=0.5 and time offset 0.2.
#   cmk = CompositeMemoryKernel([mk1, mk2]) # construct a CMK object encompasing a constant and a Rayleigh kernel.
#   print(cmk)                              # print out the description of the CMK object.
#   cmk.Alphas = np.array([1.0, 2.0])       # specify conic combination coefficients for the CMK object.
#   t = np.array([0.0, 0.1, 0.2, 0.3, 0.4]) # choose some time instances.
#   print(cmk.sf(t))                        # print out the SF (survival function) values of the CMK.
#
# DEPENDENCIES:
#   imports: numpy as np
#   classes: MemoryKernel
#
# NOTES:
#   1) The numpy.ndarray returned by phi(), psi(), pdf(), sf() and cdf() matches the dimensions of t,
#      which is assumed to be a scalar or a numpy.ndarray.
#
# AUTHOR: Georgios C. Anagnostopoulos, January 2020 (version 1.1)
#
################################################################################
class CompositeMemoryKernel(MemoryKernel):
    '''Composite memory kernel class'''

    # Instance variables
    #
    # desc:       string; description of the memory kernel; inherited from MemoryKernel base class.
    # timeOffset: real scalar; memory kernel's offset in time; inherited from MemoryKernel base class; equals the minimum offset time among kernels of the CMK object
    # KernelList: list; a list of memory kernels (objects from MemoryKernel's sub-classes) that comprise the CMK object.
    # Alphas:     numpy.ndarray of shape (M,), where M is the number of kernels comprising the CMK object.

    # Construct a CompositeMemoryKernel object
    # Usage examples: cmk = CompositeMemoryKernel()            # as an empty CMK object
    #                 cmk = CompositeMemoryKernel(mk)          # with a single memory kernel object
    #                 cmk = CompositeMemoryKernel([mk1, mk2])  # with a list of memory kernel objects
    def __init__(self, ListOfMemKernels=[]):
        desc = 'CMK:'
        self.KernelList = []
        self.Alphas = np.array([])
        super().__init__(desc, timeOffset=None)
        if ListOfMemKernels != []:
            self.addKernels(ListOfMemKernels)

    # Checks if composite memory kernel has no kernel; return True or False
    # Usage example: myBoolean = cmk.isEmpty()
    def isEmpty(self):
        return self.KernelList == []

    # Return the number of kernels that make up the composite memory kernel.
    # Usage example: M = cmk.numKernels()
    def numKernels(self):
        return len(self.KernelList)

    # Add a single MemoryKernel sub-class object or a list of such objects to the composite memory kernel.
    # Sets the time offset of the CMK to the minimum of the kernels' time offsets.
    # Usage examples: cmk.addKernels(mk)
    #                 cmk.addKernels([mk1, mk2])
    def addKernels(self, kernelOrListOfKernels):
        if not isinstance(kernelOrListOfKernels, list):
            tmpList = [kernelOrListOfKernels]
        else:
            tmpList = kernelOrListOfKernels
        for kernel in tmpList:
            self.KernelList.append(kernel)
            self.desc = self.desc + ' ' + kernel.desc
            if self.timeOffset == None:
                self.timeOffset = kernel.timeOffset
            else:
                if self.timeOffset > kernel.timeOffset:
                    self.timeOffset = kernel.timeOffset

    # Checks whether the length of Alphas matches the number of kernels.
    def __chkAlphas(self):
        if len(self.Alphas) == 0:
            raise ValueError(
                'CompositeMemoryKernel: coefficients (Alphas) not yet set!')
        if len(self.Alphas) != len(self.KernelList):
            msg = 'CompositeMemoryKernel: coefficients (Alphas) must be a vector of {:1} elements'.format(
                len(self.KernelList))
            raise ValueError(msg)

    # t: real scalar or arbitrary-dimensional numpy.ndarray of reals
    def phi(self, t, rightLimit=False):
        self.__chkAlphas()  # Check validity of coefficients
        value = np.zeros_like(t)
        for a, kernel in zip(self.Alphas, self.KernelList):
            value += a * kernel.phi(t, rightLimit)
        return value

    # t1, t2: real scalars with t1<=t2
    # The upper bound obtained may be quite loose.
    def phiUB(self, t1, t2):
        if t2 < t1:  # not a necessary check; just precautionary
            raise ValueError(
                'CompositeMemoryKernel.phiUB(): 1st argument (t1) expected to be less or equal to 2nd argument (t2)!')
        ub = 0.0
        for a, kernel in zip(self.Alphas, self.KernelList):
            ub += a * kernel.phiUB(t1, t2)
        return ub

    # t: real scalar or arbitrary-dimensional numpy.ndarray of reals
    def psi(self, t):
        self.__chkAlphas()  # Check validity of coefficients
        value = np.zeros_like(t)
        for a, kernel in zip(self.Alphas, self.KernelList):
            value += a * kernel.psi(t)
        return value

    # z: scalar or or arbitrary-dimensional numpy.ndarray of non-negative reals
    def psiInv(self, z):
        if len(self.KernelList) == 1:
            # The composite kernel is made up of only one elementary kernel,
            # whose inverse psi() function is known. Hence, use the latter function.
            a = self.Alphas[0]
            kernel = self.KernelList[0]
            value = kernel.psiInv(z / a)
        else:
            # Finding the inverse psi() function of a composite kernel can only be
            # accomplished numerically via a root-finding algorithm.
            raise Exception(
                'CompositeMemoryKernel.psiInv() not implemented for a composite memory kernel consisting of more than one memory kernel!')
        return value


################################################################################
# Spline memory kernel for $t_o=0$: $phi(t) = $.
# Integrated memory kernel for $t_o=0$: $\psi(t) =  \$
# Constraints:
#
# Notes:
#    The spline kernel is built using the univariate spline functionality of scipy.
#    Although this kernel can be used to represent any sort of deterministic function,
#    this class is usually used to obtain the function associated with nelson-aalen estimates
#    at given time instances.
class SplineMemoryKernel(MemoryKernel):
    '''Spline memory kernel class'''
    # Instance variables
    #
    # timeOffset: real scalar; memory kernel's offset in time; inherited from MemoryKernel base class.
    # x_vals:      1D array of time values. It is the linspace (time range) used for obtaining Nelson-Aalen estimates.
    # y_vals:      1D array of values of the integrated memory kernel associated with each time instance in x_vals

    def __init__(self, x_vals, y_vals, timeOffset=0.0):
        assert len(x_vals) == len(
            y_vals), "Length of x_val and y_val must be the same"
        desc = 'Memory Kernel({:1})'.format(
            timeOffset)
        super().__init__(desc, timeOffset)
        # store x_vals and y_vals in the local object.
        self.x_vals = x_vals
        self.y_vals = y_vals
        # get index array for y_vals, indicated by 1 whenever value is nan
        # this array also functions as a weight vector after performing not operation,
        # i.e, ~w is the weights that will be sent to spline fitting.
        import pandas as pd
        w = pd.isnull(self.y_vals)
        # set the nan values to 0, so it doesnt affect spline fitting later on
        self.y_vals[w] = 0.
        # this values is set by default to 1.0 since all the coefficients for the
        # polynomials are present within the spline function itself.
        self.alpha = 1.0
        # fit the x_vals and y_vals with the associated weights ~w (not operator)
        self.spline = UnivariateSpline(x=self.x_vals, y=self.y_vals, w=~w, s=0.5, k=3)


    # t: real scalar or arbitrary-dimensional numpy.ndarray of reals
    def phi(self, t, rightLimit=False):
        # obtain the derivative of the spline function, which represents the Psi() function.
        der = self.spline.derivative(n=1)
        relative_t = t - self.timeOffset
        
        return der(relative_t)

    def AntiPhi(self, t, rightLimit=False):
        # obtain the derivative of the spline function, which represents the Psi() function.
        anti_der = self.spline.derivative(n=1).antiderivative()
        relative_t = t - self.timeOffset
        return anti_der(relative_t)
    # t1, t2: real scalars with t1<=t2
    #  since we have no data beyond a point
    #  compute derivate at final point and assume that Psi is an increasing linear function (continues in  a straight line)
    #  This is so that the derivate of that point is continuous.
    #  Derivative of the spline at the last point, phi will be constant,
    #  then the sampling distribution becomes an exponential distribution.
    #  make an assumption that
    def phiUB(self, t1, t2):
        raise NotImplementedError("This has not been implemented yet.")

    # t: real scalar or arbitrary-dimensional numpy.ndarray of reals
    def psi(self, t):
        relative_t = t - self.timeOffset
        #  return the associated value of the spline function, which represents the Psi function
        return self.spline(relative_t)

    # z: scalar or or arbitrary-dimensional numpy.ndarray of non-negative reals
    # use newtons methods to get the inverse
    def psiInv(self, z):
        # define a function that zeros out the value
        def interpolation_function(x): return self.spline(x) - z
        initial_estimate = z
        root = optimize.newton(interpolation_function, initial_estimate)
        return root
