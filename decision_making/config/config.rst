======================
Intro to Configuration
======================

This is meant to provide a basic outline of how to use configuration, 
as well as highligh some of the peculiarities of the configuration behavior as it interacts with the sampling and experiment declarations.

On terminology, please note an experiment is a set of parameters to be run, whereas a trial is one instance of an experiment.
As such, a single experiment may be conducted for multiple trials. 

Basic Configuration
###################

A configuration file will consist of a dictionary containing the following elements:
    - "default": (optional) default parameters for all algorithms or environments under test. These will be overwritten by more specific described below.

Users should also specify either other of below:
    - "algs": A list of algorithms with their specific parameters
    - "envs": A list of environments with their specific parameters
These should be kept in their own files. 

```
{
    "default": 
        {
            ...,
            "sample" : #(optional, see below)
            {

            },
        },
    "algs" :
        {
            [{}, {}, {}]
        }
}
```

Algorithm Configuration
***********************

Algorithms will have their own specific parameters, however, they should contain the following at a mininum:

```
{
    "alg": <alg name or reference>
    "whitelists" : #(optional)
    {
        "combo": 
        [
            "var1", "var2", ...
        ]
    },
    "sample" : #(optional, see below)
    {

    },
    "params" :
    {

    },
    "search" : 
    {

    }
}
```

A `combo` variable is necessary to signify which variables contain lists for combining into separate experiments 
(when used with sampling, if both are to be done to a variable, they should be specified here as well).

`params` specifies algorithm functionality at a core level

'search` specifies parameters they may change at runtime and are mean generally associated with evaluation of a solution.

Environment Configuration
*************************

Environments will have their own specific parameters, however, they should contain the following at a mininum:

```
{
    "env": <env name or reference>
    "whitelists" : #(optional)
    {
        "combo": 
        [
            "var1", "var2", ...
        ]
    },
    "sample" : #(optional, see below)
    {

    },
    "params" :
    {

    },
    "max_time" : <int for maximum number of timesteps>
}
```

A `combo` variable is necessary to signify which variables contain lists for combining into separate experiments 
(when used with sampling, if both are to be done to a variable, they should be specified here as well).

`params` specifies environment functionality


Sampling Parameters
###################

Users may wish to sample variables when running several experiments. 
As described above sampling may be specified in 
    - "default" : Here a single sample is drawn for each variable every trial and will not cannot be combined with other variables
    - "alg" or "env" : Here samples are drawn as lists, overwritting sample commands from default, and maybe be combined with other features for experiment generation.

Variables to be sampled are captured with a list as follows:
```
"sample : [ "var1", "var2", ...]
```

Within "default" or with each "alg"/"env", the corresponding variable should contain a dictionary rather than a single instance of the variable.
The dictionary will contain the information necessary to sample as desired. 
For example, discretely sampling "var1" would look something like:

```
{
    "alg": alg1,
    "params": 
    {
        "var1":
        {
            "choice": [1,2,3]
        }
    }
}
```

Sampling uses `NestifyDict <https://pypi.org/project/nestifydict/>`_ so variables can be specified as their deepest key assuming this variable is only used in one place. 
Otherwise the variable should be defined as a list.

Further detail on specifying samples can be found in :ref:`Sampler <sampler>`.