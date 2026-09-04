Application Programming Interface
=================================

Batch
-----
.. autoclass:: aurora.Batch
    :members:

.. autoclass:: aurora.Metadata
    :members:

Roll-Outs
---------
.. autoclass:: aurora.rollout
    :members:

Tropical Cyclone Tracking
-------------------------
.. autoclass:: aurora.Tracker
    :members:

Models
------
.. autoclass:: aurora.Aurora
    :special-members: __init__
    :members:

.. autoclass:: aurora.AuroraPretrained
    :members:

.. autoclass:: aurora.AuroraSmallPretrained
    :members:

.. autoclass:: aurora.Aurora12hPretrained
    :members:

.. autoclass:: aurora.AuroraHighRes
    :members:

.. autoclass:: aurora.AuroraAirPollution
    :members:

.. autoclass:: aurora.AuroraWave
    :members:

.. autoclass:: aurora.AuroraV1p5
    :members:

.. autoclass:: aurora.AuroraV1p5Ensemble
    :members:

Latent Vectors
--------------
.. note::
    This section documents a feature of the RWE Supply & Trading fork.

.. autoclass:: aurora.latent.LatentCapture
    :special-members: __init__
    :members:

.. autofunction:: aurora.latent.capture_latents

.. autofunction:: aurora.latent.rollout_with_latents

.. autofunction:: aurora.latent.latents_to_grid

.. autofunction:: aurora.latent.patch_res_for
