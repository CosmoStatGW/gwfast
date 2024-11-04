.. _data_gen:

Creating and storing data
=========================

.. _data_dict:

The events dictionary
---------------------

The ``gwfast`` functions take as input dictionaries containing the parameters of the events to analyse.
As an example, the dictionary can be structured as


.. py:data:: events

  :type: dict(array, array, ...)

  events = {``'Mc'``:np.array([...]), ``'eta'``:np.array([...]), ``'dL'``:np.array([...]), ``'theta'``:np.array([...]), ``'phi'``:np.array([...]), ``'iota'``:np.array([...]), ``'psi'``:np.array([...]), ``'tcoal'``:np.array([...]), ``'Phicoal'``:np.array([...]), ``'chi1x'``:np.array([...]), ``'chi2x'``:np.array([...]), ``'chi1y'``:np.array([...]), ``'chi2y'``:np.array([...]), ``'chi1z'``:np.array([...]), ``'chi2z'``:np.array([...]), ``'LambdaTilde'``:np.array([...]), ``'deltaLambda'``:np.array([...]), ``'ecc'``:np.array([...])}

.. note::
  The arrays in the :py:data:`events` dictionary have to be 1-D and all of the same size.

.. _parameters_names:

Parameters names and conventions
--------------------------------

Here we report the naming conventions used in ``gwfast``, as well as the units of the parameters and their physical range

.. table::

  +-------------------------------+-------------------------------+-------------------------------+----------------------+----------------------------------------------+
  | Parameter symbol              | Parameter name                | Parameter description         |  Units in ``gwfast`` | Physical range                               |
  +===============================+===============================+===============================+======================+==============================================+
  | :math:`{\cal M}_c`            | ``'Mc'``                      | detector-frame chirp mass     | :math:`\rm M_{\odot}`| :math:`(0,\,+\infty)`                        |
  +-------------------------------+-------------------------------+-------------------------------+----------------------+----------------------------------------------+
  | :math:`\eta`                  | ``'eta'``                     | symmetric mass ratio          | --                   | :math:`(0,\,0.25]`                           |
  +-------------------------------+-------------------------------+-------------------------------+----------------------+----------------------------------------------+
  | :math:`m_1`                   | ``'m1'``                      | detector-frame primary mass   | :math:`\rm M_{\odot}`| :math:`(0,\,+\infty)`                        |
  |                               |                               | :math:`(m_1 \geq m_2)`        |                      |                                              |
  +-------------------------------+-------------------------------+-------------------------------+----------------------+----------------------------------------------+
  | :math:`m_2`                   | ``'m2'``                      | detector-frame secondary mass | :math:`\rm M_{\odot}`| :math:`(0,\, m_1]`                           |
  |                               |                               | :math:`(m_2 \leq m_1)`        |                      |                                              |
  +-------------------------------+-------------------------------+-------------------------------+----------------------+----------------------------------------------+
  | :math:`d_L`                   | ``'dL'``                      | luminosity distance           | :math:`\rm Gpc`      | :math:`(0,\,+\infty)`                        |
  +-------------------------------+-------------------------------+-------------------------------+----------------------+----------------------------------------------+
  | :math:`\theta,\, \phi`        | ``'theta'``, ``'phi'``        | sky position                  | :math:`\rm rad`      | :math:`[0,\,\pi]`, :math:`[0,\,2\pi]`        |
  +-------------------------------+-------------------------------+-------------------------------+----------------------+----------------------------------------------+
  | :math:`\alpha,\, \delta`      | ``'ra'``, ``'dec'``           | sky position (*astro.*)       | :math:`\rm rad`      | :math:`[0,\,2\pi]`, :math:`[-\pi/2,\,\pi/2]` |
  +-------------------------------+-------------------------------+-------------------------------+----------------------+----------------------------------------------+
  |                               |                               | inclination angle with        |                      |                                              |
  | :math:`\iota`                 | ``'iota'``                    | respect to orbital            | :math:`\rm rad`      | :math:`[0,\,\pi]`                            |
  |                               |                               | angular momentum              |                      |                                              |
  +-------------------------------+-------------------------------+-------------------------------+----------------------+----------------------------------------------+
  |                               |                               | inclination angle with        |                      |                                              |
  | :math:`\theta_{JN}`           | ``'thetaJN'``                 | respect to total              | :math:`\rm rad`      | :math:`[0,\,\pi]`                            |
  |                               |                               | angular momentum              |                      |                                              |
  +-------------------------------+-------------------------------+-------------------------------+----------------------+----------------------------------------------+
  | :math:`\psi`                  | ``'psi'``                     | polarisation angle            | :math:`\rm rad`      | :math:`[0,\,\pi]`                            |
  +-------------------------------+-------------------------------+-------------------------------+----------------------+----------------------------------------------+
  | :math:`t_{c, {\rm GMST}}`     | ``'tcoal'``                   | GMST time of                  | :math:`\rm day`      | :math:`[0,\,1]`                              |
  |                               |                               | coalescence                   |                      |                                              |
  +-------------------------------+-------------------------------+-------------------------------+----------------------+----------------------------------------------+
  | :math:`t_{c, {\rm GPS}}`      | ``'tGPS'``                    | GPS time of                   | :math:`\rm s`        | :math:`[0,\,+\infty)`                        |
  |                               |                               | coalescence                   |                      |                                              |
  +-------------------------------+-------------------------------+-------------------------------+----------------------+----------------------------------------------+
  | :math:`\Phi_c`                | ``'Phicoal'``                 | phase at coalescence          | :math:`\rm rad`      | :math:`[0,\,2\pi]`                           |
  +-------------------------------+-------------------------------+-------------------------------+----------------------+----------------------------------------------+
  | :math:`\chi_{1,x}`            | ``'chi1x'``                   | spin of object 1              | --                   | :math:`[-1,\,1]`                             |
  |                               |                               | along the axis :math:`x`      |                      |                                              |
  +-------------------------------+-------------------------------+-------------------------------+----------------------+----------------------------------------------+
  | :math:`\chi_{2,x}`            | ``'chi2x'``                   | spin of object 2              | --                   | :math:`[-1,\,1]`                             |
  |                               |                               | along the axis :math:`x`      |                      |                                              |
  +-------------------------------+-------------------------------+-------------------------------+----------------------+----------------------------------------------+
  | :math:`\chi_{1,y}`            | ``'chi1y'``                   | spin of object 1              | --                   | :math:`[-1,\,1]`                             |
  |                               |                               | along the axis :math:`y`      |                      |                                              |
  +-------------------------------+-------------------------------+-------------------------------+----------------------+----------------------------------------------+
  | :math:`\chi_{2,y}`            | ``'chi2y'``                   | spin of object 2              | --                   | :math:`[-1,\,1]`                             |
  |                               |                               | along the axis :math:`y`      |                      |                                              |
  +-------------------------------+-------------------------------+-------------------------------+----------------------+----------------------------------------------+
  | :math:`\chi_{1,z}`            | ``'chi1z'``                   | spin of object 1              | --                   | :math:`[-1,\,1]`                             |
  |                               |                               | along the axis :math:`z`      |                      |                                              |
  +-------------------------------+-------------------------------+-------------------------------+----------------------+----------------------------------------------+
  | :math:`\chi_{2,z}`            | ``'chi2z'``                   | spin of object 2              | --                   | :math:`[-1,\,1]`                             |
  |                               |                               | along the axis :math:`z`      |                      |                                              |
  +-------------------------------+-------------------------------+-------------------------------+----------------------+----------------------------------------------+
  | :math:`\chi_s`                | ``'chiS'``                    | symmetric spin                | --                   | :math:`[-1,\,1]`                             |
  |                               |                               | component                     |                      |                                              |
  +-------------------------------+-------------------------------+-------------------------------+----------------------+----------------------------------------------+
  | :math:`\chi_a`                | ``'chiA'``                    | asymmetric spin               | --                   | :math:`[-1,\,1]`                             |
  |                               |                               | component                     |                      |                                              |
  +-------------------------------+-------------------------------+-------------------------------+----------------------+----------------------------------------------+
  | :math:`\chi_1`                | ``'chi1'``                    | spin magnitude of             | --                   | :math:`[0,\,1]`                              |
  |                               |                               | object 1                      |                      |                                              |
  +-------------------------------+-------------------------------+-------------------------------+----------------------+----------------------------------------------+
  | :math:`\chi_2`                | ``'chi2'``                    | spin magnitude of             | --                   | :math:`[0,\,1]`                              |
  |                               |                               | object 2                      |                      |                                              |
  +-------------------------------+-------------------------------+-------------------------------+----------------------+----------------------------------------------+
  | :math:`\theta_{s,1}`          | ``'tilt1'``                   | spin tilt of                  | :math:`\rm rad`      | :math:`[0,\,\pi]`                            |
  |                               |                               | object 1                      |                      |                                              |
  +-------------------------------+-------------------------------+-------------------------------+----------------------+----------------------------------------------+
  | :math:`\theta_{s,2}`          | ``'tilt2'``                   | spin tilt of                  | :math:`\rm rad`      | :math:`[0,\,\pi]`                            |
  |                               |                               | object 2                      |                      |                                              |
  +-------------------------------+-------------------------------+-------------------------------+----------------------+----------------------------------------------+
  |                               |                               | azimuthal angle of            |                      |                                              |
  | :math:`\phi_{JL}`             | ``'phiJL'``                   | orbital angular               | :math:`\rm rad`      | :math:`[0,\,2\pi]`                           |
  |                               |                               | momentum relative to          |                      |                                              |
  |                               |                               | total angular momentum        |                      |                                              |
  +-------------------------------+-------------------------------+-------------------------------+----------------------+----------------------------------------------+
  |                               |                               | difference in                 |                      |                                              |
  | :math:`\phi_{1,2}`            | ``'phi12'``                   | azimuthal angle               | :math:`\rm rad`      | :math:`[0,\,2\pi]`                           |
  |                               |                               | between spin vectors          |                      |                                              |
  +-------------------------------+-------------------------------+-------------------------------+----------------------+----------------------------------------------+
  |                               |                               | adimensional tidal            |                      |                                              |
  | :math:`\Lambda_1`             | ``'Lambda1'``                 | deformability of              | --                   | :math:`[0,\,+\infty)`                        |
  |                               |                               | object 1                      |                      |                                              |
  +-------------------------------+-------------------------------+-------------------------------+----------------------+----------------------------------------------+
  |                               |                               | adimensional tidal            |                      |                                              |
  | :math:`\Lambda_2`             | ``'Lambda2'``                 | deformability of              | --                   | :math:`[0,\,+\infty)`                        |
  |                               |                               | object 2                      |                      |                                              |
  +-------------------------------+-------------------------------+-------------------------------+----------------------+----------------------------------------------+
  |                               |                               | adimensional tidal            |                      |                                              |
  | :math:`\tilde{\Lambda}`       | ``'LambdaTilde'``             | deformability of              | --                   | :math:`[0,\,+\infty)`                        |
  |                               |                               | combination                   |                      |                                              |
  |                               |                               | :math:`\tilde{\Lambda}`       |                      |                                              |
  +-------------------------------+-------------------------------+-------------------------------+----------------------+----------------------------------------------+
  |                               |                               | adimensional tidal            |                      |                                              |
  | :math:`\delta\tilde{\Lambda}` | ``'deltaLambda'``             | deformability of              | --                   | :math:`(-\infty,\,+\infty)`                  |
  |                               |                               | combination                   |                      |                                              |
  |                               |                               | :math:`\delta\tilde{\Lambda}` |                      |                                              |
  +-------------------------------+-------------------------------+-------------------------------+----------------------+----------------------------------------------+
  |                               |                               | orbital eccentricity at       |                      |                                              |
  | :math:`e_0`                   | ``'ecc'``                     | reference frequency           | --                   | :math:`[0,\,1)`                              |
  |                               |                               | :math:`f_{e_{0}}`             |                      |                                              |
  +-------------------------------+-------------------------------+-------------------------------+----------------------+----------------------------------------------+

.. warning::
  The spin components are defined on a sphere, i.e. they have to satisfy :math:`\chi_{1,x}^2 + \chi_{1,y}^2 + \chi_{1,z}^2\leq 1` and :math:`\chi_{2,x}^2 + \chi_{2,y}^2 + \chi_{2,z}^2\leq 1`.

.. note::
  The symmetric and asymmetric spin components are defined as

  .. math::
   :nowrap:

   \begin{eqnarray}
      \chi_s    & = & \frac{1}{2} (\chi_{1,z} + \chi_{2,z}) \\
      \chi_a    & = & \frac{1}{2} (\chi_{1,z} - \chi_{2,z})
   \end{eqnarray}

.. _note_labdatildeDef:
.. note::
  The adimensional tidal deformability combinations :math:`\tilde{\Lambda}` and :math:`\delta\tilde{\Lambda}` are defined as (see `arXiv:1402.5156 <https://arxiv.org/abs/1402.5156>`_).

  .. math::
   :nowrap:

   \begin{eqnarray}
      \tilde{\Lambda}       & = & \dfrac{8}{13} \left[(1+7\eta-31\eta^2)(\Lambda_1 + \Lambda_2) + \sqrt{1-4\eta}(1+9\eta-11\eta^2)(\Lambda_1 - \Lambda_2)\right] \\
      \delta\tilde{\Lambda} & = & \dfrac{1}{2} \left[\sqrt{1-4\eta} \left(1-\dfrac{13272}{1319}\eta + \dfrac{8944}{1319}\eta^2\right)(\Lambda_1 + \Lambda_2) + \right. \\
                            &   & \ \ + \left. \left(1 - \dfrac{15910}{1319}\eta + \dfrac{32850}{1319}\eta^2 + \dfrac{3380}{1319}\eta^3\right)(\Lambda_1 - \Lambda_2)\right]
   \end{eqnarray}


.. _data_storage:

Save and load data
------------------

To store a dictionary containing data in ``h5`` format, it is possible to use the function

.. autofunction:: gwfast.gwfastUtils.save_data

.. note::
  See `<https://www.h5py.org>`_ for details on the ``h5`` binary format.

The :py:class:`gwfast.gwfastUtils.load_population` function can instead be used to load a dictionary with the parameters, in ``h5`` format

.. autofunction:: gwfast.gwfastUtils.load_population


To select a subsample of events from a catalog it is possible to use the functions

.. autofunction:: gwfast.gwfastUtils.get_event

.. autofunction:: gwfast.gwfastUtils.get_events_subset

Convert between parameters
--------------------------

``gwfast`` provides some useful functions to convert between different parameters. All of them are vectorised, and can thus be used on arrays containing the parameters of multiple events.

Sky position angles
"""""""""""""""""""

Conversions between the sky position angles :math:`(\theta,\, \phi)` and :math:`(\alpha,\, \delta)`

.. autofunction:: gwfast.gwfastUtils.ra_dec_from_th_phi_rad

.. autofunction:: gwfast.gwfastUtils.th_phi_from_ra_dec_rad

.. autofunction:: gwfast.gwfastUtils.ra_dec_from_th_phi

.. autofunction:: gwfast.gwfastUtils.th_phi_from_ra_dec

.. autofunction:: gwfast.gwfastUtils.theta_to_dec_degminsec

.. autofunction:: gwfast.gwfastUtils.phi_to_ra_degminsec

.. autofunction:: gwfast.gwfastUtils.phi_to_ra_hrms


Tidal deformability parameters
""""""""""""""""""""""""""""""

Conversions between the individual tidal deformabilities of the two objects :math:`\Lambda_1` and :math:`\Lambda_2` and the combinations :math:`\tilde{\Lambda}` and :math:`\delta\tilde{\Lambda}` (see the :ref:`previous note <note_labdatildeDef>`).

.. autofunction:: gwfast.gwfastUtils.Lamt_delLam_from_Lam12

.. autofunction:: gwfast.gwfastUtils.Lam12_from_Lamt_delLam

Masses
""""""

Conversions between the component masses and the chirp mass and symmetric mass ratio.

.. autofunction:: gwfast.gwfastUtils.m1m2_from_Mceta

.. autofunction:: gwfast.gwfastUtils.Mceta_from_m1m2

Precessing spins
""""""""""""""""

Conversions between the components of the spins in cartesian frame given and the angular variables (see the :ref:`parameters table <parameters_names>`).

.. autofunction:: gwfast.gwfastUtils.TransformPrecessing_angles2comp

.. autofunction:: gwfast.gwfastUtils.TransformPrecessing_comp2angles

Time
""""

Conversion between the GPS time and the *Local Mean Sidereal Time* (LMST).

.. autofunction:: gwfast.gwfastUtils.GPSt_to_LMST

.. note::

  The *Greenwich Mean Sidereal Time* (GMST) is the LMST computed at longitude = 0°. To obtain this quantity it is then sufficient to use

  .. code-block:: python

    >>> gwfast.gwfastUtils.GPSt_to_LMST(t_GPS, lat=0., long=0.)

.. note::

  It is possible to associate a GMST to each GPS time, but to each GMST an infinite number of GPS times is associated for periodicity, thus the inverse function is not provided.
  
.. note::
  .. versionadded:: 1.1.2
  
  To avoid errors that can arise from the `astropy <https://www.astropy.org>`_ implementation when using times in the far future (having no `IERS <https://www.iers.org/IERS/EN/Home/home_node.html>`_ data), we provide an **approximate** function to compute the GMST.

  .. autofunction:: gwfast.gwfastUtils.GPSt_to_GMST_alt
