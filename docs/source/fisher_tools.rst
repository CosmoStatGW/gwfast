.. _fisher_tools:

Checking, manipulating and inverting the Fisher matrix
======================================================

``gwfast`` provides various functions to manipulate the FIM and deal with its inversion, to compute the *covariance matrix*. We here review and list the available tools.

To have more reliable estimations, most of the functions use the `mpmath library <https://mpmath.org>`_ for precision arithmetics.

.. note::
  All the following functions assume that the FIM is a 3-D array of matrices with shape :math:`(N_{\rm parameters}`, :math:`N_{\rm parameters}`, :math:`N_{\rm events})` where :math:`N_{\rm parameters}` is the number of parameters used in the analysis and :math:`N_{\rm events}` the number of simulated events.

Sanity checks on the Fisher matrix
----------------------------------

``gwfast`` features a function to compute the eigenvalues and eigenvectors of FIMs, checking if some are negative, as well as their *condition number*, i.e. the ratio of the largest to smallest eigenvalue.
If the condition number is larger than the inverse machine precision (:math:`\varepsilon^{-1} = 10^{15}` in our default case), the covariance matrix will likely be unreliable for some elements.

.. autofunction:: gwfast.fisherTools.CheckFisher

Computing the covariance matrix
-------------------------------

One of the most delicate point of the Fisher matrix analysis is dealing with its inversion, to compute the covariance matrix.
The inversion can be unreliable if the FIM is ill-conditioned, with some care in handling the inversion procedure, a better stability can be achieved.

In particular, in ``gwfast`` each row and column is normalized to the square root of the diagonal of the FIM before inversion, so that the resulting matrix has adimensional entries with ones on the diagonal and the remaining elements in the interval :math:`[−1,\, 1]`, as in `arXiv:2205.02499 <https://arxiv.org/abs/2205.02499>`_. The inverse transformation is then applied after the inversion to yield the inverse of the original matrix. This transformation is not applied in case the matrix has a zero element on the diagonal.

The `mpmath library <https://mpmath.org>`_ for precision arithmetics is employed, and various possible techniques are availeble to find the inverse, namely:

  - ``'inv'`` : the inverse as computed by `mpmath library <https://mpmath.org>`_;
  - ``'cho'`` : the inverse is computed by means of the `Cholesky decomposition <https://en.wikipedia.org/wiki/Cholesky_decomposition>`_, i.e. the (Hermitian, positive–definite) FIM is expressed as a product of a lower triangular matrix and its conjugate transpose, and the latter is inverted;

  .. note::
     In some case, the FIM may be not positive–definite due to the presence of very small eigenvalues that can assume negative values due to numerical fluctuations, in which case the Cholesky decomposition cannot be found.

  - ``'svd'`` : the `singular value decomposition <https://en.wikipedia.org/wiki/Singular_value_decomposition>`_, SVD, of the FIM is used to invert the matrix. In this case, there is the additional option of truncating the smallest singular values to the minimum allowed numerical precision, that can help regularizing badly conditioned matrices. In this case, for each singular value :math:`s_i`, if the ratio of its absolute value to the absolute value of the largest singular value, :math:`{\rm max}(s_i)`, is smaller than a threshold :math:`\lambda`, the singular value :math:`s_i` is replaced with :math:`\lambda \times {\rm max}(s_i)`.
  - ``'svd_reg'`` : the `singular value decomposition <https://en.wikipedia.org/wiki/Singular_value_decomposition>`_, SVD, of the FIM is used to invert the matrix, and eigenvalues smaller than a specified threshold are not included in the inversion. This ensures that the error on badly constrained parameters is not propagated to the other ones, see `arXiv:2205.02499 <https://arxiv.org/abs/2205.02499>`_. However, it might result in underestimating the uncertainty for the parameters whose eigenvalues are excluded, and the effect should be carefully checked.
  - ``'lu'`` : the nverse is computed by means of the `Lower-Upper <https://en.wikipedia.org/wiki/LU_decomposition>`_, LU, decomposition, i.e. the factorization of the FIM into the product of one lower triangular matrix and one upper triangular matrix. This can be useful since, as for the Cholesky decomposition, the inversion of a triangular matrix is easier than the one of a full matrix and, differently from the Cholesky decomposition, the original matrix does not have to be Hermitian and positive–definite, which can make this method more stable against numerical noise for badly–conditioned matrices.

A useful quantity that can be used to check the reliability of the inversion is the *inversion error*, defined as

.. math:: \epsilon = {\rm max}_{ij} |(\Gamma \cdot \Gamma^{-1} - \mathbb{1} )_{ij}|

The covariance matrix and the inversion error can be computed using the function

.. autofunction:: gwfast.fisherTools.CovMatr

It is also possible to compute the inversion error alone, if having already computed the Fisher and covariance matrices, using the function

.. autofunction:: gwfast.fisherTools.compute_inversion_error

Checks on the covariance matrix
-------------------------------

``gwfast`` features functions to check the stability of the FIM inversion.

To add a random perturbations to the FIM to a specified decimal point, and check if the inversion remains stable, it is possible to use

.. autofunction:: gwfast.fisherTools.perturb_Fisher

To compute the inversion error, print the difference between :math:`\Gamma \cdot \Gamma^{-1}` and the identity on the diagonal, and the off–diagonal elements of :math:`\Gamma \cdot \Gamma^{-1}` higher than a given threshold, it is possible to use

.. autofunction:: gwfast.fisherTools.check_covariance

Adding priors to the Fisher matrix
----------------------------------

To add a Gaussian prior on some parameters, one can add to the FIM a prior matrix :math:`P_{ij}` corresponding to the inverse covariance of the prior.
``gwfast`` supports the addition of a diagonal prior matrix, using the function

.. autofunction:: gwfast.fisherTools.addPrior

Fix parameters in the Fisher matrix
-----------------------------------

To fix some parameters to their fiducial values, one has to remove from the FIM the corresponding rows and columns before inverting it. This can done using the function

.. autofunction:: gwfast.fisherTools.fixParams

Compute the localisation region
-------------------------------

From the covariance matrix it is possible to compute the localisation region of the event. This is obtained as (see `arXiv:gr-qc/0310125 <https://arxiv.org/abs/gr-qc/0310125>`_ and `arXiv:1003.2504 <https://arxiv.org/abs/1003.2504>`_)

.. math:: \Delta\Omega_{{\rm X}\%} = -2\pi |{\rm sin}\theta|\sqrt{\left(\Gamma^{-1}\right)_{\theta\theta}\, \left(\Gamma^{-1}\right)_{\phi\phi} - \left(\Gamma^{-1}\right)_{\theta\phi}^2}\ {\rm ln}{\left(1 - \frac{{\rm X}}{100}\right)}

and can be computed using the function

.. autofunction:: gwfast.fisherTools.compute_localization_region

Changing parameters in Fisher and covariance matrix
---------------------------------------------------

``gwfast`` features functions to perform some parameters transformations of the Fisher and covariance matrices.

Transformation from :math:`{\rm log}(d_L)` to :math:`d_L`
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""

- To transform a Fisher matrix from :math:`{\rm log}(d_L)` to :math:`d_L` it is possible to use the function

  .. autofunction:: gwfast.fisherTools.log_dL_to_dL_derivative_fish

- To transform a covariance matrix from :math:`{\rm log}(d_L)` to :math:`d_L` it is possible to use the function

  .. autofunction:: gwfast.fisherTools.log_dL_to_dL_derivative_cov

Transformation from :math:`(m_1,\, m_2)` to :math:`({\cal M}_c,\, \eta)`
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

- To transform a Fisher matrix from :math:`m_1` and :math:`m_2` to :math:`{\cal M}_c` and :math:`\eta` it is possible to use the function

  .. autofunction:: gwfast.fisherTools.m1m2_to_Mceta_fish

- To transform a Fisher matrix from :math:`m_1` and :math:`m_2` to :math:`{\cal M}_c` and :math:`\eta` it is possible to use the function

  .. autofunction:: gwfast.fisherTools.m1m2_to_Mceta_cov

Transformation from :math:`({\cal M}_c,\, \eta)` to :math:`(m_1,\, m_2)`
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

- To transform a Fisher matrix from :math:`{\cal M}_c` and :math:`\eta` to :math:`m_1` and :math:`m_2` it is possible to use the function

  .. autofunction:: gwfast.fisherTools.Mceta_to_m1m2_fish

- To transform a Fisher matrix from :math:`{\cal M}_c` and :math:`\eta` to :math:`m_1` and :math:`m_2` it is possible to use the function

  .. autofunction:: gwfast.fisherTools.Mceta_to_m1m2_cov

Transformation from :math:`(\chi_s,\, \chi_a)` to :math:`(\chi_{1,z},\, \chi_{2,z})`
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

To transform a Fisher matrix from :math:`\chi_s` and :math:`\chi_a` to :math:`\chi_{1,z}` and :math:`\chi_{2,z}` it is possible to use the function

.. autofunction:: gwfast.fisherTools.chiSchiA_to_chi1chi2_fish

Transformation from :math:`(\chi_{1,z},\, \chi_{2,z})` to :math:`(\chi_{\rm eff},\, \Delta\chi)`
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

To transform a Fisher matrix from :math:`\chi_{1,z}` and :math:`\chi_{2,z}` to :math:`\chi_{\rm eff}` and :math:`\Delta\chi` it is possible to use the function

.. autofunction:: gwfast.fisherTools.chi1chi2_to_chieffDeltachi_fish

.. note::
  :math:`\chi_{\rm eff}` and :math:`\Delta\chi` are defined as

  .. math::
    :nowrap:

    \begin{eqnarray}
       \chi_{\rm eff}  & \equiv & \frac{m_1 \chi_{1,z} + m_2 \chi_{2,z}}{m_1 + m_2} \\
       \Delta\chi      & \equiv & \chi_{1,z} - \chi_{2,z} = 2 \chi_a
    \end{eqnarray}
