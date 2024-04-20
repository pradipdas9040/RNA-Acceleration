import warnings
from typing import Optional, Union,Tuple, List

import numpy as np
from numpy import ndarray
from scipy.sparse import issparse, spmatrix, csr_matrix

def clipped_log(x: ndarray, lb: float = 0, ub: float = 1, eps: float = 1e-6) -> ndarray:
    """Logarithmize between [lb + epsilon, ub - epsilon].

    Arguments
    ---------
    x
        Array to invert.
    lb
        Lower bound of interval to which array entries are clipped.
    ub
        Upper bound of interval to which array entries are clipped.
    eps
        Offset of boundaries of clipping interval.

    Returns
    -------
    ndarray
        Logarithm of clipped array.
    """
    return np.log(np.clip(x, lb + eps, ub - eps))


def invert(x: ndarray) -> ndarray:
    """Invert array and set infinity to NaN.

    Arguments
    ---------
    x
        Array to invert.

    Returns
    -------
    ndarray
        Inverted array.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        x_inv = 1 / x * (x != 0)
    return x_inv


def multiply(
    a: Union[ndarray, spmatrix], b: Union[ndarray, spmatrix]
) -> Union[ndarray, spmatrix]:
    """Point-wise multiplication of arrays or sparse matrices.

    Arguments
    ---------
    a
        First array/sparse matrix.
    b
        Second array/sparse matrix.

    Returns
    -------
    Union[ndarray, spmatrix]
        Point-wise product of `a` and `b`.
    """
    if issparse(a):
        return a.multiply(b)
    elif issparse(b):
        return b.multiply(a)
    else:
        return a * b


def prod_sum(
    a1: Union[ndarray, spmatrix], a2: Union[ndarray, spmatrix], axis: Optional[int]
) -> ndarray:
    """Take sum of product of two arrays along given axis.

    Arguments
    ---------
    a1
        First array.
    a2
        Second array.
    axis
        Axis along which to sum elements. If `None`, all elements will be summed.
        Defaults to `None`.

    Returns
    -------
    ndarray
        Sum of product of arrays along given axis.
    """
    if issparse(a1):
        return a1.multiply(a2).sum(axis=axis).A1
    elif axis == 0:
        return np.einsum("ij, ij -> j", a1, a2) if a1.ndim > 1 else (a1 * a2).sum()
    elif axis == 1:
        return np.einsum("ij, ij -> i", a1, a2) if a1.ndim > 1 else (a1 * a2).sum()


def sum(a: Union[ndarray, spmatrix], axis: Optional[int] = None) -> ndarray:
    """Sum array elements over a given axis.

    Arguments
    ---------
    a
        Elements to sum.
    axis
        Axis along which to sum elements. If `None`, all elements will be summed.
        Defaults to `None`.

    Returns
    -------
    ndarray
        Sum of array along given axis.
    """
    if a.ndim == 1:
        axis = 0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return a.sum(axis=axis).A1 if issparse(a) else a.sum(axis=axis)

class ExpLinearRegression:
    """Extreme quantile and constraint least square linear regression.

    Arguments
    ---------
    percentile
        Percentile of data on which linear regression line is fit. If `None`, all data
        is used, if a single value is given, it is interpreted as the upper quantile.
        Defaults to `None`.
    fit_intercept
        Whether to calculate the intercept for model. Defaults to `False`.
    positive_intercept
        Whether the intercept it constraint to positive values. Only plays a role when
        `fit_intercept=True`. Defaults to `True`.
    constrain_ratio
        Ratio to which coefficients are clipped. If `None`, the coefficients are not
        constraint. Defaults to `None`.

    Attributes
    ----------
    `coef_`
        Estimated coefficients of the linear regression line.

    `intercept_`
        Fitted intercept of linear model. Set to `0.0` if `fit_intercept=False`.

    """

    def __init__(
        self,
        percentile: Optional[Union[Tuple, int, float]] = None,
        fit_intercept: bool = False,
        positive_intercept: bool = True,
        constrain_ratio: Optional[Union[Tuple, float]] = None,
    ):
        if not fit_intercept and isinstance(percentile, (list, tuple)):
            self.percentile = percentile[1]
        else:
            self.percentile = percentile
        self.fit_intercept = fit_intercept
        self.positive_intercept = positive_intercept

        if constrain_ratio is None:
            self.constrain_ratio = [-np.inf, np.inf]
        elif len(constrain_ratio) == 1:
            self.constrain_ratio = [-np.inf, constrain_ratio]
        else:
            self.constrain_ratio = constrain_ratio

    def _trim_data(self, data: List) -> List:
        """Trim data to extreme values.

        Arguments
        ---------
        data
            Data to be trimmed to extreme quantiles.

        Returns
        -------
        List
            Number of non-trivial entries per column and trimmed data.
        """
        if not isinstance(data, List):
            data = [data]

        data = np.array(
            [data_mat.A if issparse(data_mat) else data_mat for data_mat in data]
        )

        # TODO: Add explanatory comment
        normalized_data = np.sum(
            data / data.max(axis=1, keepdims=True).clip(1e-3, None), axis=0
        )

        bound = np.percentile(normalized_data, self.percentile, axis=0)

        if bound.ndim == 1:
            trimmer = csr_matrix(normalized_data >= bound).astype(bool)
        else:
            trimmer = csr_matrix(
                (normalized_data <= bound[0]) | (normalized_data >= bound[1])
            ).astype(bool)

        return [trimmer.getnnz(axis=0)] + [
            trimmer.multiply(data_mat).tocsr() for data_mat in data
        ]

    def fit(self, x: ndarray, y: ndarray):
        """Fit linear model per column.

        Arguments
        ---------
        x
            Training data of shape `(n_obs, n_vars)`.
        y
            Target values of shape `(n_obs, n_vars)`.

        Returns
        -------
        self
            Returns an instance of self.
        """
        n_obs = x.shape[0]

        if self.percentile is not None:
            n_obs, x, y = self._trim_data(data=[x, y])
        
        x = np.exp(x)
        y = np.exp(y)

        _xx = prod_sum(x, x, axis=0)
        _xy = prod_sum(x, y, axis=0)

        if self.fit_intercept:
            _x = sum(x, axis=0) / n_obs
            _y = sum(y, axis=0) / n_obs
            self.coef_ = (_xy / n_obs - _x * _y) / (_xx / n_obs - _x**2)
            self.intercept_ = _y - self.coef_ * _x

            if self.positive_intercept:
                idx = self.intercept_ < 0
                if self.coef_.ndim > 0:
                    self.coef_[idx] = _xy[idx] / _xx[idx]
                else:
                    self.coef_ = _xy / _xx
                self.intercept_ = np.clip(self.intercept_, 0, None)
        else:
            self.coef_ = _xy / _xx
            self.intercept_ = np.zeros(x.shape[1]) if x.ndim > 1 else 0

        if not np.isscalar(self.coef_):
            self.coef_[np.isnan(self.coef_)] = 0
            self.intercept_[np.isnan(self.intercept_)] = 0
        else:
            if np.isnan(self.coef_):
                self.coef_ = 0
            if np.isnan(self.intercept_):
                self.intercept_ = 0

        self.coef_ = np.clip(self.coef_, *self.constrain_ratio)

        return self