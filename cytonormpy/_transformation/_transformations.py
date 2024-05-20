from abc import abstractmethod, ABC
import numpy as np
from typing import Optional, Union

from flowutils.transforms import (logicle,
                                  logicle_inverse,
                                  hyperlog,
                                  hyperlog_inverse,
                                  log,
                                  log_inverse)


class Transformer(ABC):
    _channel_indices: Optional[Union[list[int], np.ndarray]]

    def __init__(self,
                 channel_indices: Optional[Union[list[int], np.ndarray]]
                 ) -> None:
        self._channel_indices = channel_indices

    @abstractmethod
    def transform(self, data: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        pass

    @property
    def channel_indices(self):
        return self._channel_indices

    @channel_indices.setter
    def channel_indices(self,
                        channel_indices: Optional[Union[list[int], np.ndarray]]
                        ) -> None:
        self._channel_indices = channel_indices

    @channel_indices.deleter
    def channel_indices(self):
        del self._channel_indices


class LogicleTransformer(Transformer):
    """\
    Class to apply the Logicle Transform.

    Parameters
    ----------
    channel_indices
        The channels to be transformed. Specifying None will
        transform all channels. Defaults to None. Note that
        in cytonormpy, channel indices will be set automatically
        by the CytoNorm object.
    t
        Parameter for the top of the linear scale. Defaults to 262144.
    m
        Parameter for the number of decades the true logarithmic scale
        approaches at the high end of the scale
    w
        Parameter for the approximate number of decades in the
        linear region
    a
        Parameter for the number of additional negative decades.

    Returns
    -------
    None

    """

    def __init__(self,
                 channel_indices: Optional[Union[list[int], np.ndarray]] = None,  # noqa
                 t: int = 262144,
                 m: float = 4.5,
                 w: float = 0.5,
                 a: int = 0):
        super().__init__(channel_indices)
        self.t = t
        self.m = m
        self.w = w
        self.a = a

    def transform(self,
                  data: np.ndarray) -> np.ndarray:
        """\
        Applies logicle transform to channels specified in
        `.channel_indices`. For further documentation refer to the
        `flowutils` package.


        Parameters
        ----------
        data
            The data to transform.

        Returns
        -------
        :class:`~numpy.ndarray`

        """
        return logicle(
            data = data,
            channel_indices = self.channel_indices,
            t = self.t,
            m = self.m,
            w = self.w,
            a = self.a
        )

    def inverse_transform(self,
                          data: np.ndarray) -> np.ndarray:
        """\
        Applies inverse logicle transform to channels specified in
        `.channel_indices`. For further documentation refer to the
        `flowutils` package.

        Parameters
        ----------
        data
            The data to transform.

        Returns
        -------
        :class:`~numpy.ndarray`
        """
        return logicle_inverse(
            data = data,
            channel_indices = self.channel_indices,
            t = self.t,
            m = self.m,
            w = self.w,
            a = self.a
        )


class HyperLogTransformer(Transformer):
    """\
    Class to apply the HyperLogTransform. For further documentation
    refer to the flowutils documentation.

    Parameters
    ----------
    channel_indices
        The channels to be transformed. Specifying None will
        transform all channels. Defaults to None. Note that
        in cytonormpy, channel indices will be set automatically
        by the CytoNorm object.
    t
        Parameter for the top of the linear scale. Defaults to 262144.
    m
        Parameter for the number of decades the true logarithmic scale
        approaches at the high end of the scale
    w
        Parameter for the approximate number of decades in the
        linear region
    a
        Parameter for the number of additional negative decades.

    Returns
    -------
    None

    """

    def __init__(self,
                 channel_indices: Optional[Union[list[int], np.ndarray]] = None,  # noqa
                 t: int = 262144,
                 m: float = 4.5,
                 w: float = 0.5,
                 a: int = 0):
        super().__init__(channel_indices)
        self.t = t
        self.m = m
        self.w = w
        self.a = a

    def transform(self,
                  data: np.ndarray) -> np.ndarray:
        """\
        Applies hyperlog transform to channels specified in
        `.channel_indices`. For further documentation refer to the
        `flowutils` package.


        Parameters
        ----------
        data
            The data to transform.

        Returns
        -------
        :class:`~numpy.ndarray`

        """
        return hyperlog(
            data = data,
            channel_indices = self.channel_indices,
            t = self.t,
            m = self.m,
            w = self.w,
            a = self.a
        )

    def inverse_transform(self,
                          data: np.ndarray) -> np.ndarray:
        """\
        Applies inverse hyperlog transform to channels specified in
        `.channel_indices`. For further documentation refer to the
        `flowutils` package.

        Parameters
        ----------
        data
            The data to transform.

        Returns
        -------
        :class:`~numpy.ndarray`
        """
        return hyperlog_inverse(
            data = data,
            channel_indices = self.channel_indices,
            t = self.t,
            m = self.m,
            w = self.w,
            a = self.a
        )


class LogTransformer(Transformer):
    """\
    Class to apply the LogTransform. For further documentation
    refer to the flowutils documentation.

    Parameters
    ----------
    channel_indices
        The channels to be transformed. Specifying None will
        transform all channels. Defaults to None. Note that
        in cytonormpy, channel indices will be set automatically
        by the CytoNorm object.
    t
        Parameter for the top of the linear scale. Defaults to 262144.
    m
        Parameter for the number of decades the true logarithmic scale
        approaches at the high end of the scale

    Returns
    -------
    None

    """

    def __init__(self,
                 channel_indices: Optional[Union[list[int], np.ndarray]] = None,  # noqa
                 t: int = 262144,
                 m: float = 4.5) -> None:
        super().__init__(channel_indices)
        self.t = t
        self.m = m

    def transform(self,
                  data: np.ndarray) -> np.ndarray:
        """\
        Applies log transform to channels specified in
        `.channel_indices`. For further documentation refer to the
        `flowutils` package.


        Parameters
        ----------
        data
            The data to transform.

        Returns
        -------
        :class:`~numpy.ndarray`

        """
        return log(
            data = data,
            channel_indices = self.channel_indices,
            t = self.t,
            m = self.m
        )

    def inverse_transform(self,
                          data: np.ndarray) -> np.ndarray:
        """\
        Applies inverse hyperlog transform to channels specified in
        `.channel_indices`. For further documentation refer to the
        `flowutils` package.

        Parameters
        ----------
        data
            The data to transform.

        Returns
        -------
        :class:`~numpy.ndarray`
        """
        return log_inverse(
            data = data,
            channel_indices = self.channel_indices,
            t = self.t,
            m = self.m
        )


class AsinhTransformer(Transformer):
    """\
    Class to apply the AsinhTransform.

    Parameters
    ----------
    channel_indices
        The channels to be transformed. Specifying None will
        transform all channels. Defaults to None. Note that
        in cytonormpy, channel indices will be set automatically
        by the CytoNorm object.
    cofactors
        Specifies the divisor of the channel values before applying
        the np.asinh function. Can be a singular float that will be
        applied to all channels or a list of floats, specifiying the
        cofactor for each channel. Defaults to 5.

    Returns
    -------
    None

    """

    def __init__(self,
                 channel_indices: Optional[Union[list[int], np.ndarray]] = None,  # noqa
                 cofactors: Union[list[float], float, np.ndarray] = 5  # noqa
                 ) -> None:
        super().__init__(channel_indices)
        self.cofactors = cofactors
        if self.cofactors is None:
            self.cofactors = 5

    def transform(self,
                  data: np.ndarray) -> np.ndarray:
        """\
        Applies asinh transform to channels specified in
        `.channel_indices`.

        Parameters
        ----------
        data
            The data to transform.

        Returns
        -------
        :class:`~numpy.ndarray`

        """
        return np.arcsinh(
            np.divide(data, self.cofactors)
        )

    def inverse_transform(self,
                          data: np.ndarray) -> np.ndarray:
        """\
        Applies inverse asinh transform to channels specified in
        `.channel_indices`.
        Parameters
        ----------
        data
            The data to transform.

        Returns
        -------
        :class:`~numpy.ndarray`
        """
        return np.multiply(
            np.sinh(data),
            self.cofactors
        )
