class Globalization(object):
    """Base class for globalization of semismooth Newton method."""

    def globalize(self, norm_residual_map, norm_residual_map_old):
        """Compute step size used to globalize the semismooth Newton method.

        Parameters:
        -----------
            norm_residual_map : Function
                norm_residual mapping : [0, 1] -> IR
            norm_residual_map_old : float
                residual mapping evaluated at old iterate
            gtol : float
                termination tolerance

        Returns:
        -------
            sigma : float
                step size
            norm_residual_map_new: float
                residual mapping evaluate at new iterate
        """

        raise NotImplementedError("globalize not implemented.")
