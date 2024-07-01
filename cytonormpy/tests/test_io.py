import pytest
import os
from os import PathLike

import cytonormpy as cnp
from cytonormpy import AsinhTransformer, read_model

def test_save_and_read_model(tmpdir: PathLike):

    cytonorm = cnp.CytoNorm()
    t = AsinhTransformer
    cytonorm.add_transformer(t)

    cytonorm.save_model(os.path.join(tmpdir, "my_model.cytonorm"))

    cy_reread = read_model(os.path.join(tmpdir, "my_model.cytonorm"))

    assert isinstance(cy_reread, cnp.CytoNorm)
    assert cy_reread._transformer is not None

    assert not hasattr(cy_reread, "_datahandler")


