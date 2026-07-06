import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        shifted_res = z - np.max(z)
        expo = np.exp(shifted_res)
        res = expo / np.sum(expo)
        return np.round(res, 4)
