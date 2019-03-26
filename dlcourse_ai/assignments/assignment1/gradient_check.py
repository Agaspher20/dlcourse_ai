import numpy as np


def check_gradient(f, x, delta=1e-5, tol = 1e-4):
    '''
    Checks the implementation of analytical gradient by comparing
    it to numerical gradient using two-point formula

    Arguments:
      f: function that receives x and computes value and gradient
      x: np array, initial point where gradient is checked
      delta: step to compute numerical gradient
      tol: tolerance for comparing numerical and analytical gradient

    Return:
      bool indicating whether gradients match or not
    '''
    
    assert isinstance(x, np.ndarray)
    assert x.dtype == np.float
    
    x_copy = x.copy()
    _, analytic_grad = f(x)
    assert np.all(np.isclose(x_copy, x, tol)), "Functions shouldn't modify input variables"
    assert analytic_grad.shape == x.shape

    # We will go through every dimension of x and compute numeric
    # derivative for it
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index
        analytic_grad_at_ix = analytic_grad[ix]

        x_val = x[ix]
        x_copy[ix] = x_val + delta
        f_plus_delta, _ = f(x_copy)
        x_copy[ix] = x_val - delta
        f_minus_delta, _ = f(x_copy)
        x_copy[ix] = x_val

        numeric_grad_at_ix = (f_plus_delta - f_minus_delta)/(2*delta)

        if not np.isclose(numeric_grad_at_ix, analytic_grad_at_ix, tol):
            print("Gradients are different at %s. Analytic: %2.5f, Numeric: %2.5f" % (ix, analytic_grad_at_ix, numeric_grad_at_ix))
            return False

        it.iternext()

    print("Gradient check passed!")
    return True
