

def zheng_current(beta, i_typ):
    def current(t):
        if t < 1800:
            h = 1/1800 * t
        else:
            h = -1 / 1800 * t + 2
        return (-h * beta + 1) * i_typ

    return current

#  simulation.solve(t_eval, inputs={"Current function [A]": 1.6})
