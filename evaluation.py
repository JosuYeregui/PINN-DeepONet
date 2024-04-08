from scr.SPM import Solid_Phase
from scr.pinn import FFNN
from scr.utils import load_params

if __name__ == "__main__":

    parameters = load_params()

    model = FFNN(2, 1)
    PINN_pos = Solid_Phase(model, parameters)
    PINN_pos.load_model("/models/previous.pt")