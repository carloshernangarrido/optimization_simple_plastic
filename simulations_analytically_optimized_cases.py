import itertools
import pickle
import unittest
import numpy as np
from matplotlib import pyplot as plt

from macro_model import protected_structure, update_model_mass, update_model_muN
from models.chain_like import Model, Mesh, Constraint, InitialCondition, Load, Element


def main():
    baseline = False
    verbose = False
    load_factor = 0.5
    m_add = 10.

    brick_area = 0.23*0.24
    simulation = {
        "delta_t": 0.0001,
        "t_fin": .01
    }

    structure = {
        "m_s": 57.6 * 2 / np.pi,
        "c_s": 0.0,
        "k_s": (57.6 * 2 / np.pi) * (2 * np.pi * 120) ** 2,
        "length_s": 0.10
    }

    load = {
        "t_0": 5e-3,  # s
        "p_0": 10e6*.294*brick_area * load_factor  # N
    }

    length_p = 0.05
    n_dof_p = 3
    muN_p_max = 70e3
    muN_p_min = 0.1*muN_p_max
    protection = {
        "length_p": length_p,  # m
        "linear_mass_density_p": 150*brick_area,  # kg/m
        "n_dof_p": n_dof_p,
        "muN_p": {'value': muN_p_min, 'v_th': 1.0},
        "gap_p": {'value': .55 * (length_p / n_dof_p), 'contact_stiffness': ((300e3-150e3)/10e-3) * (0.145/length_p)}
    }

    original_model = protected_structure(baseline, structure, protection, simulation, load, verbose)

    if baseline:
        m123 = [(0, 0, 0)]
        f123 = [(0, 0, 0)]
    else:
        # m_steps = 5
        # m_options = [i*m_add/(m_steps-1) for i in range(m_steps)]
        # m123 = [(_[0], _[1], m_add-(sum(_))) for _ in list(itertools.product(m_options, m_options))
        #         if min([_[0], _[1], m_add-(sum(_))]) >= 0.]
        m123 = [(0., 0., m_add)]

        # f_add = muN_p_max-muN_p_min
        # f_steps = 5
        # f_options = [i*f_add/f_steps for i in range(f_steps+1)]
        # f123 = list(itertools.product(f_options, repeat=3))
        if m_add == 30:
            if load_factor == 2.0:
                f123 = [(f, f, f) for f in [136259]]
            elif load_factor == 1.0:
                f123 = [(f, f, f) for f in [54258]]
            elif load_factor == 0.5:
                f123 = [(f, f, f) for f in [17335]]
        elif m_add == 20:
            if load_factor == 2.0:
                f123 = [(f, f, f) for f in [153351]]
            elif load_factor == 1.0:
                f123 = [(f, f, f) for f in [64099]]
            elif load_factor == 0.5:
                f123 = [(f, f, f) for f in [23137]]
        elif m_add == 10:
            if load_factor == 2.0:
                f123 = [(f, f, f) for f in [182936]]
            elif load_factor == 1.0:
                f123 = [(f, f, f) for f in [80640]]
            elif load_factor == 0.5:
                f123 = [(f, f, f) for f in [32695]]

    X = []
    y = []
    X_models = []

    print(m123)
    print(f123)

    if baseline:
        filename_output = 'dataset_baseline'
    else:
        filename_output = f'dataset_optimized_madd{int(m_add)}'
    if load_factor == 0.5:
        filename_output += '_half'
    elif load_factor == 1.0:
        pass
    elif load_factor == 2.0:
        filename_output += '_double'
    else:
        raise ValueError
    filename_output += '.pkl'
    print(filename_output)

    for m123_ in m123:
        m1, m2, m3 = m123_
        for f123_ in f123:
            f1, f2, f3 = f123_
            model = original_model.deepcopy()
            model = update_model_mass(model, 1+1, m1)
            model = update_model_mass(model, 1+2, m2)
            model = update_model_mass(model, 1+3, m3)
            model = update_model_muN(model, 1, f1)
            model = update_model_muN(model, 2, f2)
            model = update_model_muN(model, 3, f3)

            # Step 6: Solve the System
            model.solve()
            performance = np.max(np.abs(model.displacements(1)))
            x = [m1, m2, m3, f1, f2, f3]
            print('\n', x, performance)
            X.append(x)
            y.append(performance)
            X_models.append(model)

    with open(filename_output, 'wb') as file:
        pickle.dump((X_models, X, y), file=file)


if __name__ == '__main__':
    main()
