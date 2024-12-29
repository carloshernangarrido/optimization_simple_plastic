import matplotlib.pyplot as plt
import numpy as np
from models.chain_like import Model, Mesh, Constraint, InitialCondition, Load, Element


def protected_structure(baseline: bool, structure: dict, protection: dict, simulation: dict, load: dict,
                        verbose: bool = False) -> Model:
    # Simulation
    t_vector = np.arange(0, simulation["t_fin"], simulation["delta_t"]).reshape((-1,))

    # load
    f_vector = np.zeros_like(t_vector)
    f_vector[0] = 0.
    for i, t in enumerate(t_vector):
        f = load["p_0"] - t*(load["p_0"]/load["t_0"])
        if f <= 0:
            break
        f_vector[i] = f
    if verbose:
        plt.figure()
        plt.plot(t_vector, f_vector)
        plt.xlabel('time (s)')
        plt.ylabel('load force (N)')
        plt.show()

    # Step 1: Create a Mesh
    n_dof = 1+1+protection["n_dof_p"]
    mesh = Mesh(length=1.0, n_dof=n_dof, total_mass=None, lumped_masses=True)
    mesh.coordinates = [0.0, structure["length_s"]] + \
                       [structure["length_s"] + (i+1)*(protection["length_p"]/protection["n_dof_p"])
                        for i in range(protection["n_dof_p"])]

    # Step 2: Define Elements
    if verbose:
        print('f =', ((structure["k_s"]/structure["m_s"])**(1/2))/6.28, 'Hz')
        print('T =', 1/(((structure["k_s"]/structure["m_s"])**(1/2))/6.28), 's')
        print('m_s =', structure["m_s"], 'kg')
        print('k_s =', structure["k_s"], 'N/m')

    # Adding mass elements
    m_p = protection['linear_mass_density_p']*protection["length_p"]
    mesh.fill_elements('m', [structure["m_s"]] + [m_p/protection["n_dof_p"] for _ in range(protection["n_dof_p"])])

    # Adding stiffness elements
    k_p = 100*structure['k_s'] if baseline else 0
    mesh.fill_elements('k', [structure['k_s']] + [k_p for _ in range(protection["n_dof_p"])])

    # Adding damping elements
    c_p = 100*structure['c_s'] if baseline else 0
    mesh.fill_elements('c', [structure['c_s']] + [c_p for _ in range(protection["n_dof_p"])])

    # Adding plastic elements
    mesh.fill_elements('muN', [{'value': 0.0, 'v_th': 1}] +
                       [protection["muN_p"].copy() for _ in range(protection["n_dof_p"])])

    # Adding gap elements
    mesh.fill_elements('gap', [{'value': 1., 'contact_stiffness': 0.0}] +
                       [protection["gap_p"] for _ in range(protection["n_dof_p"])])

    # Step 3: Set Initial Conditions
    initial_displacement = np.zeros((n_dof,))
    initial_velocity = np.zeros((n_dof,))

    initial_conditions = []
    for i, (disp, vel) in enumerate(zip(initial_displacement, initial_velocity)):
        initial_conditions.append(InitialCondition(dof_s=i, ic_type='displacement', value=disp))
        initial_conditions.append(InitialCondition(dof_s=i, ic_type='velocity', value=vel))

    # Step 4: Specify Constraints and Loads
    constraints = Constraint(0)
    loads = [Load(dof_s=n_dof-1, t=t_vector, force=f_vector)]

    # Step 5: Create a Model
    model = Model(
        mesh=mesh,
        constraints=constraints,
        loads=loads,
        initial_conditions=initial_conditions,
        options={'t_vector': t_vector, 'method': 'RK45'}
    )
    return model


def update_model_mass(model: Model, i_dof, delta_m) -> Model:
    model.dof_masses[i_dof] += delta_m
    return model


def update_model_muN(model: Model, i_dof, delta_f) -> Model:
    for i in range(len(model.mesh.elements)):
        if model.mesh.elements[i].element_type == 'muN':
            if model.mesh.elements[i].i == i_dof and model.mesh.elements[i].j == i_dof+1:
                model.mesh.elements[i].props['value'] = model.mesh.elements[i].props['value'] + delta_f
    return model


def objective_fun(x, *args) -> float:
    # # Step 6: Solve the System
    # solution = model.solve()
    #
    # # Step 7: Analyze Results
    # # Displacements and velocities
    # displacements = model.displacements(i_dof=0)
    pass
