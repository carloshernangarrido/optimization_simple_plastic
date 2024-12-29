from copy import deepcopy, copy
from typing import Union, List
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt, animation

element_types: dict = {'m': {'description': 'mass',
                             'props': ['value']},
                       'k': {'description': 'linear stiffness',
                             'props': ['value', 'allowable_deformation']},
                       'c': {'description': 'linear viscous damping',
                             'props': ['value']},
                       'muN': {'description': 'Coulomb friction',
                               'props': ['value', 'v_th', 'c_th']},
                       'gap': {'description': 'closing gap contact',
                               'props': ['value', 'contact_stiffness']},
                       'penalty_gap': {'description': 'closing gap contact with penalty stiffness',
                                       'props': ['value', 'contact_stiffness', 'penetration', 'quadratic_coefficient']}}

constraint_types = ['fixed_dof']

initial_condition_types = ['displacement', 'velocity']


def s(x, x_0: float = 0, smoothness: float = 1):
    """
    Helper function to smooth transitions between two functions.

    Example of usage:
        h(x) = (1-s(x, x_0)) * f(x) + s(x, x_0) * g(x)
    """
    return 0.5 + 0.5 * np.tanh((x - x_0) / smoothness)


class Element:
    def __init__(self, element_type: str, i: int, j: int, props: Union[None, dict, float, int]):
        assert element_type in element_types.keys(), 'Unsupported element_type'
        if props is None:
            props = {'value': 0.0}
        elif isinstance(props, (float, int)):
            props = {'value': float(props)}
        elif isinstance(props, dict):
            pass
        else:
            raise AssertionError('props must be None, dict or numeric')
        assert all([isinstance(props[given_prop], (float, int)) for given_prop in props.keys()]), 'props must be ' \
                                                                                                  'numeric '
        self.element_type = element_type
        self.i = i
        self.j = j
        self.props = props
        # Precalculated
        self.update_props()
        assert all([req_prop in props.keys() for req_prop in element_types[element_type]['props']]), 'Required prop'
        assert all([prop in element_types[element_type]['props'] for prop in props.keys()]), 'Unsupported prop'

    def update_props(self, prop: str = None):
        if prop is not None:
            self.props.pop(prop, None)
        if self.element_type == 'muN':
            if 'c_th' not in self.props.keys():
                self.props.update({'c_th': self.props['value'] / self.props['v_th']})
            if 'v_th' not in self.props.keys():
                self.props.update({'v_th': self.props['value'] / self.props['c_th']})
        if self.element_type == 'penalty_gap':
            if 'quadratic_coefficient' not in self.props.keys():
                self.props.update({'quadratic_coefficient':
                                       self.props['contact_stiffness'] / (4 * self.props['penetration'])})
        if self.element_type == 'k':
            if 'allowable_deformation' not in self.props.keys():
                self.props.update({'allowable_deformation': np.inf})

    def __str__(self, ji: bool = False):
        return f'{self.element_type}_{self.j}_{self.i}' if ji else f'{self.element_type}_{self.i}_{self.j}'

    def force_ij(self, u, u_dot):
        """
        Returns the force exerted by the element depending on the displacement and velocities.

        :param u: vector of displacements
        :param u_dot: vector of velocities.
        :return: float force exerted by the element.
        """
        if self.element_type == 'm':
            raise ValueError('inertial pseudo-force is not supported')
        try:
            if self.element_type == 'k':
                deformation = u[self.i] - u[self.j]
                if abs(deformation) > self.props['allowable_deformation']:
                    raise Exception('deformation larger than allowable deformation')
                else:
                    return self.props['value'] * deformation
            elif self.element_type == 'c':
                return self.props['value'] * (u_dot[self.i] - u_dot[self.j])
            elif self.element_type == 'muN':
                return self.props['value'] * np.tanh((u_dot[self.i] - u_dot[self.j]) / self.props['v_th'])
                # if self.props['value'] == 0:
                #     return 0.0
                # vel = u_dot[self.i] - u_dot[self.j]
                # if abs(vel) <= self.props['v_th']:
                #     return self.props['c_th'] * vel
                # else:
                #     return self.props['value'] * np.sign(vel)
            elif self.element_type == 'gap':
                if self.props['value'] >= 0:
                    contact_deformation = (u[self.i] - u[self.j]) - self.props['value']
                    if contact_deformation > 0:
                        return self.props['contact_stiffness'] * contact_deformation
                    else:
                        return 0.0
                else:
                    contact_deformation = (u[self.i] - u[self.j]) - self.props['value']
                    if contact_deformation < 0:
                        return self.props['contact_stiffness'] * contact_deformation
                    else:
                        return 0.0
            elif self.element_type == 'penalty_gap':
                if self.props['value'] >= 0:
                    contact_deformation = (u[self.i] - u[self.j]) - self.props['value']
                    if contact_deformation < -self.props['penetration']:
                        return 0.0
                    elif contact_deformation > self.props['penetration']:
                        return contact_deformation * self.props['contact_stiffness']
                    else:
                        return self.props['quadratic_coefficient'] * (
                                contact_deformation + self.props['penetration']) ** 2
                else:
                    raise NotImplementedError
        except KeyError:
            raise KeyError('Insufficient responses to calculate element force.')

    def aliases(self):
        return [self.__str__(), self.__str__(ji=True)]

    def is_same_as(self, compare_element):
        return True if compare_element.__str__() in self.aliases() else False


class Mesh:
    def __init__(self, length: Union[float, int], n_dof: int, total_mass: Union[float, int] = None,
                 lumped_masses: bool = False):
        """
        Creates a Mesh object that represents a 1D deformable body.

        :param total_mass: Total mass of the body
        :param length: Total length of the body
        :param n_dof: Number of degrees of freedom.
        :param lumped_masses: If True, the element mass is lumped in the DOF with larger index
        """
        assert isinstance(length, (int, float)) and length > 0
        assert isinstance(n_dof, int) and n_dof > 0

        self.n_dof = n_dof
        self.coordinates = np.linspace(0, length, n_dof)
        self.lumped_masses = lumped_masses
        if total_mass is not None:
            assert isinstance(total_mass, (int, float))  # and total_mass > 0
            element_mass = total_mass / (n_dof - 1)
            self.elements = [Element('m', i=i + 1 if self.lumped_masses else i, j=i + 1, props=element_mass)
                             for i in range(self.n_dof - 1)]
        else:
            self.elements = []

    def fill_elements(self, element_type, props_s: Union[List[Union[dict, float, int]], dict, float, int]):
        """
        Fill with elements all the mesh, assigning the values provided in values.

        :param element_type: Type of elements
        :param props_s: If it is a list, it has to be the same length as the n_dof-1.
        If it is a dict of numbers, these same numbers are used as props for all the elements. If it is numeric, this
        number is used as 'value' in props of all the elements.
        :return: The updated mesh
        """

        if isinstance(props_s, list) and len(props_s) == self.n_dof - 1:
            for i, props in enumerate(props_s):
                self.elements.append(Element(element_type,
                                             i=i + 1 if (self.lumped_masses and element_type == 'm') else i,
                                             j=i + 1, props=props))
        elif isinstance(props_s, (float, int, dict)):
            for i in range(self.n_dof - 1):
                if isinstance(props_s, dict):
                    props_s = props_s.copy()
                self.elements.append((Element(element_type,
                                              i=i + 1 if (self.lumped_masses and element_type == 'm') else i,
                                              j=i + 1, props=props_s)))
        else:
            raise TypeError
        return self


class DofWise:
    def __init__(self, dof_s: Union[int, List[int]]):
        if isinstance(dof_s, int):
            dof_s = [dof_s]
        assert all([isinstance(dof, int) and dof >= 0 for dof in dof_s])
        self.dof_s = dof_s


class Constraint(DofWise):
    def __init__(self, dof_s, constraint_type: str = 'fixed_dof', value: Union[int, float] = 0.0):
        super().__init__(dof_s)
        assert constraint_type in constraint_types
        assert isinstance(value, (float, int))

        self.constraint_type = constraint_type
        self.value = float(value)


class Load(DofWise):
    def __init__(self, dof_s, t, force):
        super().__init__(dof_s)
        assert np.ndim(t) == 1, 't must be 1D array like'
        assert np.ndim(force) == 1, 'force must be 1D array like'
        assert len(t) == len(force)

        self.t = np.array(t)
        self.force = np.array(force)

    def force_at_t(self, t: float):
        return self.force[((self.t >= t).nonzero())[0][0]]


class InitialCondition(DofWise):
    """
    All the initial conditions (displacements and velocities) are assumed 0.0, except for those defined using this.
    """

    def __init__(self, dof_s, ic_type: str = 'displacement', value: Union[float, int] = 0.0):
        super().__init__(dof_s)
        assert ic_type in initial_condition_types, f"Supported initial conditions types are: " \
                                                   f"{initial_condition_types}, but {ic_type} was specified."
        self.ic_type = ic_type
        self.value = float(value)


class Model:
    def __init__(self, mesh: Mesh, constraints: Union[Constraint, List[Constraint]] = None,
                 loads: Union[Load, List[Load]] = None,
                 initial_conditions: Union[InitialCondition, List[InitialCondition]] = None,
                 options: dict = None):
        assert isinstance(mesh, Mesh)
        assert isinstance(constraints, (list, Constraint)) or constraints is None
        assert isinstance(loads, (list, Load)) or loads is None
        assert isinstance(initial_conditions, (list, InitialCondition)) or initial_conditions is None

        self.options = {'t_vector': None, 'method': None} if options is None else options
        assert isinstance(self.options, dict)

        self.mesh = mesh
        self.constraints = [] if constraints is None else constraints
        self.constraints = [self.constraints] if isinstance(constraints, Constraint) else constraints
        self.loads = [] if loads is None else loads
        self.loads = [self.loads] if isinstance(loads, Load) else self.loads
        self.initial_conditions = [] if initial_conditions is None else initial_conditions
        self.initial_conditions = [self.initial_conditions] if isinstance(initial_conditions, InitialCondition) \
            else self.initial_conditions

        assert all([isinstance(constraint, Constraint) for constraint in self.constraints])
        assert all([isinstance(load, Load) for load in self.loads])

        self.sol = None
        self.constraining = None
        self.y0 = None
        self.loading = None
        self.connectivity = None
        self.dof_masses = None
        self.n_elements = None
        self.n_dof = None
        self.update_model()

    def deepcopy(self):
        return deepcopy(self)

    def update_model(self):
        for element in self.mesh.elements:
            element.update_props()
        self.n_dof = self.mesh.n_dof
        self.n_elements = len(self.mesh.elements)
        mass_elements = \
            [self.mesh.elements[i] for i in range(self.n_elements) if self.mesh.elements[i].element_type == 'm']
        self.dof_masses = np.zeros(self.n_dof)
        for mass_element in mass_elements:
            self.dof_masses[mass_element.i] += mass_element.props['value'] / 2
            self.dof_masses[mass_element.j] += mass_element.props['value'] / 2
        # massless elements connected to each dof
        self.connectivity = [[i_e for i_e in range(self.n_elements) if
                              (self.mesh.elements[i_e].element_type != 'm') and
                              (self.mesh.elements[i_e].i == i_dof or self.mesh.elements[i_e].j == i_dof)]
                             for i_dof in range(self.n_dof)]
        # load applied to each dof
        self.loading = [[i_l for i_l in range(len(self.loads)) if (i_dof in self.loads[i_l].dof_s)]
                        for i_dof in range(self.n_dof)]
        self.y0 = np.zeros(2 * self.n_dof)
        for initial_condition in self.initial_conditions:
            if initial_condition.ic_type == 'displacement':
                self.y0[initial_condition.dof_s] += initial_condition.value
            elif initial_condition.ic_type == 'velocity':
                self.y0[[i_dof + self.n_dof for i_dof in initial_condition.dof_s]] += initial_condition.value
            else:
                raise ValueError(f"Supported initial conditions types are: {initial_condition_types}"
                                 f", but {initial_condition.ic_type} was specified.")
        # fixed dof_s
        self.constraining = []
        for constraint in self.constraints:
            if constraint.constraint_type == 'fixed_dof':
                self.constraining.extend(constraint.dof_s)
        for i_const in self.constraining:
            self.dof_masses[i_const] = 1
        # Solution
        self.sol = None

    def element_forces_at(self, i_dof: int, y: np.ndarray):
        force_sum = 0.0
        for i_e in self.connectivity[i_dof]:
            if self.mesh.elements[i_e].i == i_dof:
                force_sum += self.mesh.elements[i_e].force_ij(u=y[0:self.n_dof], u_dot=y[self.n_dof:])
            else:
                force_sum -= self.mesh.elements[i_e].force_ij(u=y[0:self.n_dof], u_dot=y[self.n_dof:])
        return force_sum

    def external_forces_at(self, i_dof: int, t: float):
        force_sum = 0
        for i_l in self.loading[i_dof]:
            force_sum += self.loads[i_l].force_at_t(t)
        return force_sum

    def f(self, t: float, y: np.ndarray):
        """
        Function characterizing the set of ordinary differential equations defined as:

        dy / dt = f(t, y)
        y(t0) = y0

        :param t: time
        :param y: 1D vector of system states. y = [ u, u_dot ]
        :returns y_dot: 1D vector of derivative system states. y = [ u_dot, -forces_sum/masses ]
        """
        forces_sum = np.zeros(self.n_dof)
        for i_dof in range(self.n_dof):
            # element forces
            forces_sum[i_dof] = self.element_forces_at(i_dof, y) + self.external_forces_at(i_dof, t)
            # external forces
            # for i_l in self.loading[i_dof]:
            #     forces_sum[i_dof] += self.loads[i_l].force_at_t(t)

        y_dot = np.hstack((y[self.n_dof:], -forces_sum / self.dof_masses))
        # constraining
        for i_const in self.constraining:
            y_dot[i_const] = 0
            y_dot[self.n_dof + i_const] = 0

        return y_dot

    def solve(self, t_vector: np.ndarray = None, method: str = None):
        if t_vector is None:
            t_vector = self.options['t_vector']
        if method is None:
            try:
                method = self.options['method']
            except KeyError:
                method = None
        self.sol = sp.integrate.solve_ivp(self.f, [t_vector[0], t_vector[-1]], self.y0, t_eval=t_vector,
                                          method=method)
        return self.sol

    def reactions(self, i_dof: int):
        return np.array([self.element_forces_at(i_dof, y_) + self.external_forces_at(i_dof, t_)
                         for (y_, t_) in zip(self.sol.y.T, self.sol.t)])

    def impulses(self, i_dof: int):
        return np.hstack((0, sp.integrate.cumtrapz(self.reactions(i_dof), x=self.sol.t)))

    def displacements(self, i_dof: int):
        return self.sol.y[i_dof]

    def deformations(self):
        return np.array([self.displacements(i_dof+1) - self.displacements(i_dof) for i_dof in range(self.n_dof-1)]).T

    def velocities(self, i_dof: int):
        return self.sol.y[self.n_dof + i_dof]

    def animate(self, each: int = 1):
        colors = plt.cm.viridis(np.linspace(0, 1, self.n_dof))  # Usar un mapa de colores

        fig, ax = plt.subplots(1, 1)
        deformed, = ax.plot(self.mesh.coordinates + self.sol.y[0:self.n_dof, 0], np.zeros(self.n_dof), marker='o',
                            markerfacecolor='r', markeredgecolor='k', linestyle='-')

        def animate_frame(i):
            deformed.set_xdata(self.mesh.coordinates + self.sol.y[0:self.n_dof, i * each])
            return deformed,

        ani = animation.FuncAnimation(fig, animate_frame, interval=1, blit=True, frames=len(self.sol.t) // each)
        return fig, ax, ani

    def animate_colors(self, each: int = 1):
        colors = plt.cm.viridis(np.linspace(0, 1, self.n_dof))  # Usar un mapa de colores
        print(colors)

        fig, ax = plt.subplots(1, 1)
        # Inicializar los puntos con un color predeterminado
        deformed, = ax.plot(self.mesh.coordinates, np.zeros(self.n_dof), marker='o',
                            markerfacecolor='r', markeredgecolor='k', linestyle='-')

        # Actualizar la función animate_frame para usar colores diferentes
        def animate_frame(i):
            xdata = self.mesh.coordinates + self.sol.y[0:self.n_dof, i * each]
            # Actualizar los datos de cada punto y su color
            deformed.set_xdata(xdata)
            # Aquí estamos actualizando el color de cada punto
            for j in range(self.n_dof):
                deformed.set_markerfacecolor(colors[j][:3])
            return deformed,

        ani = animation.FuncAnimation(fig, animate_frame, interval=1, blit=True, frames=len(self.sol.t) // each)
        return fig, ax, ani
