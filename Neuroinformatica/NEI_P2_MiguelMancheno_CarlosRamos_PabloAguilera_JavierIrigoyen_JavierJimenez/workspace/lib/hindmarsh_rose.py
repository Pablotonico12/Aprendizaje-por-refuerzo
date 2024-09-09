import numpy as np


class HindmarshRoseNeuron:

    class hr_state:
        def __init__(self, **kwargs):
            self.x_ = kwargs.x
            self.y_ = kwargs.y
            self.z_ = kwargs.z

    hr_params = {
        'e_i': 3,
        'a': 1,
        'b': 3,
        'c': 1,
        'd': 5,
        'mu': 0.0021,  # mu = r in the HR paper
        's': 4,
        'x1': -1.6,
        'state': (-1.5, 1, 1),
        'delta_t': 0.001,
        'umbral_subida': -1.2,
        'umbral_bajada': -1.2,
        'hp_speed': 1,
        'dp_speed': 1,
    }

    def __init__(self, *args, **kwargs):
        self.e_i_ = kwargs['e_i']
        self.a_ = kwargs['a']
        self.b_ = kwargs['b']
        self.c_ = kwargs['c']
        self.d_ = kwargs['d']
        self.mu_ = kwargs['mu']
        self.s_ = kwargs['s']
        self.x1_ = kwargs['x1']
        self.delta_t_ = kwargs['delta_t']
        self.umbral_subida_ = kwargs['umbral_subida']
        self.umbral_bajada_ = kwargs['umbral_bajada']
        self.hp_speed_ = kwargs['hp_speed']
        self.dp_speed_ = kwargs['dp_speed']
        self.mode_3d_ = kwargs['mode_3d']
        self.signo_dx_ = 1
        self.x_ = kwargs['state'][0]
        self.y_ = kwargs['state'][1]
        if self.mode_3d_:
            self.z_ = kwargs['state'][2]
        else:
            self.z_ = 0

    def d_x_(self, state_ant: tuple):
        x_ant, y_ant, z_ant = state_ant
        return self.delta_t_ * (y_ant - self.a_ * x_ant**3 + self.b_ * x_ant**2
                                - z_ant + self.e_i_)

    def d_y_(self, state_ant: tuple):
        x_ant, y_ant, _ = state_ant
        return self.delta_t_ * (self.c_ - self.d_ * x_ant**2 - y_ant)

    def d_z_(self, state_ant: tuple):
        x_ant, _, z_ant = state_ant
        return self.delta_t_ * self.mu_ * (-z_ant + self.s_ *
                                           (x_ant - self.x1_))

    def signal_step(self, state_ant=None, e_i=0):

        if state_ant is not None:
            if not isinstance(state_ant, tuple):
                print("El estado debe ser una tupla de floats \
                      de la forma (x, y, z)")
                return None
            else:
                self.x_ = state_ant[0]
                self.y_ = state_ant[1]
                if self.mode_3d_:
                    self.z_ = state_ant[2]
        else:
            state_ant = (self.x_, self.y_, self.z_)

        self.e_i_ = e_i

        dx_value = self.d_x_(state_ant)
        if self.mode_3d_:
            new_z = self.z_ + self.d_z_(state_ant)
        else:
            new_z = 0

        new_x = self.x_ + dx_value
        new_y = self.y_ + self.d_y_(state_ant)

        new_signo_dx = np.sign(dx_value)

        spike = False
        hiper_pol = False

        # Compruebo si hay spike
        if new_x > self.umbral_subida_ and self.signo_dx_ > 0 \
                and new_signo_dx < 0:
            spike = True

        # Compruebo si se trata de un inicio de hiper_polarización
        if new_x < self.umbral_bajada_:
            if self.signo_dx_ < 0 and new_signo_dx > 0:
                hiper_pol = True

        self.x_ = new_x
        self.y_ = new_y
        self.z_ = new_z
        self.signo_dx_ = new_signo_dx

        return (new_x, new_y, new_z), spike, hiper_pol


def i_fast_x(g_fast_yx, v_x, v_y, e_syn=-1.92, v_fast=-1.66, s_fast=0.44):
    return g_fast_yx * (v_x - e_syn) / (1.0 + np.exp(s_fast * (v_fast - v_y)))


def i_slow_x(g_slow_yx, m_slow_x, v_x, e_syn):
    return g_slow_yx * m_slow_x * (v_x - e_syn)


def dm_slow_x(k1_x, k2_x, m_slow_x, s_slow, v_slow, v_y):
    return ((k1_x * (1.0 - m_slow_x)) /
            (1.0 + np.exp(s_slow * (v_slow - v_y)))) - k2_x * m_slow_x


def i_elec_x(g_elec_yx, v_x, v_y):
    return g_elec_yx * (v_y - v_x)
