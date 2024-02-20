from functools import cache

import networkx as nx
import numpy as np
import pandas as pd
from numpy import sqrt, zeros
from scipy.sparse import csr_matrix
class LinDistModelQ:
    n = 1
    der = False
    battery = False
    def __init__(
        self,
        branch_data: pd.DataFrame,
        bus_data: pd.DataFrame,
        gen_data: pd.DataFrame,
        cap_data: pd.DataFrame,
    ):
        # ~~~~~~~~~~~~~~~~~~~~ Load Model ~~~~~~~~~~~~~~~~~~~~
        self.bus = bus_data.sort_values(by="id", ignore_index=True)
        self.branch = branch_data.sort_values(by="tb", ignore_index=True)
        self.gen = gen_data.sort_values(by="id", ignore_index=True)
        self.cap = cap_data.sort_values(by="id", ignore_index=True)
        self.bus.index = self.bus.id - 1
        self.gen.index = self.gen.id - 1
        self.cap.index = self.cap.id - 1
        if self.gen.shape[0] == 0:
            self.gen = self.gen.reindex(index=[-1])
        if self.cap.shape[0] == 0:
            self.cap = self.cap.reindex(index=[-1])
        # Make v_up_sq a three-phase array of voltage magnitude squared using whatever v_up is passed in.
        v0 = (
            self.bus.loc[self.bus.bus_type == "SWING", ["v_a", "v_b", "v_c"]]
            .to_numpy()
            .flatten()
        )
        self.v_up_sq = v0**2

        self.cvr_p = self.bus.cvr_p
        self.cvr_q = self.bus.cvr_q
        # ~~~~~~~~~~~~~~~~~~~~ prepare data ~~~~~~~~~~~~~~~~~~~~
        self.nb = len(self.bus.id)
        self.r, self.x = self._init_rx(self.branch)
        self.der_bus = {
            "a": self.gen.phases.str.contains("a").index.to_numpy(),
            "b": self.gen.phases.str.contains("b").index.to_numpy(),
            "c": self.gen.phases.str.contains("c").index.to_numpy(),
        }
        self.battery_bus = {
            "a": self.gen.phases.str.contains("a").index.to_numpy(),
            "b": self.gen.phases.str.contains("b").index.to_numpy(),
            "c": self.gen.phases.str.contains("c").index.to_numpy(),
        }
        # ~~ initialize index pointers ~~
        self.nl_a , self.nl_b, self.nl_c, self.line_a, self.line_b, self.line_c, self.basic_length = self.basic_var_length(self.branch)
        if LinDistModelQ.der and LinDistModelQ.battery:
            self.period = int(self.basic_length) + len(self.der_bus["a"]) +len(self.der_bus["b"])+len(self.der_bus["c"])+\
                3*len(self.battery_bus["a"])+3*len(self.battery_bus["b"])+3*len(self.battery_bus["c"])
        elif LinDistModelQ.der:
            self.period = int(self.basic_length) + len(self.der_bus["a"]) + len(self.der_bus["b"]) + len(self.der_bus["c"])
        elif LinDistModelQ.battery:
            self.period = int(self.basic_length) + 3 * len(self.battery_bus["a"]) + 3 * len(self.battery_bus["b"]) + 3 * len(self.battery_bus["c"])
        else:
            self.period = int(self.basic_length)
        self.x_maps, self.ctr_var_start_idx = self._variable_tables(self.branch)
        if LinDistModelQ.der and LinDistModelQ.battery:
            (
                self.p_der_start_phase_idx,
                self.q_der_start_phase_idx,
                self.pd_bat_start_phase_idx,
                self.pc_bat_start_phase_idx,
                self.b_bat_start_phase_idx,
            ) = self._control_variables(self.der_bus, self.ctr_var_start_idx, self.battery_bus)
            self.n_x = int(self.b_bat_start_phase_idx[LinDistModelQ.n - 1]["c"] + len(self.battery_bus["c"]))
            self.row_no = self.n_x
            self.n_x_first = int(self.b_bat_start_phase_idx[0]["c"] + len(self.battery_bus["c"]))
            self.a_eq, self.b_eq, self.a_ineq, self.b_ineq = self.create_model()
            self.bounds = self.init_bounds(self.bus, self.gen)
        elif LinDistModelQ.der:
            (self.p_der_start_phase_idx, self.q_der_start_phase_idx) = self._control_variables(self.der_bus, self.ctr_var_start_idx, self.battery_bus)
            self.n_x = int(self.q_der_start_phase_idx[LinDistModelQ.n - 1]["c"] + len(self.der_bus["c"]))
            self.row_no = int(self.x_maps[LinDistModelQ.n-1]["c"].vj.max() +1 )
            self.n_x_first = int(self.q_der_start_phase_idx[0]["c"] + len(self.der_bus["c"]))
            self.a_eq, self.b_eq, self.a_ineq, self.b_ineq = self.create_model()
            self.bounds = self.init_bounds(self.bus, self.gen)
        elif LinDistModelQ.battery:
            (self.pd_bat_start_phase_idx, self.pc_bat_start_phase_idx, self.b_bat_start_phase_idx) = self._control_variables(self.der_bus,self.ctr_var_start_idx, self.battery_bus)
            self.n_x = int(self.b_bat_start_phase_idx[LinDistModelQ.n - 1]["c"] + len(self.battery_bus["c"]))
            self.row_no = self.n_x
            self.n_x_first = int(self.b_bat_start_phase_idx[0]["c"] + len(self.battery_bus["c"]))
            self.a_eq, self.b_eq, self.a_ineq, self.b_ineq = self.create_model()
            self.bounds = self.init_bounds(self.bus, self.gen)
        else:
            self.n_x = self.x_maps[LinDistModelQ.n-1]["c"].vj.max() +1
            self.row_no = self.n_x
            self.n_x_first = int(self.x_maps[0]["c"].vj.max() +1)
            self.a_eq, self.b_eq, self.a_ineq, self.b_ineq = self.create_model()
            self.bounds = self.init_bounds(self.bus, self.gen)

    @staticmethod
    def _init_rx(branch):
        row = np.array(np.r_[branch.fb, branch.tb], dtype=int) - 1
        col = np.array(np.r_[branch.tb, branch.fb], dtype=int) - 1
        r = {
            "aa": csr_matrix((np.r_[branch.raa, branch.raa], (row, col))),
            "ab": csr_matrix((np.r_[branch.rab, branch.rab], (row, col))),
            "ac": csr_matrix((np.r_[branch.rac, branch.rac], (row, col))),
            "bb": csr_matrix((np.r_[branch.rbb, branch.rbb], (row, col))),
            "bc": csr_matrix((np.r_[branch.rbc, branch.rbc], (row, col))),
            "cc": csr_matrix((np.r_[branch.rcc, branch.rcc], (row, col))),
        }
        x = {
            "aa": csr_matrix((np.r_[branch.xaa, branch.xaa], (row, col))),
            "ab": csr_matrix((np.r_[branch.xab, branch.xab], (row, col))),
            "ac": csr_matrix((np.r_[branch.xac, branch.xac], (row, col))),
            "bb": csr_matrix((np.r_[branch.xbb, branch.xbb], (row, col))),
            "bc": csr_matrix((np.r_[branch.xbc, branch.xbc], (row, col))),
            "cc": csr_matrix((np.r_[branch.xcc, branch.xcc], (row, col))),
        }
        return r, x
    def basic_var_length(self,branch):
        a_indices = (branch.raa != 0) | (branch.xaa != 0)
        b_indices = (branch.rbb != 0) | (branch.xbb != 0)
        c_indices = (branch.rcc != 0) | (branch.xcc != 0)
        line_a = branch.loc[a_indices, ["fb", "tb"]].values
        line_b = branch.loc[b_indices, ["fb", "tb"]].values
        line_c = branch.loc[c_indices, ["fb", "tb"]].values
        nl_a = len(line_a)
        nl_b = len(line_b)
        nl_c = len(line_c)
        basic_length = 3*nl_a + 3*nl_b + 3*nl_c +3
        return nl_a, nl_b, nl_c, line_a, line_b, line_c, basic_length

    def _variable_tables(self,branch):
        nl_a = self.nl_a
        nl_b = self.nl_b
        nl_c = self.nl_c
        line_a = self.line_a
        line_b = self.line_b
        line_c = self.line_c

        g = nx.Graph()
        g_a = nx.Graph()
        g_b = nx.Graph()
        g_c = nx.Graph()
        g.add_edges_from(branch[["fb", "tb"]].values.astype(int) - 1)
        g_a.add_edges_from(line_a.astype(int) - 1)
        g_b.add_edges_from(line_b.astype(int) - 1)
        g_c.add_edges_from(line_c.astype(int) - 1)

        t_a = np.array(list(nx.dfs_edges(g_a, source=0)))
        t_b = np.array(list(nx.dfs_edges(g_b, source=0)))
        t_c = np.array(list(nx.dfs_edges(g_c, source=0)))

        p_a_end = 1 * nl_a
        q_a_end = 2 * nl_a
        v_a_end = 3 * nl_a + 1
        p_b_end = v_a_end + 1 * nl_b
        q_b_end = v_a_end + 2 * nl_b
        v_b_end = v_a_end + 3 * nl_b + 1
        p_c_end = v_b_end + 1 * nl_c
        q_c_end = v_b_end + 2 * nl_c
        v_c_end = v_b_end + 3 * nl_c + 1
        x_maps = {}

        for t in range(LinDistModelQ.n):
            df_a_t = pd.DataFrame(
                {
                    "bi": t_a[:, 0],
                    "bj": t_a[:, 1],
                    "pij": np.array([i + t * self.period for i in range(p_a_end)]),
                    "qij": np.array([i + t * self.period for i in range(p_a_end, q_a_end)]),
                    "vi": np.zeros_like(t_a[:, 0]),
                    "vj": np.array([i + t * self.period for i in range(q_a_end + 1, v_a_end)]),
                },
                dtype=np.int32,
            )
            df_b_t = pd.DataFrame(
                {
                    "bi": t_b[:, 0],
                    "bj": t_b[:, 1],
                    "pij": np.array([i+t * self.period for i in range(v_a_end, p_b_end)]),
                    "qij": np.array([i+t * self.period for i in range(p_b_end, q_b_end)]),
                    "vi": np.zeros_like(t_b[:, 0]),
                    "vj": np.array([i+t * self.period for i in range(q_b_end + 1, v_b_end)]),
                },
                dtype=np.int32,
            )
            df_c_t = pd.DataFrame(
                {
                    "bi": t_c[:, 0],
                    "bj": t_c[:, 1],
                    "pij": [i+t * self.period for i in range(v_b_end, p_c_end)],
                    "qij": [i+t * self.period for i in range(p_c_end, q_c_end)],
                    "vi": np.zeros_like(t_c[:, 0]),
                    "vj": [i+t * self.period for i in range(q_c_end + 1, v_c_end)],
                },
                dtype=np.int32,
            )
            df_a_t.loc[0, "vi"] = df_a_t.at[0, "vj"] - 1
            for i in df_a_t.bi.values[1:]:
                df_a_t.loc[df_a_t.loc[:, "bi"] == i, "vi"] = df_a_t.loc[
                    df_a_t.bj == i, "vj"
                ].values[0]
            df_b_t.loc[0, "vi"] = df_b_t.at[0, "vj"] - 1
            for i in df_b_t.bi.values[1:]:
                df_b_t.loc[df_b_t.loc[:, "bi"] == i, "vi"] = df_b_t.loc[
                    df_b_t.bj == i, "vj"
                ].values[0]
            df_c_t.loc[0, "vi"] = df_c_t.at[0, "vj"] - 1
            for i in df_c_t.bi.values[1:]:
                df_c_t.loc[df_c_t.loc[:, "bi"] == i, "vi"] = df_c_t.loc[
                    df_c_t.bj == i, "vj"
                ].values[0]
            x_maps_t = {"a": df_a_t, "b": df_b_t, "c": df_c_t}
            x_maps[t] = x_maps_t
            ctr_var_start_idx = v_c_end
        return x_maps, ctr_var_start_idx

    def _control_variables(self,der_bus, ctr_var_start_idx, battery_bus):
        ctr_var_start_idx = int(ctr_var_start_idx)
        if LinDistModelQ.der:
            ng_a = len(der_bus["a"])
            ng_b = len(der_bus["b"])
            ng_c = len(der_bus["c"])
            p_der_start_phase_idx = {}
            q_der_start_phase_idx = {}
        if LinDistModelQ.battery:
            nb_a = len(battery_bus["a"])
            nb_b = len(battery_bus["b"])
            nb_c = len(battery_bus["c"])
            pd_bat_start_phase_idx = {}
            pc_bat_start_phase_idx = {}
            b_bat_start_phase_idx = {}
        if LinDistModelQ.der and LinDistModelQ.battery:
            for t in range(LinDistModelQ.n):
                p_der_start_phase_idx_t = {
                    "a": ctr_var_start_idx+t*self.period,
                    "b": ctr_var_start_idx+t*self.period,
                    "c": ctr_var_start_idx+t*self.period
                }
                p_der_start_phase_idx[t] = p_der_start_phase_idx_t
                q_der_start_phase_idx_t = {
                    "a": ctr_var_start_idx+t*self.period,
                    "b": ctr_var_start_idx + ng_a+t*self.period,
                    "c": ctr_var_start_idx + ng_a + ng_b+t*self.period,
                }
                q_der_start_phase_idx[t] = q_der_start_phase_idx_t
                pd_bat_start_phase_idx_t = {
                    "a": q_der_start_phase_idx_t["c"] + nb_c,
                    "b": q_der_start_phase_idx_t["c"] + nb_c + nb_a,
                    "c": q_der_start_phase_idx_t["c"] + nb_c + nb_a + nb_b,
                }
                pd_bat_start_phase_idx[t] = pd_bat_start_phase_idx_t
                pc_bat_start_phase_idx_t = {
                    "a": pd_bat_start_phase_idx_t["c"] + nb_c,
                    "b": pd_bat_start_phase_idx_t["c"] + nb_c + nb_a,
                    "c": pd_bat_start_phase_idx_t["c"] + nb_c + nb_a + nb_b
                }
                pc_bat_start_phase_idx[t] = pc_bat_start_phase_idx_t
                b_bat_start_phase_idx_t = {
                    "a": pc_bat_start_phase_idx_t["c"] + nb_c,
                    "b": pc_bat_start_phase_idx_t["c"] + nb_c + nb_a,
                    "c": pc_bat_start_phase_idx_t["c"] + nb_c + nb_a + nb_b
                }
                b_bat_start_phase_idx[t] = b_bat_start_phase_idx_t
            return p_der_start_phase_idx, q_der_start_phase_idx, pd_bat_start_phase_idx, pc_bat_start_phase_idx,b_bat_start_phase_idx
        elif LinDistModelQ.der:
            for t in range(LinDistModelQ.n):
                p_der_start_phase_idx_t = {
                    "a": ctr_var_start_idx+t*self.period,
                    "b": ctr_var_start_idx+t*self.period,
                    "c": ctr_var_start_idx+t*self.period
                }
                p_der_start_phase_idx[t] = p_der_start_phase_idx_t
                q_der_start_phase_idx_t = {
                    "a": ctr_var_start_idx+t*self.period,
                    "b": ctr_var_start_idx + ng_a+t*self.period,
                    "c": ctr_var_start_idx + ng_a + ng_b+t*self.period,
                }
                q_der_start_phase_idx[t] = q_der_start_phase_idx_t
            return p_der_start_phase_idx, q_der_start_phase_idx
        elif LinDistModelQ.battery:
            for t in range(LinDistModelQ.n):
                pd_bat_start_phase_idx_t = {
                    "a": ctr_var_start_idx+t*self.period,
                    "b": ctr_var_start_idx + nb_a+t*self.period,
                    "c": ctr_var_start_idx + nb_a + nb_b+t*self.period,
                }
                pd_bat_start_phase_idx[t] = pd_bat_start_phase_idx_t
                pc_bat_start_phase_idx_t = {
                    "a": pd_bat_start_phase_idx_t["c"] + nb_c,
                    "b": pd_bat_start_phase_idx_t["c"] + nb_c + nb_a,
                    "c": pd_bat_start_phase_idx_t["c"] + nb_c + nb_a + nb_b
                }
                pc_bat_start_phase_idx[t] = pc_bat_start_phase_idx_t
                b_bat_start_phase_idx_t = {
                    "a": pc_bat_start_phase_idx_t["c"] + nb_c,
                    "b": pc_bat_start_phase_idx_t["c"] + nb_c + nb_a,
                    "c": pc_bat_start_phase_idx_t["c"] + nb_c + nb_a + nb_b
                }
                b_bat_start_phase_idx[t] = b_bat_start_phase_idx_t
            return pd_bat_start_phase_idx, pc_bat_start_phase_idx, b_bat_start_phase_idx
        else:
            return None


    def init_bounds(self, bus, gen, pij_max=100e3):
        x_maps = self.x_maps
        #bounds = []
        x_lim_lower = np.ones(self.n_x)
        x_lim_upper = np.ones(self.n_x)
        # ~~~~~~~~~~ x limits ~~~~~~~~~~
        for t in range(LinDistModelQ.n):
            for ph in "abc":
                if self.phase_exists(ph,t):
                    x_lim_lower[x_maps[t][ph].loc[:, "pij"]] = -pij_max  # P
                    x_lim_upper[x_maps[t][ph].loc[:, "pij"]] = pij_max  # P
                    x_lim_lower[x_maps[t][ph].loc[:, "qij"]] = -pij_max  # Q
                    x_lim_upper[x_maps[t][ph].loc[:, "qij"]] = pij_max  # Q
                    # ~~ v limits ~~:
                    i_swing = bus.loc[bus.bus_type == "SWING", "id"].to_numpy()[0] - 1
                    i_v_swing = (
                        x_maps[t][ph]
                        .loc[x_maps[t][ph].loc[:, "bi"] == i_swing, "vi"]
                        .to_numpy()[0]
                    )
                    x_lim_lower[i_v_swing] = bus.loc[i_swing, "v_min"] ** 2
                    x_lim_upper[i_v_swing] = bus.loc[i_swing, "v_max"] ** 2
                    x_lim_lower[x_maps[t][ph].loc[:, "vj"]] = (
                        bus.loc[x_maps[t][ph].loc[:, "bj"], "v_min"] ** 2
                    )
                    x_lim_upper[x_maps[t][ph].loc[:, "vj"]] = (
                        bus.loc[x_maps[t][ph].loc[:, "bj"], "v_max"] ** 2
                    )
                    # ~~ DER limits  ~~:
                    if LinDistModelQ.der:
                        for i in range(self.der_bus[ph].shape[0]):
                            i_q = self.q_der_start_phase_idx[t][ph] + i
                            # reactive power bounds
                            s_rated = gen["s" + ph + "_max"]
                            p_out = gen["p" + ph]
                            q_min = -sqrt((s_rated**2) - (p_out**2))
                            q_max = sqrt((s_rated**2) - (p_out**2))
                            x_lim_lower[i_q] = q_min[self.der_bus[ph][i]]
                            x_lim_upper[i_q] = q_max[self.der_bus[ph][i]]
                # ~~ Battery limits ~~:
                    if LinDistModelQ.battery:
                        for i in range(self.der_bus[ph].shape[0]):
                            pb_max = 100e3
                            b_min = 0.3
                            b_max = 0.95
                            i_d = self.pd_bat_start_phase_idx[t][ph] + i
                            i_c = self.pc_bat_start_phase_idx[t][ph] + i
                            i_b = self.b_bat_start_phase_idx[t][ph] + i
                            #battery active power charge/discharge and s.o.c bounds
                            x_lim_lower[i_d] = 0
                            x_lim_lower[i_c] = 0
                            x_lim_lower[i_b] = b_min
                            x_lim_upper[i_d] = pb_max
                            x_lim_upper[i_c] = pb_max
                            x_lim_upper[i_b] = b_max
        bounds = [(l, u) for (l, u) in zip(x_lim_lower, x_lim_upper)]
        return bounds

    @cache
    def branch_into_j(self, var, j, phase, t):
        return self.x_maps[t][phase].loc[self.x_maps[t][phase].bj == j, var].to_numpy()

    @cache
    def branches_out_of_j(self, var, j, phase, t):
        return self.x_maps[t][phase].loc[self.x_maps[t][phase].bi == j, var].to_numpy()

    @cache
    def idx(self, var, index, phase, t):
        if var == "pg":
            raise ValueError("pg is fixed and is not a valid variable.")
        if var == "qg":
            if index in set(self.der_bus[phase]):
                return (
                    self.q_der_start_phase_idx[t][phase]
                    + np.where(self.der_bus[phase] == index)[0]
                )
            return []
        if var == "pd":
            if index in set(self.der_bus[phase]):
                return (
                    self.pd_bat_start_phase_idx[t][phase]
                    + np.where(self.der_bus[phase] == index)[0]
                )
            return []
        if var == "pc":
            if index in set(self.der_bus[phase]):
                return (
                    self.pc_bat_start_phase_idx[t][phase]
                    + np.where(self.der_bus[phase] == index)[0]
                )
            return []
        if var == "b":
            if index in set(self.der_bus[phase]):
                return (
                    self.b_bat_start_phase_idx[t][phase]
                    + np.where(self.der_bus[phase] == index)[0]
                )
            return []
        return self.x_maps[t][phase].loc[self.x_maps[t][phase].bj == index, var].to_numpy()

    @cache
    def _row(self, var, index, phase, t):
        return self.idx(var, index, phase,t)

    @cache
    def phase_exists(self, phase, t, index: int = None):
        if index is None:
            return self.x_maps[t][phase].shape[0] > 0
        return len(self.idx("bj", index, phase,t)) > 0

    def create_model(self):
        v_up = {"a": self.v_up_sq[0], "b": self.v_up_sq[1], "c": self.v_up_sq[2]}
        x_maps = self.x_maps
        r, x = self.r, self.x
        bus = self.bus
        g = self.gen
        cap = self.cap

        # ########## Aeq and Beq Formation ###########
        n_c = 0.95
        n_d = 0.95
        h_max = 100e3
        n_rows = self.row_no
        n_col = self.n_x
        a_eq = zeros((n_rows, n_col))  # Aeq has the same number of rows as equations with a column for each x
        b_eq = zeros(n_rows)
        a_ineq = zeros((n_rows, n_col))
        b_ineq = zeros(n_rows)

        for t in range(LinDistModelQ.n):
            for j in range(1, self.nb):

                def col(var, phase):
                    return self.idx(var, j, phase, t)

                def coll(var, phase):
                    return self.idx(var, j, phase, t - 1)

                def row(var, phase):
                    return self._row(var, j, phase, t)

                def roww(var, phase):
                    return self._row(var, j, phase, t - 1)

                def children(var, phase):
                    return self.branches_out_of_j(var, j, phase, t)

                for ph in ["abc", "bca", "cab"]:
                    a, b, c = ph[0], ph[1], ph[2]
                    aa = "".join(sorted(a + a))
                    ab = "".join(
                        sorted(a + b)
                    )  # if ph=='cab', then a+b=='ca'. Sort so ab=='ac'
                    ac = "".join(sorted(a + c))
                    if not self.phase_exists(a, t, j):
                        continue
                    # P equation
                    a_eq[row("pij", a), col("pij", a)] = 1
                    a_eq[row("pij", a), col("vj", a)] = (
                            -(bus.cvr_p[j] / 2) * bus["pl_" + a][j]
                    )
                    a_eq[row("pij", a), children("pij", a)] = -1
                    b_eq[row("pij", a)] = (1 - (bus.cvr_p[j] / 2)) * bus["pl_" + a][j]
                    # Q equation
                    a_eq[row("qij", a), col("qij", a)] = 1
                    a_eq[row("qij", a), col("vj", a)] = (
                            -(bus.cvr_q[j] / 2) * bus["ql_" + a][j]
                    )
                    a_eq[row("qij", a), children("qij", a)] = -1
                    b_eq[row("qij", a)] = (1 - (bus.cvr_q[j] / 2)) * bus["ql_" + a][
                        j
                    ] - cap["q" + a].get(j, 0)
                    # V equation
                    i = self.idx("bi", j, a, t)[0]
                    if i == 0:  # Swing bus
                        a_eq[row("vi", a), col("vi", a)] = 1
                        b_eq[row("vi", a)] = v_up[a]
                    a_eq[row("vj", a), col("vj", a)] = 1
                    a_eq[row("vj", a), col("vi", a)] = -1
                    a_eq[row("vj", a), col("pij", a)] = 2 * r[aa][i, j]
                    a_eq[row("vj", a), col("qij", a)] = 2 * x[aa][i, j]
                    a_eq[row("vj", a), col("pij", b)] = -r[ab][i, j] + sqrt(3) * x[ab][i, j]
                    a_eq[row("vj", a), col("qij", b)] = -x[ab][i, j] - sqrt(3) * r[ab][i, j]
                    a_eq[row("vj", a), col("pij", c)] = -r[ac][i, j] - sqrt(3) * x[ac][i, j]
                    a_eq[row("vj", a), col("qij", c)] = -x[ac][i, j] + sqrt(3) * r[ac][i, j]
                    if LinDistModelQ.der:
                        a_eq[row("qij", a), col("qg", a)] = 1
                        b_eq[row("pij", a)] = (1 - (bus.cvr_p[j] / 2)) * bus["pl_" + a][j] - g[
                            "p" + a
                            ].get(j, 0)
                        b_eq[row("qij", a)] = (1 - (bus.cvr_q[j] / 2)) * bus["ql_" + a][
                            j
                        ] - cap["q" + a].get(j, 0)
                    if LinDistModelQ.battery:
                        a_eq[row("pij", a), col("pd", a)] = 1
                        a_eq[row("pij", a), col("pc", a)] = -1
                        a_eq[row("b", a), col("b", a)] = 1
                        a_eq[row("b", a), col("pc", a)] = -n_c
                        a_eq[row("b", a), col("pd", a)] = 1 / n_d
                        if t == 0:
                            b_eq[row("b", a)] = 0.5
                        else:
                            b_eq[row("b", a)] = 0
                            a_eq[row("b", a), coll("b", a)] = -1
                        a_ineq[row("pd", a), col("pd", a)] = 1
                        a_ineq[row("pd", a), col("pc", a)] = -1
                        b_ineq[row("pd", a)] = h_max


        return a_eq, b_eq, a_ineq, b_ineq


    # def fast_model_update(self, bus_data, gen_data, cap_data):
    #     # ~~~~~~~~~~~~~~~~~~~~ Load Model ~~~~~~~~~~~~~~~~~~~~
    #     self.bus = bus_data.sort_values(by="id", ignore_index=True)
    #     self.gen = gen_data.sort_values(by="id", ignore_index=True)
    #     self.cap = cap_data.sort_values(by="id", ignore_index=True)
    #     self.bus.index = self.bus.id - 1
    #     self.gen.index = self.gen.id - 1
    #     self.cap.index = self.cap.id - 1
    #     if self.gen.shape[0] == 0:
    #         self.gen = self.gen.reindex(index=[-1])
    #     if self.cap.shape[0] == 0:
    #         self.cap = self.cap.reindex(index=[-1])
    #     # Make v_up_sq a three-phase array of voltage magnitude squared using whatever v_up is passed in.
    #     v0 = self.bus.loc[self.bus.bus_type == "SWING", "v_nom"]
    #     self.v_up_sq = np.array([np.abs(v0) ** 2, np.abs(v0) ** 2, np.abs(v0) ** 2])
    #
    #     self.cvr_p = self.bus.cvr_p
    #     self.cvr_q = self.bus.cvr_q
    #     self.nb = len(self.bus.id)
    #     # ~~~~~~~~~~~~~~~~~~~~ initialize Aeq and beq and objective gradient ~~~~~~~~~~~~~~~~~~~~
    #     self.a_eq, self.b_eq = self._fast_update_aeq_beq()
    #     self.bounds = self.init_bounds(self.bus, self.gen)
    #
    # def _fast_update_aeq_beq(self):
    #     v_up = {"a": self.v_up_sq[0], "b": self.v_up_sq[1], "c": self.v_up_sq[2]}
    #     x_maps = self.x_maps
    #     bus = self.bus
    #     g = self.gen
    #     cap = self.cap
    #
    #     # ########## Aeq and Beq Formation ###########
    #     n_rows = x_maps[LinDistModelQ.n-1]["c"].vj.max() + 1  # last index + 1
    #     n_col = self.n_x
    #     a_eq = zeros(
    #         (n_rows, n_col)
    #     )  # Aeq has the same number of rows as equations with a column for each x
    #     b_eq = zeros(n_rows)
    #     for j in range(1, self.nb):
    #
    #         def col(var, phase):
    #             return self.idx(var, j, phase)
    #
    #         def row(var, phase):
    #             return self._row(var, j, phase)
    #
    #         for ph in ["abc", "bca", "cab"]:
    #             a, b, c = ph[0], ph[1], ph[2]
    #             if not self.phase_exists(a, j):
    #                 continue
    #             # P equation
    #             a_eq[row("pij", a), col("vj", a)] = (
    #                 -(bus.cvr_p[j] / 2) * bus["pl_" + a][j]
    #             )
    #             b_eq[row("pij", a)] = (1 - (bus.cvr_p[j] / 2)) * bus["pl_" + a][j] - g[
    #                 "p" + a
    #             ].get(j, 0)
    #             # Q equation
    #             a_eq[row("qij", a), col("vj", a)] = (
    #                 -(bus.cvr_q[j] / 2) * bus["ql_" + a][j]
    #             )
    #             b_eq[row("qij", a)] = (1 - (bus.cvr_q[j] / 2)) * bus["ql_" + a][
    #                 j
    #             ] - cap["q" + a].get(j, 0)
    #
    #             # V equation
    #             i = self.idx("bi", j, a)[0]
    #             if i == 0:  # Swing bus
    #                 b_eq[row("vi", a)] = v_up[a]
    #
    #     return a_eq, b_eq

    def get_q_dec_variables(self, x_sol):
        ng_a = len(self.der_bus["a"])
        ng_b = len(self.der_bus["b"])
        ng_c = len(self.der_bus["c"])
        if LinDistModelQ.der and LinDistModelQ.battery:
            qi = self.q_der_start_phase_idx
            pdi = self.pd_bat_start_phase_idx
            pci = self.pc_bat_start_phase_idx
            bi = self.b_bat_start_phase_idx
            dec_var={}
            dec_d_var={}
            dec_c_var={}
            dec_b_var={}
            for t in range(LinDistModelQ.n):
                dec_var_t = pd.DataFrame(0, columns=["a", "b", "c"], index=np.array(range(1, self.nb + 1)),dtype=np.float64, )
                dec_d_var_t = pd.DataFrame(0, columns=["a", "b", "c"], index=np.array(range(1, self.nb + 1)), dtype=np.float64, )
                dec_c_var_t = pd.DataFrame(0, columns=["a", "b", "c"], index=np.array(range(1, self.nb + 1)), dtype=np.float64, )
                dec_b_var_t = pd.DataFrame(0, columns=["a", "b", "c"], index=np.array(range(1, self.nb + 1)), dtype=np.float64, )
                for j in range(ng_a):
                    for ph in "abc":
                        dec_var_t.loc[self.der_bus[ph][j]+1, ph] = x_sol[qi[t][ph] + j]
                        dec_d_var_t.loc[self.der_bus[ph][j]+1, ph] = x_sol[pdi[t][ph] + j]
                        dec_c_var_t.loc[self.der_bus[ph][j]+1, ph] = x_sol[pci[t][ph] + j]
                        dec_b_var_t.loc[self.der_bus[ph][j]+1, ph] = x_sol[bi[t][ph] + j]
                dec_var[t] = dec_var_t
                dec_d_var[t] = dec_d_var_t
                dec_c_var[t] = dec_c_var_t
                dec_b_var[t] = dec_b_var_t
            return dec_var, dec_d_var, dec_c_var, dec_b_var
        elif LinDistModelQ.der:
            qi = self.q_der_start_phase_idx
            dec_var = {}
            for t in range(LinDistModelQ.n):
                dec_var_t = pd.DataFrame(0, columns=["a", "b", "c"], index=np.array(range(1, self.nb + 1)),
                                         dtype=np.float64, )
                for j in range(ng_a):
                    for ph in "abc":
                        dec_var_t.loc[self.der_bus[ph][j] + 1, ph] = x_sol[qi[t][ph] + j]
                dec_var[t] = dec_var_t
            return dec_var
        elif LinDistModelQ.battery:
            pdi = self.pd_bat_start_phase_idx
            pci = self.pc_bat_start_phase_idx
            bi = self.b_bat_start_phase_idx
            dec_d_var = {}
            dec_c_var = {}
            dec_b_var = {}
            for t in range(LinDistModelQ.n):
                dec_d_var_t = pd.DataFrame(0, columns=["a", "b", "c"], index=np.array(range(1, self.nb + 1)),
                                           dtype=np.float64, )
                dec_c_var_t = pd.DataFrame(0, columns=["a", "b", "c"], index=np.array(range(1, self.nb + 1)),
                                           dtype=np.float64, )
                dec_b_var_t = pd.DataFrame(0, columns=["a", "b", "c"], index=np.array(range(1, self.nb + 1)),
                                           dtype=np.float64, )
                for j in range(ng_a):
                    for ph in "abc":
                        dec_d_var_t.loc[self.der_bus[ph][j] + 1, ph] = x_sol[pdi[t][ph] + j]
                        dec_c_var_t.loc[self.der_bus[ph][j] + 1, ph] = x_sol[pci[t][ph] + j]
                        dec_b_var_t.loc[self.der_bus[ph][j] + 1, ph] = x_sol[bi[t][ph] + j]
                dec_d_var[t] = dec_d_var_t
                dec_c_var[t] = dec_c_var_t
                dec_b_var[t] = dec_b_var_t
            return dec_d_var, dec_c_var, dec_b_var
        else:
            return None


    def get_v_solved(self, x_sol):
        v_df ={}
        for t in range(LinDistModelQ.n):
            v_df_t = pd.DataFrame(
                columns=["a", "b", "c"],
                index=np.array(range(1, self.nb + 1)),
                dtype=np.float64,
            )
            for ph in "abc":
                v_df_t.loc[1, ph] = x_sol[self.x_maps[t][ph].vi[0]].astype(np.float64)
                v_df_t.loc[self.x_maps[t][ph].bj.values + 1, ph] = np.sqrt(x_sol[
                    self.x_maps[t][ph].vj.values
                ].astype(np.float64))
            v_df[t] = v_df_t
        return v_df

    def get_s_solved(self, x_sol):
        s_df = {}
        for t in range(LinDistModelQ.n):
            s_df_t = pd.DataFrame(
                columns=["fb", "tb", "a", "b", "c"], index=range(2, self.nb + 1)
            )
            s_df_t["a"] = s_df_t["a"].astype(complex)
            s_df_t["b"] = s_df_t["b"].astype(complex)
            s_df_t["c"] = s_df_t["c"].astype(complex)
            for ph in "abc":
                s_df_t.loc[self.x_maps[t][ph].bj.values + 1, "fb"] = (
                    self.x_maps[t][ph].bi.values + 1
                )
                s_df_t.loc[self.x_maps[t][ph].bj.values + 1, "tb"] = (
                    self.x_maps[t][ph].bj.values + 1
                )
                s_df_t.loc[self.x_maps[t][ph].bj.values + 1, ph] = (
                    x_sol[self.x_maps[t][ph].pij] + 1j * x_sol[self.x_maps[t][ph].qij]
                )
            s_df[t] = s_df_t
        return s_df
