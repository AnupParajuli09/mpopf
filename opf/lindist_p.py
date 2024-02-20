from functools import cache

import networkx as nx
import numpy as np
import pandas as pd
from numpy import sqrt, zeros
from scipy.sparse import csr_matrix


class LinDistModelP:
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
        # ~~ initialize index pointers ~~
        self.x_maps, self.ctr_var_start_idx = self._variable_tables(self.branch)
        (
            self.p_der_start_phase_idx,
            self.q_der_start_phase_idx,
        ) = self._control_variables(self.der_bus, self.ctr_var_start_idx)
        self.n_x = int(self.q_der_start_phase_idx["c"] + len(self.der_bus["c"]))
        self.der_end_idx = int(self.q_der_start_phase_idx["c"] + len(self.der_bus["c"]))

        # ~~~~~~~~~~~~~~~~~~~~ initialize Aeq and beq and objective gradient ~~~~~~~~~~~~~~~~~~~~
        self.a_eq, self.b_eq = self.create_model()
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

    @staticmethod
    def _variable_tables(branch):
        a_indices = (branch.raa != 0) | (branch.xaa != 0)
        b_indices = (branch.rbb != 0) | (branch.xbb != 0)
        c_indices = (branch.rcc != 0) | (branch.xcc != 0)
        line_a = branch.loc[a_indices, ["fb", "tb"]].values
        line_b = branch.loc[b_indices, ["fb", "tb"]].values
        line_c = branch.loc[c_indices, ["fb", "tb"]].values
        nl_a = len(line_a)
        nl_b = len(line_b)
        nl_c = len(line_c)
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

        df_a = pd.DataFrame(
            {
                "bi": t_a[:, 0],
                "bj": t_a[:, 1],
                "pij": np.array([i for i in range(p_a_end)]),
                "qij": np.array([i for i in range(p_a_end, q_a_end)]),
                "vi": np.zeros_like(t_a[:, 0]),
                "vj": np.array([i for i in range(q_a_end + 1, v_a_end)]),
            },
            dtype=np.int32,
        )
        df_b = pd.DataFrame(
            {
                "bi": t_b[:, 0],
                "bj": t_b[:, 1],
                "pij": np.array([i for i in range(v_a_end, p_b_end)]),
                "qij": np.array([i for i in range(p_b_end, q_b_end)]),
                "vi": np.zeros_like(t_b[:, 0]),
                "vj": np.array([i for i in range(q_b_end + 1, v_b_end)]),
            },
            dtype=np.int32,
        )
        df_c = pd.DataFrame(
            {
                "bi": t_c[:, 0],
                "bj": t_c[:, 1],
                "pij": [i for i in range(v_b_end, p_c_end)],
                "qij": [i for i in range(p_c_end, q_c_end)],
                "vi": np.zeros_like(t_c[:, 0]),
                "vj": [i for i in range(q_c_end + 1, v_c_end)],
            },
            dtype=np.int32,
        )

        x_maps = {"a": df_a, "b": df_b, "c": df_c}

        df_a.loc[0, "vi"] = df_a.at[0, "vj"] - 1
        for i in df_a.bi.values[1:]:
            df_a.loc[df_a.loc[:, "bi"] == i, "vi"] = df_a.loc[
                df_a.bj == i, "vj"
            ].values[0]
        df_b.loc[0, "vi"] = df_b.at[0, "vj"] - 1
        for i in df_b.bi.values[1:]:
            df_b.loc[df_b.loc[:, "bi"] == i, "vi"] = df_b.loc[
                df_b.bj == i, "vj"
            ].values[0]
        df_c.loc[0, "vi"] = df_c.at[0, "vj"] - 1
        for i in df_c.bi.values[1:]:
            df_c.loc[df_c.loc[:, "bi"] == i, "vi"] = df_c.loc[
                df_c.bj == i, "vj"
            ].values[0]
        ctr_var_start_idx = v_c_end  # start with the largest index so far

        return x_maps, ctr_var_start_idx

    @staticmethod
    def _control_variables(der_bus, ctr_var_start_idx):
        ctr_var_start_idx = int(ctr_var_start_idx)
        ng_a = len(der_bus["a"])
        ng_b = len(der_bus["b"])
        ng_c = len(der_bus["c"])
        ng = ng_a + ng_b + ng_c
        p_der_start_phase_idx = {
            "a": ctr_var_start_idx,
            "b": ctr_var_start_idx + ng_a,
            "c": ctr_var_start_idx + ng_a + ng_b,
        }
        q_der_start_phase_idx = {
            "a": ctr_var_start_idx + ng,
            "b": ctr_var_start_idx + ng,
            "c": ctr_var_start_idx + ng,
        }
        return p_der_start_phase_idx, q_der_start_phase_idx

    def init_bounds(self, bus, gen, pij_max=100e3):
        x_maps = self.x_maps
        # ~~~~~~~~~~ x limits ~~~~~~~~~~
        x_lim_lower = np.ones(self.n_x) * -100e3
        x_lim_upper = np.ones(self.n_x) * 100e3
        for ph in "abc":
            if self.phase_exists(ph):
                x_lim_lower[x_maps[ph].loc[:, "pij"]] = -pij_max  # P
                x_lim_upper[x_maps[ph].loc[:, "pij"]] = pij_max  # P
                x_lim_lower[x_maps[ph].loc[:, "qij"]] = -pij_max  # Q
                x_lim_upper[x_maps[ph].loc[:, "qij"]] = pij_max  # Q
                # ~~ v limits ~~:
                i_swing = bus.loc[bus.bus_type == "SWING", "id"].to_numpy()[0] - 1
                i_v_swing = (
                    x_maps[ph]
                    .loc[x_maps[ph].loc[:, "bi"] == i_swing, "vi"]
                    .to_numpy()[0]
                )
                x_lim_lower[i_v_swing] = bus.loc[i_swing, "v_min"] ** 2
                x_lim_upper[i_v_swing] = bus.loc[i_swing, "v_max"] ** 2
                x_lim_lower[x_maps[ph].loc[:, "vj"]] = (
                    bus.loc[x_maps[ph].loc[:, "bj"], "v_min"] ** 2
                )
                x_lim_upper[x_maps[ph].loc[:, "vj"]] = (
                    bus.loc[x_maps[ph].loc[:, "bj"], "v_max"] ** 2
                )
                # ~~ DER limits  ~~:
                for i in range(self.der_bus[ph].shape[0]):
                    i_p = self.p_der_start_phase_idx[ph] + i
                    # reactive power bounds
                    p_out = gen["p" + ph]
                    p_max = p_out
                    p_min = p_out * 0
                    # active power bounds
                    x_lim_lower[i_p] = p_min[self.der_bus[ph][i]]
                    x_lim_upper[i_p] = p_max[self.der_bus[ph][i]]
        bounds = [(l, u) for (l, u) in zip(x_lim_lower, x_lim_upper)]
        return bounds

    @cache
    def branch_into_j(self, var, j, phase):
        return self.x_maps[phase].loc[self.x_maps[phase].bj == j, var].to_numpy()

    @cache
    def branches_out_of_j(self, var, j, phase):
        return self.x_maps[phase].loc[self.x_maps[phase].bi == j, var].to_numpy()

    @cache
    def idx(self, var, index, phase):
        if var == "pg":
            if index in set(self.der_bus[phase]):
                return (
                    self.p_der_start_phase_idx[phase]
                    + np.where(self.der_bus[phase] == index)[0]
                )
            return []
        if var == "qg":
            raise ValueError("qg is fixed and is not a valid variable.")
        return self.x_maps[phase].loc[self.x_maps[phase].bj == index, var].to_numpy()

    @cache
    def _row(self, var, index, phase):
        if var == "vj":
            return self.idx(var, index, phase) - 1
        if var == "vi" and self.idx("bi", index, phase)[0] == 0:
            return self.x_maps[phase]["vj"].to_numpy()[-1]
        return self.idx(var, index, phase)

    @cache
    def phase_exists(self, phase, index: int = None):
        if index is None:
            return self.x_maps[phase].shape[0] > 0
        return len(self.idx("bj", index, phase)) > 0

    def create_model(self):
        v_up = {"a": self.v_up_sq[0], "b": self.v_up_sq[1], "c": self.v_up_sq[2]}
        x_maps = self.x_maps
        r, x = self.r, self.x
        bus = self.bus
        g = self.gen
        cap = self.cap

        # ########## Aeq and Beq Formation ###########
        n_rows = x_maps["c"].vj.max() + 1  # last index + 1
        n_col = self.n_x
        a_eq = zeros(
            (n_rows, n_col)
        )  # Aeq has the same number of rows as equations with a column for each x
        b_eq = zeros(n_rows)
        for j in range(1, self.nb):

            def col(var, phase):
                return self.idx(var, j, phase)

            def row(var, phase):
                return self._row(var, j, phase)

            def children(var, phase):
                return self.branches_out_of_j(var, j, phase)

            for ph in ["abc", "bca", "cab"]:
                a, b, c = ph[0], ph[1], ph[2]
                aa = "".join(sorted(a + a))
                ab = "".join(
                    sorted(a + b)
                )  # if ph=='cab', then a+b=='ca'. Sort so ab=='ac'
                ac = "".join(sorted(a + c))
                if not self.phase_exists(a, j):
                    continue
                # P equation
                a_eq[row("pij", a), col("pij", a)] = 1
                a_eq[row("pij", a), col("vj", a)] = (
                    -(bus.cvr_p[j] / 2) * bus["pl_" + a][j]
                )
                a_eq[row("pij", a), children("pij", a)] = -1
                a_eq[row("pij", a), col("pg", a)] = 1
                b_eq[row("pij", a)] = (1 - (bus.cvr_p[j] / 2)) * bus["pl_" + a][j]
                # Q equation
                a_eq[row("qij", a), col("qij", a)] = 1
                a_eq[row("qij", a), col("vj", a)] = (
                    -(bus.cvr_q[j] / 2) * bus["ql_" + a][j]
                )
                a_eq[row("qij", a), children("qij", a)] = -1
                b_eq[row("qij", a)] = (
                    (1 - (bus.cvr_q[j] / 2)) * bus["ql_" + a][j]
                    - cap["q" + a].get(j, 0)
                    - g["q" + a].get(j, 0)
                )

                # V equation
                i = self.idx("bi", j, a)[0]
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

        return a_eq, b_eq

    def fast_model_update(self, bus_data, gen_data, cap_data):
        # ~~~~~~~~~~~~~~~~~~~~ Load Model ~~~~~~~~~~~~~~~~~~~~
        self.bus = bus_data.sort_values(by="id", ignore_index=True)
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
        v0 = self.bus.loc[self.bus.bus_type == "SWING", "v_nom"]
        self.v_up_sq = np.array([np.abs(v0) ** 2, np.abs(v0) ** 2, np.abs(v0) ** 2])

        self.cvr_p = self.bus.cvr_p
        self.cvr_q = self.bus.cvr_q
        self.nb = len(self.bus.id)
        # ~~~~~~~~~~~~~~~~~~~~ initialize Aeq and beq and objective gradient ~~~~~~~~~~~~~~~~~~~~
        self.a_eq, self.b_eq = self._fast_update_aeq_beq()
        self.bounds = self.init_bounds(self.bus, self.gen)

    def _fast_update_aeq_beq(self):
        v_up = {"a": self.v_up_sq[0], "b": self.v_up_sq[1], "c": self.v_up_sq[2]}
        x_maps = self.x_maps
        bus = self.bus
        g = self.gen
        cap = self.cap

        # ########## Aeq and Beq Formation ###########
        n_rows = x_maps["c"].vj.max() + 1  # last index + 1
        n_col = self.n_x
        a_eq = zeros(
            (n_rows, n_col)
        )  # Aeq has the same number of rows as equations with a column for each x
        b_eq = zeros(n_rows)
        for j in range(1, self.nb):

            def col(var, phase):
                return self.idx(var, j, phase)

            def row(var, phase):
                return self._row(var, j, phase)

            for ph in ["abc", "bca", "cab"]:
                a, b, c = ph[0], ph[1], ph[2]
                if not self.phase_exists(a, j):
                    continue
                # P equation
                a_eq[row("pij", a), col("vj", a)] = (
                    -(bus.cvr_p[j] / 2) * bus["pl_" + a][j]
                )
                b_eq[row("pij", a)] = (1 - (bus.cvr_p[j] / 2)) * bus["pl_" + a][j]
                # Q equation
                a_eq[row("qij", a), col("vj", a)] = (
                    -(bus.cvr_q[j] / 2) * bus["ql_" + a][j]
                )
                b_eq[row("qij", a)] = (
                    (1 - (bus.cvr_q[j] / 2)) * bus["ql_" + a][j]
                    - g["q" + a].get(j, 0)
                    - cap["q" + a].get(j, 0)
                )

                # V equation
                i = self.idx("bi", j, a)[0]
                if i == 0:  # Swing bus
                    b_eq[row("vi", a)] = v_up[a]

        return a_eq, b_eq

    def get_p_dec_variables(self, x_sol):
        ng_a = len(self.der_bus["a"])
        ng_b = len(self.der_bus["b"])
        ng_c = len(self.der_bus["c"])
        pi = self.p_der_start_phase_idx
        dec_var = np.zeros((self.nb, 3))
        for j in range(ng_a):
            dec_var[self.der_bus["a"][j], 0] = x_sol[pi["a"] + j]
        for j in range(ng_b):
            dec_var[self.der_bus["b"][j], 1] = x_sol[pi["b"] + j]
        for j in range(ng_c):
            dec_var[self.der_bus["c"][j], 2] = x_sol[pi["c"] + j]
        return dec_var

    def get_v_solved(self, x_sol):
        v_df = pd.DataFrame(
            columns=["a", "b", "c"],
            index=np.array(range(1, self.nb + 1)),
            dtype=np.float64,
        )
        for ph in "abc":
            v_df.loc[1, ph] = x_sol[self.x_maps[ph].vi[0]].astype(np.float64)
            v_df.loc[self.x_maps[ph].bj.values + 1, ph] = x_sol[
                self.x_maps[ph].vj.values
            ].astype(np.float64)
        return np.sqrt(v_df)

    def get_s_solved(self, x_sol):
        s_df = pd.DataFrame(
            columns=["fb", "tb", "a", "b", "c"], index=range(2, self.nb + 1)
        )
        s_df["a"] = s_df["a"].astype(complex)
        s_df["b"] = s_df["b"].astype(complex)
        s_df["c"] = s_df["c"].astype(complex)
        for ph in "abc":
            s_df.loc[self.x_maps[ph].bj.values + 1, "fb"] = (
                self.x_maps[ph].bi.values + 1
            )
            s_df.loc[self.x_maps[ph].bj.values + 1, "tb"] = (
                self.x_maps[ph].bj.values + 1
            )
            s_df.loc[self.x_maps[ph].bj.values + 1, ph] = (
                x_sol[self.x_maps[ph].pij] + 1j * x_sol[self.x_maps[ph].qij]
            )
        return s_df
