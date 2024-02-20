import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from opf import opf_solver
from opf.lindist_p import LinDistModelP
from opf.lindist_q import LinDistModelQ
from opf.test.legacy.opf_var import QModel
from opf.test.legacy.opf_watt import PModel

branchdata_path = Path("./opf/test/branchdata.csv")
powerdata_path = Path("./opf/test/powerdata.csv")
legacy_powerdata_path = Path("./opf/test/legacy/powerdata.csv")
bus_data_path = Path("./opf/test/bus_data.csv")
gen_data_path = Path("./opf/test/gen_data.csv")
cap_data_path = Path("./opf/test/cap_data.csv")


def assert_q_results_equal(model_new, model_old, res_new, res_old):
    v_old = model_old.get_v_solved(res_old.x)
    v_new = model_new.get_v_solved(res_new.x)
    p_old = np.real(model_old.get_s_solved(res_old.x).loc[:, ["a", "b", "c"]])
    p_new = np.real(model_new.get_s_solved(res_new.x).loc[:, ["a", "b", "c"]])
    q_old = np.imag(model_old.get_s_solved(res_old.x).loc[:, ["a", "b", "c"]])
    q_new = np.imag(model_new.get_s_solved(res_new.x).loc[:, ["a", "b", "c"]])
    q_gen_old = model_old.get_dec_variables(res_old.x)
    q_gen_new = model_new.get_q_dec_variables(res_new.x)
    assert res_new.fun == res_old.fun
    assert np.array_equal(v_old, v_new, equal_nan=True)
    assert np.array_equal(p_old, p_new, equal_nan=True)
    assert np.array_equal(q_old, q_new, equal_nan=True)
    assert np.array_equal(q_gen_old, q_gen_new, equal_nan=True)


def assert_p_results_equal(model_new, model_old, res_new, res_old):
    v_old = model_old.get_v_solved(res_old.x)
    v_new = model_new.get_v_solved(res_new.x)
    p_old = np.real(model_old.get_s_solved(res_old.x).loc[:, ["a", "b", "c"]])
    p_new = np.real(model_new.get_s_solved(res_new.x).loc[:, ["a", "b", "c"]])
    q_old = np.imag(model_old.get_s_solved(res_old.x).loc[:, ["a", "b", "c"]])
    q_new = np.imag(model_new.get_s_solved(res_new.x).loc[:, ["a", "b", "c"]])
    q_gen_old = model_old.get_dec_variables(res_old.x)
    q_gen_new = model_new.get_p_dec_variables(res_new.x)
    assert res_new.fun == res_old.fun
    assert np.array_equal(v_old, v_new, equal_nan=True)
    assert np.array_equal(p_old, p_new, equal_nan=True)
    assert np.array_equal(q_old, q_new, equal_nan=True)
    assert np.array_equal(q_gen_old, q_gen_new, equal_nan=True)


class TestObjectives(unittest.TestCase):
    def test_loss(self):
        branch_data = pd.read_csv(branchdata_path, header=0)
        bus_data = pd.read_csv(bus_data_path)
        gen_data = pd.read_csv(gen_data_path)
        cap_data = pd.read_csv(cap_data_path)
        gen_data.loc[:, ["pa", "pb", "pc"]] *= 5
        gen_data.loc[:, ["sa_max", "sb_max", "sc_max"]] *= 5

        model_new = LinDistModelQ(branch_data, bus_data, gen_data, cap_data)
        powerdata = pd.read_csv(legacy_powerdata_path, header=0)
        model_old = QModel(
            branch_data,
            powerdata,
            p_rating_mult=5,
            v_up=1.05,
            p_base_gld=1e6,
            v_ll_base_gld=4160,
        )
        print("Solve old")
        res_old = model_old.solve(objective="loss", gld_correction=False)
        print("Solve new")
        res_new = opf_solver.cvxpy_solve(model_new, opf_solver.cp_obj_loss)
        assert_q_results_equal(model_new, model_old, res_new, res_old)

    def test_cp_obj_target_q_3ph(self):
        branch_data = pd.read_csv(branchdata_path, header=0)
        bus_data = pd.read_csv(bus_data_path)
        gen_data = pd.read_csv(gen_data_path)
        cap_data = pd.read_csv(cap_data_path)
        gen_data.loc[:, ["pa", "pb", "pc"]] *= 5
        gen_data.loc[:, ["sa_max", "sb_max", "sc_max"]] *= 5

        model_new = LinDistModelQ(branch_data, bus_data, gen_data, cap_data)
        powerdata = pd.read_csv(legacy_powerdata_path, header=0)
        model_old = QModel(
            branch_data,
            powerdata,
            p_rating_mult=5,
            v_up=1.05,
            p_base_gld=1e6,
            v_ll_base_gld=4160,
        )
        print("Solve old")
        model_old.loss_percent = np.array([0.1, 0.1, 0.1])
        res_old = model_old.solve(
            objective="q_target", target=np.array([0.3, 0.3, 0.3])
        )
        print("Solve new")
        res_new = opf_solver.cvxpy_solve(
            model_new,
            opf_solver.cp_obj_target_q_3ph,
            target=np.array([0.3, 0.3, 0.3]),
            loss_percent=np.array([0.1, 0.1, 0.1]),
        )
        assert_q_results_equal(model_new, model_old, res_new, res_old)

    def test_cp_obj_target_q_total(self):
        branch_data = pd.read_csv(branchdata_path)
        bus_data = pd.read_csv(bus_data_path)
        gen_data = pd.read_csv(gen_data_path)
        cap_data = pd.read_csv(cap_data_path)
        gen_data.loc[:, ["pa", "pb", "pc"]] *= 5
        gen_data.loc[:, ["sa_max", "sb_max", "sc_max"]] *= 5

        model_new = LinDistModelQ(branch_data, bus_data, gen_data, cap_data)
        powerdata = pd.read_csv(legacy_powerdata_path)
        model_old = QModel(
            branch_data,
            powerdata,
            p_rating_mult=5,
            v_up=1.05,
            p_base_gld=1e6,
            v_ll_base_gld=4160,
        )
        print("Solve old")
        model_old.loss_percent = np.array([0.1, 0.1, 0.1])
        res_old = model_old.solve(objective="q_target", target=0.9)
        print("Solve new")
        res_new = opf_solver.cvxpy_solve(
            model_new,
            opf_solver.cp_obj_target_q_total,
            target=0.9,
            loss_percent=np.array([0.1, 0.1, 0.1]),
        )
        assert_q_results_equal(model_new, model_old, res_new, res_old)

    def test_cp_obj_target_p_3ph(self):
        area_dir = Path("./")
        assert area_dir.exists()
        branch_data = pd.read_csv(branchdata_path)
        bus_data = pd.read_csv(bus_data_path)
        gen_data = pd.read_csv(gen_data_path)
        cap_data = pd.read_csv(cap_data_path)
        gen_data.loc[:, ["pa", "pb", "pc"]] *= 5
        gen_data.loc[:, ["sa_max", "sb_max", "sc_max"]] *= 5
        bus_data.loc[:, ["pl_a", "pl_b", "pl_c"]] *= 0.5
        bus_data.loc[:, ["ql_a", "ql_b", "ql_c"]] *= 0.5

        model_new = LinDistModelP(branch_data, bus_data, gen_data, cap_data)
        powerdata = pd.read_csv(legacy_powerdata_path)
        model_old = PModel(
            branch_data,
            powerdata,
            p_rating_mult=5,
            load_mult=0.5,
            v_up=1.05,
            p_base_gld=1e6,
            v_ll_base_gld=4160,
        )
        print("Solve old")
        model_old.loss_percent = np.array([0.1, 0.1, 0.1])
        res_old = model_old.solve(
            objective="p_target", target=np.array([0.3, 0.3, 0.3])
        )
        print("Solve new")
        res_new = opf_solver.cvxpy_solve(
            model_new,
            opf_solver.cp_obj_target_p_3ph,
            target=np.array([0.3, 0.3, 0.3]),
            loss_percent=np.array([0.1, 0.1, 0.1]),
        )
        assert_p_results_equal(model_new, model_old, res_new, res_old)

    def test_cp_obj_target_p_total(self):
        area_dir = Path("./")
        assert area_dir.exists()
        branch_data = pd.read_csv(branchdata_path)
        bus_data = pd.read_csv(bus_data_path)
        gen_data = pd.read_csv(gen_data_path)
        cap_data = pd.read_csv(cap_data_path)
        gen_data.loc[:, ["pa", "pb", "pc"]] *= 5
        gen_data.loc[:, ["sa_max", "sb_max", "sc_max"]] *= 5
        bus_data.loc[:, ["pl_a", "pl_b", "pl_c"]] *= 0.5
        bus_data.loc[:, ["ql_a", "ql_b", "ql_c"]] *= 0.5
        model_new = LinDistModelP(branch_data, bus_data, gen_data, cap_data)
        powerdata = pd.read_csv(legacy_powerdata_path)
        model_old = PModel(
            branch_data,
            powerdata,
            p_rating_mult=5,
            v_up=1.05,
            load_mult=0.5,
            p_base_gld=1e6,
            v_ll_base_gld=4160,
        )
        print("Solve old")
        model_old.loss_percent = np.array([0.1, 0.1, 0.1])
        res_old = model_old.solve(objective="p_target", target=0.9)
        print("Solve new")
        res_new = opf_solver.cvxpy_solve(
            model_new,
            opf_solver.cp_obj_target_p_total,
            target=0.9,
            loss_percent=np.array([0.1, 0.1, 0.1]),
        )
        assert_p_results_equal(model_new, model_old, res_new, res_old)

    def test_cp_obj_quadratic_curtail(self):
        area_dir = Path("./")
        assert area_dir.exists()
        branch_data = pd.read_csv(branchdata_path)
        bus_data = pd.read_csv(bus_data_path)
        gen_data = pd.read_csv(gen_data_path)
        cap_data = pd.read_csv(cap_data_path)
        gen_data.loc[:, ["pa", "pb", "pc"]] *= 10
        gen_data.loc[:, ["sa_max", "sb_max", "sc_max"]] *= 10
        bus_data.loc[:, ["pl_a", "pl_b", "pl_c"]] *= 0.1
        bus_data.loc[:, ["ql_a", "ql_b", "ql_c"]] *= 0.1
        model_new = LinDistModelP(branch_data, bus_data, gen_data, cap_data)
        powerdata = pd.read_csv(legacy_powerdata_path)
        model_old = PModel(
            branch_data,
            powerdata,
            p_rating_mult=10,
            v_up=1.05,
            load_mult=0.1,
            p_base_gld=1e6,
            v_ll_base_gld=4160,
        )
        print("Solve old")
        res_old = model_old.solve(objective="quadratic curtail")
        print("Solve new")
        res_new = opf_solver.cvxpy_solve(model_new, opf_solver.cp_obj_curtail)
        assert_p_results_equal(model_new, model_old, res_new, res_old)

    def test_cp_obj_curtail(self):
        branch_data = pd.read_csv(branchdata_path)
        bus_data = pd.read_csv(bus_data_path)
        gen_data = pd.read_csv(gen_data_path)
        cap_data = pd.read_csv(cap_data_path)
        gen_data.loc[:, ["pa", "pb", "pc"]] *= 10
        gen_data.loc[:, ["sa_max", "sb_max", "sc_max"]] *= 10
        bus_data.loc[:, ["pl_a", "pl_b", "pl_c"]] *= 0.1
        bus_data.loc[:, ["ql_a", "ql_b", "ql_c"]] *= 0.1
        model_new = LinDistModelP(branch_data, bus_data, gen_data, cap_data)
        powerdata = pd.read_csv(legacy_powerdata_path)
        model_old = PModel(
            branch_data,
            powerdata,
            p_rating_mult=10,
            v_up=1.05,
            load_mult=0.1,
            p_base_gld=1e6,
            v_ll_base_gld=4160,
        )
        print("Solve old")
        res_old = model_old.solve(objective="curtail")
        print("Solve new")
        res_new = opf_solver.solve_lin(
            model_new, opf_solver.gradient_curtail(model_new)
        )
        assert_p_results_equal(model_new, model_old, res_new, res_old)
