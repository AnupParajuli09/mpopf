from time import perf_counter

import cvxpy as cp
import numpy as np
from scipy.optimize import OptimizeResult, linprog
from scipy.sparse import csr_array
from opf import LinDistModelQ


def gradient_load_min(model):
    c = np.zeros(model.n_x)
    for ph in "abc":
        if model.phase_exists(ph):
            c[model.branches_out_of_j("pij", 0, ph)] = 1
    return c


def gradient_curtail(model):
    c = np.zeros(model.n_x)
    for i in range(
        model.p_der_start_phase_idx["a"],
        model.p_der_start_phase_idx["c"] + len(model.der_bus["c"]),
    ):
        c[i] = -1
    return c


# ~~~ Quadratic objective with linear constraints for use with solve_quad()~~~
# def cp_obj_loss(model, xk):
#     f: cp.Expression = 0
#     for t in range(LinDistModelQ.n):
#         for j in range(1, model.nb):
#             for a in "abc":
#                 if model.phase_exists(a, t, j):
#                     i = model.idx("bi", j, a, t)
#                     f += model.r[a + a][i, j] * (xk[model.idx("pij", j, a, t)[0]] ** 2)
#                     f += model.r[a + a][i, j] * (xk[model.idx("qij", j, a, t)[0]] ** 2)
#                     dis = model.idx("pd", j, a, t)
#                     ch = model.idx("pc", j, a, t)
#                     if ch:
#                         f += 1e-1*(1 - 0.95) * (xk[ch])
#                     if dis:
#                         f += 1e-1*((1/0.95)-1) * (xk[dis])
#     return f

def cp_obj_loss(model, xk):
    f: cp.Expression = 0
    for t in range(LinDistModelQ.n):
        for j in range(1, model.nb):
            for a in "abc":
                if model.phase_exists(a, t, j):
                    i = model.idx("bi", j, a, t)
                    f += model.r[a + a][i, j] * (xk[model.idx("pij", j, a, t)[0]] ** 2)
                    f += model.r[a + a][i, j] * (xk[model.idx("qij", j, a, t)[0]] ** 2)
                    if LinDistModelQ.battery:
                        dis = model.idx("pd", j, a, t)
                        ch = model.idx("pc", j, a, t)
                        if ch:
                            f += 1e-1*(1 - model.bat["nc_" + a].get(j,1)) * (xk[ch])
                        if dis:
                            f += 1e-1*((1/model.bat["nd_" + a].get(j,1))-1) * (xk[dis])
    return f


def cp_obj_target_p_3ph(model, xk, **kwargs):
    f = cp.Constant(0)
    target = kwargs["target"]
    loss_percent = kwargs.get("loss_percent", np.zeros(3))
    for i, ph in enumerate("abc"):
        if model.phase_exists(ph):
            p = 0
            for out_branch in model.branches_out_of_j("pij", 0, ph):
                p = p + xk[out_branch]
            f += (target[i] - p * (1 + loss_percent[i] / 100)) ** 2
    return f


def cp_obj_target_p_total(model, xk, **kwargs):
    actual = 0
    target = kwargs["target"]
    loss_percent = kwargs.get("loss_percent", np.zeros(3))
    for i, ph in enumerate("abc"):
        if model.phase_exists(ph):
            p = 0
            for out_branch in model.branches_out_of_j("pij", 0, ph):
                p = p + xk[out_branch]
            actual += p
    f = (target - actual * (1 + loss_percent[0] / 100)) ** 2
    return f


def cp_obj_target_q_3ph(model, xk, **kwargs):
    target_q = kwargs["target"]
    loss_percent = kwargs.get("loss_percent", np.zeros(3))
    f = cp.Constant(0)
    for i, ph in enumerate("abc"):
        if model.phase_exists(ph):
            q = 0
            for out_branch in model.branches_out_of_j("qij", 0, ph):
                q = q + xk[out_branch]
            f += (target_q[i] - q * (1 + loss_percent[i] / 100)) ** 2
    return f


def cp_obj_target_q_total(model, xk, **kwargs):
    actual = 0
    target = kwargs["target"]
    loss_percent = kwargs.get("loss_percent", np.zeros(3))
    for i, ph in enumerate("abc"):
        if model.phase_exists(ph):
            q = 0
            for out_branch in model.branches_out_of_j("qij", 0, ph):
                q = q + xk[out_branch]
            actual += q
    f = (target - actual * (1 + loss_percent[0] / 100)) ** 2
    return f


def cp_obj_curtail(model, xk):
    f = cp.Constant(0)
    for i in range(model.p_der_start_phase_idx["a"], model.q_der_start_phase_idx["a"]):
        f += (model.bounds[i][1] - xk[i]) ** 2
    return f


def cp_obj_none(model, xk):
    return cp.Constant(0)


def cvxpy_solve(model, obj_func, **kwargs):
    m = model
    tic = perf_counter()
    solver = kwargs.get("solver", cp.OSQP)
    x0 = kwargs.get("x0", None)
    if x0 is None:
        lin_res = solve_lin(m, np.zeros(m.n_x))
        x0 = lin_res.x.copy()
    x = cp.Variable(shape=(m.n_x,), name="x", value=x0)
    g = [m.a_eq @ x - m.b_eq.flatten() == 0]
    if LinDistModelQ.battery:
        h = [m.a_ineq @ x - m.b_ineq.flatten() <= 0]
    else:
        h = []
    lb = [x[i] >= m.bounds[i][0] for i in range(m.n_x)]
    ub = [x[i] <= m.bounds[i][1] for i in range(m.n_x)]
    loss_percent = kwargs.get("loss_percent", np.zeros(3))
    target = kwargs.get("target", None)
    if target is not None:
        expression = obj_func(m, x, target=target, loss_percent=loss_percent)
    else:
        expression = obj_func(m, x)
    prob = cp.Problem(cp.Minimize(expression), g + h + ub + lb)
    prob.solve(verbose=False, solver=solver)

    x_res = x.value
    result = OptimizeResult(
        fun=prob.value,
        success=(prob.status == "optimal"),
        message=prob.status,
        x=x_res,
        nit=prob.solver_stats.num_iters,
        runtime=perf_counter() - tic,
    )
    return result


def solve_lin(model, c):
    tic = perf_counter()
    if LinDistModelQ.battery:
        res = linprog(
            c, A_eq=csr_array(model.a_eq), b_eq=model.b_eq.flatten(), A_ub=csr_array(model.a_ineq), b_ub= model.b_ineq.flatten(), bounds=model.bounds
        )
    else:
        res = linprog(
        c, A_eq=csr_array(model.a_eq), b_eq=model.b_eq.flatten(), bounds=model.bounds
    )
    runtime = perf_counter() - tic
    res["runtime"] = runtime
    return res
