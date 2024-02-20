from opf import LinDistModelQ, cvxpy_solve, cp_obj_loss, gradient_load_min
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

branch_data = pd.read_csv("branchdata.csv", header=0)
bus_data = pd.read_csv("bus_data.csv", header=0)
gen_data = pd.read_csv("gen_data.csv", header=0)
cap_data = pd.read_csv("cap_data.csv", header=0)
bat_data = pd.read_csv("battery_data.csv", header=0)
loadshape_data = pd.read_csv("default_loadshape.csv", header=0)
pv_loadshape_data = pd.read_csv("pv_loadshape.csv",header=0)
print(loadshape_data.columns)
# modify generator power ratings to be 5x larger (alternatively, could modify csv directly or create helper function)
gen_data.loc[:, ["pa", "pb", "pc"]] *= 4
gen_data.loc[:, ["sa_max", "sb_max", "sc_max"]] *= 4
# Initialize model (create matrices, bounds, indexing methods, etc.)
model = LinDistModelQ(branch_data, bus_data, gen_data, cap_data, bat_data, loadshape_data, pv_loadshape_data)
print(5)
# Solve model using provided objective function
res = cvxpy_solve(model, cp_obj_loss)
print(res.fun)
v = model.get_v_solved(res.x)
s = model.get_s_solved(res.x)
if LinDistModelQ.der and LinDistModelQ.battery:
    dec_var, dec_d_var, dec_c_var, dec_b_var = model.get_q_dec_variables(res.x)
elif LinDistModelQ.der:
    dec_var = model.get_q_dec_variables(res.x)
elif LinDistModelQ.battery:
    dec_d_var, dec_c_var, dec_b_var = model.get_q_dec_variables(res.x)
else:
    None


print(v)
print(s)
print(dec_c_var)
print(dec_d_var)
sum=0
for t in range(LinDistModelQ.n):
    for j in range(1, model.nb):
        for a in "abc":
            if model.phase_exists(a, t, j):
                sum += res.x[model.idx("pij", j, a, t)[0]]
print(sum)
import matplotlib.pyplot as plt
import numpy as np

# Data
xvalues = [t for t in range(LinDistModelQ.n)]
for ph in "abc":
    for j in model.battery_bus[ph]:
        plt.figure()  # Create a new figure for each combination of ph and j
        dvalues = [dec_d_var[t][ph][j+1] for t in range(LinDistModelQ.n)]
        cvalues = [dec_c_var[t][ph][j+1] for t in range(LinDistModelQ.n)]
        qvalues = [dec_var[t][ph][j+1] for t in range(LinDistModelQ.n)]
        load = [model.bus["pl_"+ph][j] * model.loadshape["M"][t] for t in range(LinDistModelQ.n)]
        pv = [model.gen["p"+ph][j] * model.pv_loadshape["PV"][t] for t in range(LinDistModelQ.n)]
        soc = [dec_b_var[t][ph][j+1] for t in range(LinDistModelQ.n)]
        vvalues = [v[t][ph][j+1] for t in range(LinDistModelQ.n)]
        plt.plot(xvalues,[val * 1e3 for val in load],label="load variation over time", color="red")
        plt.plot(xvalues, [val*1e3 for val in pv], label="PV active power variation over time", color="green")
        plt.legend()
        plt.xlabel("time(h)")
        plt.ylabel("Power(KW)")
        plt.title("Load and PV generation for node {} phase {}".format(j + 1, ph))
        plt.figure()
        plt.bar(xvalues, [val * 1e3 for val in dvalues], label="Discharge", color="green")
        plt.bar(xvalues, [-val * 1e3 for val in cvalues], label="Charge", color="red")
        plt.xlabel("time(h)")
        plt.ylabel("Discharge/charge Power(KW)")
        plt.title("SCD for node {} phase {}".format(j+1, ph))
        plt.legend()  # Show legend
        plt.figure()
        plt.bar(xvalues, [val*1e3 for val in qvalues])
        plt.xlabel("time(h)")
        plt.ylabel("q control PV")
        plt.title("reactive control for node {} phase {} PV".format(j+1, ph))
        plt.figure()
        plt.bar(xvalues, [(val/0.02)*100 for val in soc])
        plt.xlabel("time(h)")
        plt.ylabel("s.o.c (%)")
        plt.figure()
        plt.plot(xvalues, [np.sqrt(val) for val in vvalues])
        plt.xlabel("time(h)")
        plt.ylabel("voltage in p.u")
        plt.title("voltage of node {} phase {}".format(j+1,ph))
        plt.show()  # Show the figure for each ph of each j

# plt.figure(1)
# plt.bar(xvalues,[val*1e3 for val in dvalues], label="Discharge", color="green")
# plt.bar(xvalues,[-val*1e3 for val in cvalues], label="Charge", color="red")
# plt.xlabel("time(h)")
# plt.ylabel("Dicharge/charge Power(KW)")
# plt.title("SCD for node 3 phase a battery")
# plt.legend()
# plt.figure(2)
# plt.bar(xvalues,[(val/0.02)*100 for val in soc])
# plt.xlabel("time(h)")
# plt.ylabel("s.o.c (%)")
# plt.title("S.O.C for node 3 phase a battery")
# plt.legend()
# plt.figure(3)
# plt.bar(xvalues,[val*1e3 for val in qvalues])
# plt.xlabel("time(h)")
# plt.ylabel("q control PV")
# plt.title("reactive control for node 3 phase a PV")
# plt.figure(4)
# plt.plot(xvalues,[np.sqrt(val) for val in vvalues])
# plt.xlabel("time(h)")
# plt.ylabel("voltage in p.u")
# plt.title("voltage of node 3 phase a ")
# xvalues = [t for t in range(LinDistModelQ.n)]
# dvalues1 = [dec_d_var[t]["a"][50] for t in range(LinDistModelQ.n)]
# cvalues1 = [dec_c_var[t]["a"][50] for t in range(LinDistModelQ.n)]
# qvalues1 = [dec_var[t]["a"][50] for t in range(LinDistModelQ.n)]
# soc1= [dec_b_var[t]["a"][50] for t in range(LinDistModelQ.n)]
# vvalues1 = [v[t]["a"][50] for t in range(LinDistModelQ.n)]
# plt.figure(5)
# plt.bar(xvalues,[val*1e3 for val in dvalues1], label="Discharge", color="green")
# plt.bar(xvalues,[-val*1e3 for val in cvalues1], label="Charge", color="red")
# plt.xlabel("time(h)")
# plt.ylabel("Dicharge/charge Power(KW)")
# plt.title("SCD for node 50 phase a battery")
# plt.legend()
# plt.figure(6)
# plt.bar(xvalues,[(val/0.02)*100 for val in soc1])
# plt.xlabel("time(h)")
# plt.ylabel("s.o.c (%)")
# plt.title("S.O.C for node 50 phase a battery")
# plt.figure(7)
# plt.bar(xvalues,[val*1e3 for val in qvalues1])
# plt.xlabel("time(h)")
# plt.ylabel("q control PV")
# plt.title("reactive control for node 50 phase a PV")
# plt.figure(8)
# plt.plot(xvalues,[np.sqrt(val) for val in vvalues1])
# plt.xlabel("time(h)")
# plt.ylabel("voltage in p.u")
# plt.title("voltage of node 50 phase a")
# plt.show()
# # with np.printoptions(threshold=np.inf):
# #     print(np.nonzero(model.a_eq))
# plt.show()
print(5)