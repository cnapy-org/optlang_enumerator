#%%
import cobra
import optlang_enumerator.cobra_cnapy
import optlang
import optlang_enumerator.mcs_computation as mcs_computation
import numpy
import pickle
from pathlib import Path
results_cache_dir = None # do not cache preprocessing results
# results_cache_dir = Path(r"E:\cnapy_tmp\results_cache") # cache preprocessing results in the given directory

#%% 
from importlib import reload
import optlang_enumerator
reload(optlang_enumerator)
import optlang_enumerator.mcs_computation as mcs_computation

#%%
ecc2 = optlang_enumerator.cobra_cnapy.CNApyModel.read_sbml_model("ECC2comp.sbml")
# allow all reactions that are not boundary reactions as cuts (same as exclude_boundary_reactions_as_cuts option of compute_mcs)
cuts = numpy.array([not r.boundary for r in ecc2.reactions])
reac_id = ecc2.reactions.list_attr('id') # list of reaction IDs in the model
# define target (multiple targets are possible; each target can have multiple linear inequality constraints)
ecc2_mue_target = [[("Growth", ">=", 0.01)]] # one target with one constraint, a.k.a. syntehtic lethals
# this constraint alone would not be sufficient, but there are uptake limits defined in the reaction bounds
# of the model that are integerated automatically by the compute_mcs function into all target and desired regions
# convert into matrix/vector relation format
ecc2_mue_target = [mcs_computation.relations2leq_matrix(
                   mcs_computation.parse_relations(t, reac_id_symbols=mcs_computation.get_reac_id_symbols(reac_id)), reac_id)
                   for t in ecc2_mue_target]
# convert non-default bounds of the newtork model into matrix/vector relation format
# in this network these are substrate uptake bounds
# bounds_mat, bounds_rhs = mcs_computation.reaction_bounds_to_leq_matrix(ecc2)
# integrate the relations defined through the network bounds into every target (still matrix/vector relation format)
# ecc2_mue_target = [(scipy.sparse.vstack((t[0], bounds_mat), format='csr'), numpy.hstack((t[1], bounds_rhs))) for t in ecc2_mue_target]
mcs_computation.integrate_model_bounds(ecc2, ecc2_mue_target)
# convert into constraints that can be added to the COBRApy model (e.g. in context)
ecc2_mue_target_constraints= mcs_computation.get_leq_constraints(ecc2, ecc2_mue_target)
for c in ecc2_mue_target_constraints[0]: # print constraints that make up the first target
    print(c)
#%%
ecc2_mcs,_ = mcs_computation.compute_mcs(ecc2, ecc2_mue_target, cuts=cuts, enum_method=3, max_mcs_size=3, network_compression=True,
                                         include_model_bounds=False, results_cache_dir=results_cache_dir)
print(len(ecc2_mcs))
# show MCS as n-tuples of reaction IDs
ecc2_mcs_rxns= [tuple(reac_id[r] for r in mcs) for mcs in ecc2_mcs]
print(ecc2_mcs_rxns)
# check that all MCS disable the target
print(all(mcs_computation.check_mcs(ecc2, ecc2_mue_target[0], ecc2_mcs, optlang.interface.INFEASIBLE)))

# %% same calculation without network compression
ecc2_mcsF,_ = mcs_computation.compute_mcs(ecc2, ecc2_mue_target, cuts=cuts, enum_method=3, max_mcs_size=3, network_compression=False,
                                          include_model_bounds=False, results_cache_dir=results_cache_dir)
print(set(ecc2_mcs) == set(ecc2_mcsF))

# %% desired behaviour for cMCS calculation
ecc2_ethanol_desired = [[("EthEx", ">=", 1)]]
ecc2_ethanol_desired = [mcs_computation.relations2leq_matrix(
                   mcs_computation.parse_relations(t, reac_id_symbols=mcs_computation.get_reac_id_symbols(reac_id)), reac_id)
                   for t in ecc2_ethanol_desired]
mcs_computation.integrate_model_bounds(ecc2, ecc2_ethanol_desired)
ecc2_ethanol_desired_constraints= mcs_computation.get_leq_constraints(ecc2, ecc2_ethanol_desired)
for c in ecc2_ethanol_desired_constraints[0]: # print constraints that make up the first desired region
    print(c)

# %% calculate cMCS
ecc2_cmcs,_ = mcs_computation.compute_mcs(ecc2, ecc2_mue_target, desired=ecc2_ethanol_desired, cuts=cuts, enum_method=3, max_mcs_size=3, network_compression=True,
                                          include_model_bounds=False, results_cache_dir=results_cache_dir)

# %% check cMCS
print(len(ecc2_cmcs))
# all cMCS disable the target
print(all(mcs_computation.check_mcs(ecc2, ecc2_mue_target[0], ecc2_cmcs, optlang.interface.INFEASIBLE)))
# all cMCS allow the desired behaviour
print(all(mcs_computation.check_mcs(ecc2, ecc2_ethanol_desired[0], ecc2_cmcs, optlang.interface.OPTIMAL)))
# cMCS are a subset of the MCS 
print(set(ecc2_cmcs).issubset(ecc2_mcs))
# cMCS are those MCS that preserve the desired behaviour
print(set(ecc2_cmcs) ==
    set(m for m,c in zip(ecc2_mcs, mcs_computation.check_mcs(ecc2, ecc2_ethanol_desired[0], ecc2_mcs, optlang.interface.OPTIMAL)) if c))

# %%
with open("ecc2_mcs.pkl","rb") as f:
    ref = pickle.load(f)
set(ecc2_mcs) - ref

# %% full FVA
# with ecc2 as model:
model = ecc2.copy() # copy model because switching solver in context sometimes gives an error (?!?)
model.objective = model.problem.Objective(0)
fva_tol = 1e-8 # with CPLEX 1e-8 leads to removal of EX_adp_c, 1e-9 keeps EX_adp_c
model.tolerance = fva_tol # prevent essential EX_meoh_ex from being blocked, sets solver feasibility/optimality tolerances
# model.solver.configuration.tolerances.feasibility = 1e-9
model.solver = 'glpk_exact' # appears to make problems for context management
# model_mue_target_constraints= get_leq_constraints(model, ecc2_mue_target)
# model.add_cons_vars(model_mue_target_constraints[0])
fva_res = cobra.flux_analysis.flux_variability_analysis(model, fraction_of_optimum=0, processes=1) # no interactive multiprocessing on Windows
print(fva_res.loc['EX_adp_c',:])
blocked = []
blocked_rxns = []
for i in range(fva_res.values.shape[0]):
    if fva_res.values[i, 0] >= -fva_tol and fva_res.values[i, 1] <= fva_tol:
        blocked.append(i)
        blocked_rxns.append(fva_res.index[i])
print(blocked_rxns)

#%%
import efmtool_link.efmtool4cobra as efmtool4cobra
ecc2c = ecc2.copy()
subT = efmtool4cobra.compress_model_sympy(ecc2c, remove_rxns=blocked_rxns)
print(len(ecc2c.metabolites), len(ecc2c.reactions))
rev_rd = [r.reversibility for r in ecc2c.reactions]
efmtool4cobra.remove_conservation_relations_sympy(ecc2c)
# reduced = cobra.util.array.create_stoichiometric_matrix(ecc2c, array_type='dok', dtype=numpy.object)
# rd = efmtool4cobra.dokRatMat2lilFloatMat(reduced)
rd = cobra.util.array.create_stoichiometric_matrix(ecc2c, array_type='lil')

# %% check that context restores hash values/objects
print(ecc2.reactions[0].hash_value, ecc2.reactions[0].bounds, ecc2.stoichiometry_hash_object,
        ecc2.stoichiometry_hash_object.digest(), "\n")
with ecc2 as model:
    model.reactions[0].bounds = (0, 100)
    model.reactions[0].set_hash_value()
    model.set_stoichiometry_hash_object()
    print(model.reactions[0].hash_value, model.reactions[0].bounds, model.stoichiometry_hash_object,
           model.stoichiometry_hash_object.digest(), "\n")
print(ecc2.reactions[0].hash_value, ecc2.reactions[0].bounds, ecc2.stoichiometry_hash_object,
        ecc2.stoichiometry_hash_object.digest(), "\n")

# %% OK
ecc2B = cobra.io.model_from_dict(cobra.io.model_to_dict(ecc2))
all(r1.id==r2.id for r1,r2 in zip(ecc2.reactions, ecc2B.reactions))

# %% OK
# ecc2cB = cobra.io.model_from_dict(compressed_model_to_dict(ecc2c))
# ecc2cB = cobra.io.model_from_dict(pickle.loads(pickle.dumps(compressed_model_to_dict(ecc2c))))
with open("test.pkl", "wb") as file:
    pickle.dump((mcs_computation.compressed_model_to_dict(ecc2c), 42), file)
with open("test.pkl", "rb") as file:
    (ecc2cB,_) = pickle.load(file)
    ecc2cB = cobra.io.model_from_dict(ecc2cB)
print(all(r1.id==r2.id for r1,r2 in zip(ecc2c.reactions, ecc2cB.reactions)))
print(all(all(r1.subset_rxns==r2.subset_rxns) for r1,r2 in zip(ecc2c.reactions, ecc2cB.reactions)))
print(all(all(r1.subset_stoich==r2.subset_stoich) for r1,r2 in zip(ecc2c.reactions, ecc2cB.reactions)))
# %%
