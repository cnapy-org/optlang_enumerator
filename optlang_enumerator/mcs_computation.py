import numpy
import scipy
from pathlib import Path
import pickle
import hashlib
import pandas
import cobra
from cobra.io.dict import _ORDERED_OPTIONAL_REACTION_KEYS, _OPTIONAL_REACTION_ATTRIBUTES
import optlang_enumerator.cobra_cnapy
import optlang.glpk_interface
from swiglpk import GLP_DUAL
try:
    import optlang.cplex_interface
except:
    optlang.cplex_interface = None # make sure this symbol is defined for type() comparisons
try:
    import optlang.gurobi_interface
except:
    optlang.gurobi_interface = None # make sure this symbol is defined for type() comparisons
try:
    import optlang.coinor_cbc_interface
except:
    optlang.coinor_cbc_interface = None # make sure this symbol is defined for type() comparisons
import itertools
from typing import List, Tuple, Union, Set, FrozenSet
import sympy
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
from cobra.core.configuration import Configuration
import efmtool_link.efmtool4cobra as efmtool4cobra
import optlang_enumerator.cMCS_enumerator as cMCS_enumerator

def expand_mcs(mcs: List[Union[Tuple[int], Set[int], FrozenSet[int]]], subT) -> List[Tuple[int]]:
    mcs = [[list(m)] for m in mcs] # list of lists; mcs[i] will contain a list of MCS expanded from it
    rxn_in_sub = [numpy.where(subT[:, i])[0] for i in range(subT.shape[1])]
    for i in range(len(mcs)):
        num_iv = len(mcs[i][0]) # number of interventions in this MCS
        for s_idx in range(num_iv): # subset index
            for j in range(len(mcs[i])):
                rxns = rxn_in_sub[mcs[i][j][s_idx]]
                mcs[i][j][s_idx] = rxns[0]
                for k in range(1, len(rxns)):
                    mcs[i].append(mcs[i][j].copy())
                    mcs[i][-1][s_idx] = rxns[k]
    mcs = list(itertools.chain(*mcs))
    return list(map(tuple, map(numpy.sort, mcs)))

def matrix_row_expressions(mat, vars):
    # mat can be a numpy matrix or scipy sparse matrix (csc, csr, lil formats work; COO/DOK formats do not work)
    # expr = [None] * mat.shape[0]
    # for i in range(mat.shape[0]):
    #     idx = numpy.nonzero(mat)
    ridx, cidx = mat.nonzero() # !! assumes that the indices in ridx are grouped together, not fulfilled for DOK !! 
    if len(ridx) == 0:
        return []
    # expr = []
    expr = [None] * mat.shape[0]
    first = 0
    current_row = ridx[0]
    i = 1
    while True:
        at_end = i == len(ridx)
        if at_end or ridx[i] != current_row:
            # expr[current_row] = sympy.simplify(add([mat[current_row, c] * vars[c] for c in cidx[first:i]])) # simplify to flatten the sum, slow/hangs
            expr[current_row] = sympy.Add(*[mat[current_row, c] * vars[c] for c in cidx[first:i]]) # gives flat sum
            # expr[current_row] = sum([mat[current_row, c] * vars[c] for c in cidx[first:i]]) # gives flat sum, slow/hangs
            if at_end:
                break
            first = i
            current_row = ridx[i]
        i = i + 1
    return expr

def leq_constraints(optlang_constraint_class, row_expressions, rhs):
    return [optlang_constraint_class(expr, ub=ub) for expr, ub in zip(row_expressions, rhs)]

def check_mcs(model, constr, mcs, expected_status, knock_in_idx=frozenset(), flux_expr=None):
    # mcs: list of tuples/sets
    check_ok= numpy.zeros(len(mcs), dtype=bool)
    with model as constr_model:
        constr_model.problem.Objective(0)
        if isinstance(constr[0], optlang.interface.Constraint):
            constr_model.add_cons_vars(constr)
        else:
            if flux_expr is None:
                flux_expr = [r.flux_expression for r in constr_model.reactions]
            rexpr = matrix_row_expressions(constr[0], flux_expr)
            constr_model.add_cons_vars(leq_constraints(constr_model.problem.Constraint, rexpr, constr[1]))
        for m in range(len(mcs)):
            with constr_model as KO_model:
                cuts = mcs[m]
                if len(knock_in_idx): # assumes that cuts is a tuple of indices
                    cuts = set(cuts)
                    cuts.symmetric_difference_update(knock_in_idx) # knock out all unused knock-ins but keep the used ones
                for r in cuts:
                    if isinstance(r, str):
                        KO_model.reactions.get_by_id(r).knock_out()
                    else: # assume r is an index if it is not a string
                        KO_model.reactions[r].knock_out()
                KO_model.slim_optimize()
                check_ok[m] = KO_model.solver.status == expected_status
    return check_ok

from swiglpk import glp_adv_basis # for direkt use of glp_exact, experimental only
def make_minimal_cut_set(model, cut_set, target_constraints):
    original_bounds = [model.reactions[r].bounds for r in cut_set]
    keep_ko = [True] * len(cut_set)
    try:
        for r in cut_set:
            model.reactions[r].knock_out()
        for i in range(len(cut_set)):
            r = cut_set[i]
            model.reactions[r].bounds = original_bounds[i]
            still_infeasible = True
            for target in target_constraints:
                with model as target_model:
                    target_model.problem.Objective(0)
                    target_model.add_cons_vars(target)
                    if type(target_model.solver) is optlang.glpk_exact_interface.Model:
                        target_model.solver.update() # need manual update because GLPK is called through private function
                        status = target_model.solver._run_glp_exact() # optimize would run GLPK first
                        if status == 'undefined':
                            # print('Making fresh model')
                            # target_model_copy = target_model.copy() # kludge to lose the old basis
                            # status = target_model_copy.solver._run_glp_exact()
                            print("Make new basis")
                            glp_adv_basis(target_model.solver.problem, 0) # probably not with rational arithmetric?
                            status = target_model.solver._run_glp_exact() # optimize would run GLPK first
                        print(status)
                    else:
                        target_model.slim_optimize()
                        status = target_model.solver.status
                    still_infeasible = still_infeasible and status == optlang.interface.INFEASIBLE
                    if still_infeasible is False:
                        break
            if still_infeasible:
                keep_ko[i] = False # this KO is redundant
            else:
                model.reactions[r].knock_out() # reactivate
        mcs = tuple(ko for(ko, keep) in zip(cut_set, keep_ko) if keep)
    # don't handle the exception, just make sure the model is restored
    finally:
        for i in range(len(cut_set)):
            r = cut_set[i]
            model.reactions[r].bounds = original_bounds[i]
        model.solver.update() # just in case...
    return mcs

def make_minimal_intervention_set(model, interventions: list, target_constraints,
                                  desired_constraints=None, knock_in_idx=frozenset()):
    intervention_set = set(interventions)
    original_bounds = [model.reactions[r].bounds for r in interventions]
    with model as KO_model:
        for r in knock_in_idx - intervention_set: # knock-ins that are not active in this intervention set
            # print("deactivate", r)
            KO_model.reactions[r].knock_out()
        keep_intervention = [True] * len(interventions)
        try:
            for r in intervention_set - knock_in_idx: # leave active knock-ins operational
                if r not in knock_in_idx:
                    # print("knock out", r)
                    KO_model.reactions[r].knock_out()
            for i in range(len(interventions)):
                r = interventions[i]
                if r in knock_in_idx:
                    # print("Checking KI", r)
                    is_knock_in = True
                    KO_model.reactions[r].knock_out()
                else:
                    # print("Checking KO", r)
                    is_knock_in = False
                    KO_model.reactions[r].bounds = original_bounds[i]
                targets_still_infeasible = True
                for target in target_constraints:
                    with KO_model as target_model:
                        target_model.problem.Objective(0)
                        target_model.add_cons_vars(target)
                        target_model.slim_optimize()
                        targets_still_infeasible = target_model.solver.status == optlang.interface.INFEASIBLE
                        if not targets_still_infeasible:
                            break
                desired_still_feasible = True
                if is_knock_in and desired_constraints is not None:
                    for desired in desired_constraints:
                        with KO_model as desired_model:
                            desired_model.problem.Objective(0)
                            desired_model.add_cons_vars(desired)
                            desired_model.slim_optimize()
                            desired_still_feasible = desired_model.solver.status == optlang.interface.OPTIMAL
                            if not desired_still_feasible:
                                break
                if targets_still_infeasible and desired_still_feasible: # this intervention is redundant
                    keep_intervention[i] = False
                else: # this intervention is necessary
                    if is_knock_in: # reactivate
                        KO_model.reactions[r].bounds = original_bounds[i]
                    else:
                        KO_model.reactions[r].knock_out()
            mcs = tuple(ko for(ko, keep) in zip(interventions, keep_intervention) if keep)
        # don't handle the exception, just make sure KO_model is restored
        finally:
            for i in range(len(interventions)):
                r = interventions[i]
                KO_model.reactions[r].bounds = original_bounds[i]
            KO_model.solver.update() # just in case...
    return mcs

def parse_relation(lhs : str, rhs : float, reac_id_symbols=None):
    transformations = (standard_transformations + (implicit_multiplication_application,))
    slash = lhs.find('/')
    if slash >= 0:
        denominator = lhs[slash+1:]
        numerator = lhs[0:slash]
        denominator = parse_expr(denominator, transformations=transformations, evaluate=False, local_dict=reac_id_symbols)
        denominator = sympy.collect(sympy.expand(denominator), denominator.free_symbols)
        numerator = parse_expr(numerator, transformations=transformations, evaluate=False, local_dict=reac_id_symbols)
        numerator = sympy.collect(sympy.expand(numerator), numerator.free_symbols)
        lhs = numerator - rhs*denominator
        rhs = 0
    else:
        lhs = parse_expr(lhs, transformations=transformations, evaluate=False, local_dict=reac_id_symbols)
    lhs = sympy.collect(sympy.expand(lhs), lhs.free_symbols, evaluate=False)
    
    return lhs, rhs

def parse_relations(relations: list, reac_id_symbols=None):
    for r in range(len(relations)):
        lhs, rhs = parse_relation(relations[r][0], relations[r][2], reac_id_symbols=reac_id_symbols)
        relations[r] = (lhs, relations[r][1], rhs)
    return relations

def get_reac_id_symbols(reac_id) -> dict:
    return {rxn: sympy.symbols(rxn) for rxn in reac_id}

def get_reaction_id_symbols(reactions: cobra.DictList) -> dict:
    return {rxn: sympy.symbols(rxn.id) for rxn in reactions}

def relations2leq_matrix(relations : List, variables):
    num_inequalities = len(relations)
    for rel in relations:
        if rel[1] == "=":
            num_inequalities += 1
    matrix = numpy.zeros((num_inequalities, len(variables)))
    rhs = numpy.zeros(num_inequalities)
    i = 0
    for rel in relations:
        if rel[1] == ">=":
            f = -1.0
        elif rel[1] == "<=" or rel[1] == "=":
            f = 1.0
        else:
            raise ValueError('Only "<=", ">=" and "=" relations are supported.')
        for r,c in rel[0].items(): # the keys are symbols
            matrix[i][variables.index(str(r))] = f*c
        rhs[i] = f*rel[2]
        i += 1
        if rel[1] == "=":
            matrix[i, :] = -matrix[i-1, :]
            rhs[i] = -rhs[i-1]
            i += 1
    return matrix, rhs # matrix <= rhs

def get_leq_constraints(model: cobra.Model, leq_mat : List[Tuple], flux_expr=None):
    # leq_mat can be either targets or desired (as matrices)
    # returns contstraints that can be added to model 
    if flux_expr is None:
        flux_expr = [r.flux_expression for r in model.reactions]
    return [leq_constraints(model.problem.Constraint, matrix_row_expressions(lqm[0], flux_expr), lqm[1]) for lqm in leq_mat]

def reaction_bounds_to_leq_matrix(model):
    config = Configuration()
    lb_idx = []
    ub_idx = []
    for i in range(len(model.reactions)):
        if model.reactions[i].lower_bound not in (0, config.lower_bound, -float('inf')):
            lb_idx.append(i)
            # print(model.reactions[i].id, model.reactions[i].lower_bound)
        if model.reactions[i].upper_bound not in (0, config.upper_bound, float('inf')):
            ub_idx.append(i)
            # print(model.reactions[i].id, model.reactions[i].upper_bound)
    num_bounds = len(lb_idx) + len(ub_idx)
    leq_mat = scipy.sparse.lil_matrix((num_bounds, len(model.reactions)))
    rhs = numpy.zeros(num_bounds)
    count = 0
    for r in lb_idx:
        leq_mat[count, r] = -1.0
        rhs[count] = -model.reactions[r].lower_bound
        count += 1
    for r in ub_idx:
        leq_mat[count, r] = 1.0
        rhs[count] = model.reactions[r].upper_bound
        count += 1
    return leq_mat, rhs

def integrate_model_bounds(model, targets, desired=None):
    bounds_mat, bounds_rhs = reaction_bounds_to_leq_matrix(model)
    for i in range(len(targets)):
        targets[i] = (scipy.sparse.vstack((targets[i][0], bounds_mat), format='lil'), numpy.hstack((targets[i][1], bounds_rhs)))
    if desired is not None:
        for i in range(len(desired)):
            desired[i] = (scipy.sparse.vstack((desired[i][0], bounds_mat), format='lil'), numpy.hstack((desired[i][1], bounds_rhs)))

def compressed_model_to_dict(model):
    global _ORDERED_OPTIONAL_REACTION_KEYS, _OPTIONAL_REACTION_ATTRIBUTES
    subset_attributes = ["subset_rxns", "subset_stoich"]
    _ORDERED_OPTIONAL_REACTION_KEYS += subset_attributes
    for attr in subset_attributes:
        _OPTIONAL_REACTION_ATTRIBUTES[attr] = []
    try:
        model_dict = cobra.io.model_to_dict(model)
    except:
        raise
    finally:
        _ORDERED_OPTIONAL_REACTION_KEYS = _ORDERED_OPTIONAL_REACTION_KEYS[:-len(subset_attributes)]
        for attr in subset_attributes:
            del _OPTIONAL_REACTION_ATTRIBUTES[attr]
    return model_dict

def flux_variability_analysis(model: optlang_enumerator.cobra_cnapy.cobra.Model, loopless=False, fraction_of_optimum=0.0,
                              processes=None, results_cache_dir: Path=None, fva_hash=None, print_func=print):
    # all bounds in the model must be finite because the COBRApy FVA treats unbounded results as errors
    model_stoichiometry_hash_object = model.stoichiometry_hash_object
    model._stoichiometry_hash_object = None # in case model needs to be pickled
    if results_cache_dir is not None:
        fva_hash.update(pickle.dumps((loopless, fraction_of_optimum, model.tolerance))) # integrate solver tolerances?
        fva_hash.update(pickle.dumps(model.reactions.list_attr("objective_coefficient")))
        fva_hash.update(model.objective_direction.encode())
        file_path = results_cache_dir / (model.id+"_FVA_"+fva_hash.hexdigest())
        fva_result = None
        if Path.exists(file_path):
            try:
                fva_result = pandas.read_pickle(file_path)
                print_func("Loaded FVA result from", str(file_path))
            except:
                print_func("Loading FVA result from", str(file_path), "failed, running FVA.")
        else:
            print_func("No cached result available, running FVA...")
        if fva_result is None:
            fva_result = cobra.flux_analysis.flux_variability_analysis(model, reaction_list=None, loopless=loopless,
                                                             fraction_of_optimum=fraction_of_optimum,
                                                             pfba_factor=None, processes=processes)
            try:
                fva_result.to_pickle(file_path)
                print_func("Saved FVA result to ", str(file_path))
            except:
                print_func("Failed to write FVA result to ", str(file_path))
    else:
        fva_result = cobra.flux_analysis.flux_variability_analysis(model, reaction_list=None, loopless=loopless,
                                                             fraction_of_optimum=fraction_of_optimum,
                                                             pfba_factor=None, processes=processes)
    model.restore_stoichiometry_hash_object(model_stoichiometry_hash_object)
    return fva_result

class InfeasibleRegion(Exception):
    pass

# convenience function
def compute_mcs(model: optlang_enumerator.cobra_cnapy.cobra.Model, targets, desired=None,
                cuts=None, knock_in_idx=None, intervention_costs=None,
                enum_method=1, max_mcs_size=2, max_mcs_num=1000, timeout=600, use_kn_in_dual=False,
                exclude_boundary_reactions_as_cuts=False, network_compression:bool=True, fva_tolerance=1e-9,
                include_model_bounds=True, bigM=0, mip_opt_tol=1e-6, mip_feas_tol=1e-6, mip_int_tol=1e-6,
                set_mip_parameters_callback=None, results_cache_dir: Path=None) -> List[Tuple[int]]:
    # if include_model_bounds=True this function integrates non-default reaction bounds of the model into the
    # target and desired regions and directly modifies(!) these parameters
    if desired is None:
        desired = []
    if knock_in_idx is None:
        knock_in_idx = []

    flux_expr = [r.flux_expression for r in model.reactions]
    target_constraints = get_leq_constraints(model, targets, flux_expr=flux_expr)
    desired_constraints = get_leq_constraints(model, desired, flux_expr=flux_expr)
    del flux_expr

    # check whether all target/desired regions are feasible
    for i in range(len(targets)):
        with model as feas:
            feas.objective = model.problem.Objective(0.0)
            feas.add_cons_vars(target_constraints[i])
            feas.slim_optimize()
            if feas.solver.status != 'optimal':
                raise InfeasibleRegion('Target region '+str(i)+' is not feasible; solver status is: '+feas.solver.status)
    for i in range(len(desired)):
        with model as feas:
            feas.objective = model.problem.Objective(0.0)
            feas.add_cons_vars(desired_constraints[i])
            feas.slim_optimize()
            if feas.solver.status != 'optimal':
                raise InfeasibleRegion('Desired region'+str(i)+' is not feasible; solver status is: '+feas.solver.status)

    if include_model_bounds:
        integrate_model_bounds(model, targets, desired)

    if cuts is None:
        cuts = numpy.full(len(model.reactions), True, dtype=bool)
    else:
        cuts = numpy.asarray(cuts) # in case it was passed as list
    if exclude_boundary_reactions_as_cuts:
        for r in range(len(model.reactions)):
            if model.reactions[r].boundary:
                cuts[r] = False
    cuts[knock_in_idx] = False # knock-ins supersede cuts
    intervenable = cuts.copy() # needed for MCS expansion and sorting according to cost
    intervenable[knock_in_idx] = True
    if intervention_costs is not None:
        intervention_costs = numpy.asarray(intervention_costs) # in case it was passed as list

    compressed_model = None
    if results_cache_dir is None:
        fva_hash = None
    else:
        fva_hash = model.stoichiometry_hash_object.copy()
        # different from regular FVA because of additional settings for GLPK/coinor-cbc
        fva_hash.update(b" blocked reactions")
        if network_compression:
            compressed_model_hash = fva_hash.copy()
            compressed_model_hash.update(b"subsest compression"+pickle.dumps(fva_tolerance))
            compressed_model_file: Path = results_cache_dir / \
                (model.id+"_subsets_compressed_"+compressed_model_hash.hexdigest())
            if compressed_model_file.exists():
                try:
                    with open(compressed_model_file, "rb") as file:
                        (compressed_model, subT) = pickle.load(file)
                    compressed_model = cobra.io.model_from_dict(compressed_model)
                    print("Loaded compressed model from", str(compressed_model_file))
                except:
                    print("Failed to load compressed model from", str(compressed_model_file))
                    compressed_model = None

    if compressed_model is None:
        print("FVA to find blocked reactions...")
        with model as fva: # can be skipped when a compressed model is available
            # when include_model_bounds=False modify bounds so that only reversibilites are used?
            # fva.solver = 'glpk_exact' # too slow for large models
            fva.tolerance = fva_tolerance
            fva.objective = model.problem.Objective(0.0)
            if fva.problem.__name__ == 'optlang.glpk_interface':
                # should emulate setting an optimality tolerance (which GLPK simplex does not have)
                fva.solver.configuration._smcp.meth = GLP_DUAL
                fva.solver.configuration._smcp.tol_dj = fva_tolerance
            elif fva.problem.__name__ == 'optlang.coinor_cbc_interface':
                fva.solver.problem.opt_tol = fva_tolerance
            fva_res = flux_variability_analysis(fva, fraction_of_optimum=0.0, processes=1, results_cache_dir=results_cache_dir,
                                fva_hash=fva_hash)

    kn = None
    if network_compression:
        if compressed_model is None:
            compressed_model = model.copy() # preserve the original model
            # integrate FVA bounds and flip reactions where necessary
            for i in range(fva_res.values.shape[0]): # assumes the FVA results are ordered same as the model reactions
                if abs(fva_res.values[i, 0]) > fva_tolerance: # resolve with glpk_exact?
                    compressed_model.reactions[i].lower_bound = fva_res.values[i, 0]
                else:
                    compressed_model.reactions[i].lower_bound = 0
                if abs(fva_res.values[i, 1]) > fva_tolerance: # resolve with glpk_exact?
                    compressed_model.reactions[i].upper_bound = fva_res.values[i, 1]
                else:
                    compressed_model.reactions[i].upper_bound = 0
            # rows of subT are the reactions, columns the subsets
            print("Network compression...")
            subT = efmtool4cobra.compress_model_sympy(compressed_model, protected_reactions=knock_in_idx)
            if results_cache_dir is not None:
                try:
                    with open(compressed_model_file, "wb") as file:
                        pickle.dump((compressed_model_to_dict(compressed_model), subT), file)
                    print("Saved compressed model to", str(compressed_model_file))
                except:
                    print("Failed to save compressed model to", str(compressed_model_file))
        model_reactions = model.reactions.list_attr("id")
        model = compressed_model
        for r in model.reactions:
            if len(r.subset_rxns) > 1:
                r.subset_id = "{"+"|".join(model_reactions[i] for i in r.subset_rxns)+"}"
            else:
                r.subset_id = r.id
        del model_reactions
        if results_cache_dir is not None:
            model.set_reaction_hashes()
            model.set_stoichiometry_hash_object()
        stoich_mat = cobra.util.array.create_stoichiometric_matrix(model, array_type='lil')
        if use_kn_in_dual:
            kn = efmtool4cobra.jRatMat2sparseFloat(
                    efmtool4cobra.kernel(efmtool4cobra.get_jRatMat_stoichmat(model)))
        targets = [[T@subT, t] for T, t in targets]
        # as a result of compression empty constraints can occur (e.g. limits on reactions that turn out to be blocked)
        for i in range(len(targets)): # remove empty target constraints
            keep = numpy.any(targets[i][0], axis=1)
            targets[i][0] = targets[i][0][keep, :]
            targets[i][1] = targets[i][1][keep]
        desired = [[D@subT, d] for D, d in desired]
        if results_cache_dir is not None:
            desired_hash_value = [None] * len(desired)
        for i in range(len(desired)): # remove empty desired constraints
            keep = numpy.any(desired[i][0], axis=1)
            desired[i][0] = desired[i][0][keep, :]
            desired[i][1] = desired[i][1][keep]
            if results_cache_dir is not None:
                # not optimal as it integrates the matrix type into the hash value
                desired_hash_value[i] = hashlib.md5(pickle.dumps(desired[i])).digest()
        cuts = numpy.any(subT[cuts, :], axis=0)
        knock_in_idx = [numpy.where(subT[i, :])[0][0] for i in knock_in_idx]
        if intervention_costs is not None:
            iv_cost_uncompressed = intervention_costs
            intervention_costs = numpy.zeros(subT.shape[1])
            for i in range(subT.shape[1]):
                idx = numpy.where(subT[:, i])[0]
                if len(idx > 0):
                    intervention_costs[i] = min(iv_cost_uncompressed[idx])
    else:
        stoich_mat = cobra.util.array.create_stoichiometric_matrix(model, array_type='lil')
        blocked_rxns = []
        for i in range(fva_res.values.shape[0]):
            if fva_res.values[i, 0] >= -fva_tolerance and fva_res.values[i, 1] <= fva_tolerance:
                blocked_rxns.append(fva_res.index[i])
                cuts[i] = False
        print("Found", len(blocked_rxns), "blocked reactions:\n", blocked_rxns)

    rev = [r.lower_bound < 0 for r in model.reactions] # use this as long as there might be irreversible backwards only reactions
    # add FVA bounds for desired
    desired_constraints= get_leq_constraints(model, desired)
    if len(desired) > 0:
        print("Running FVA for desired regions...")
    for i in range(len(desired)):
        with model as fva_desired:
            fva_desired.tolerance = fva_tolerance
            fva_desired.objective = model.problem.Objective(0.0)
            if fva_desired.problem.__name__ == 'optlang.glpk_interface':
                # should emulate setting an optimality tolerance (which GLPK simplex does not have)
                fva_desired.solver.configuration._smcp.meth = GLP_DUAL
                fva_desired.solver.configuration._smcp.tol_dj = fva_tolerance
            elif fva_desired.problem.__name__ == 'optlang.coinor_cbc_interface':
                fva_desired.solver.problem.opt_tol = fva_tolerance
            fva_desired.add_cons_vars(desired_constraints[i]) # need hash for this
            if results_cache_dir is None:
                fva_hash = None
            else:
                fva_hash = model.stoichiometry_hash_object.copy()
                fva_hash.update(desired_hash_value[i])
            fva_res = flux_variability_analysis(fva_desired, fraction_of_optimum=0.0, processes=1,
                                     results_cache_dir=results_cache_dir, fva_hash=fva_hash)
            # make tiny FVA values zero
            fva_res.values[numpy.abs(fva_res.values) < fva_tolerance] = 0
            essential = numpy.where(numpy.logical_or(fva_res.values[:, 0] > fva_tolerance, fva_res.values[:, 1] < -fva_tolerance))[0]
            print(len(essential), "essential reactions in desired region", i)
            cuts[essential] = False
            desired[i] = (desired[i][0], desired[i][1], fva_res.values[:, 0], fva_res.values[:, 1])
            
    optlang_interface = model.problem
    if optlang_interface.Constraint._INDICATOR_CONSTRAINT_SUPPORT and bigM == 0:
        bigM = 0.0
        print("Using indicators.")
    else:
        bigM = 1000.0
        print("Using big M.")

    e = cMCS_enumerator.ConstrainedMinimalCutSetsEnumerator(optlang_interface, stoich_mat, rev, targets,
            desired=desired, bigM=bigM, threshold=0.1, kn=kn,
            cuts=cuts, intervention_costs=intervention_costs, knock_in_idx=knock_in_idx,
            split_reversible_v=not network_compression, irrev_geq=kn is None)
    if enum_method == 3:
        if optlang_interface.__name__ == 'optlang.cplex_interface':
            e.model.problem.parameters.mip.tolerances.mipgap.set(0.98)
        elif optlang_interface.__name__ == 'optlang.gurobi_interface':
            e.model.problem.Params.MipGap = 0.98
        elif optlang_interface.__name__ == 'optlang.glpk_interface':
            e.model.configuration._iocp.mip_gap = 0.98
        elif optlang_interface.__name__ == 'optlang.coinor_cbc_interface':
            e.model.problem.max_solutions = 1 # stop with first feasible solutions
        else:
            print('No method implemented for this solver to stop with a suboptimal incumbent, will behave like enum_method 1.')
    elif enum_method == 4:
        e.model.configuration.verbosity = 3
    # if optlang_interface.__name__ == 'optlang.coinor_cbc_interface':
    #    e.model.problem.threads = -1 # activate multithreading
    
    e.evs_sz_lb = 1 # feasibility of all targets has been checked
    if optlang_interface.__name__ == 'optlang.glpk_interface':
        e.model.configuration._smcp.tol_dj = mip_opt_tol
    else:
        e.model.configuration.tolerances.optimality = mip_opt_tol
    e.model.configuration.tolerances.feasibility = mip_feas_tol
    e.model.configuration.tolerances.integrality = mip_int_tol
    if set_mip_parameters_callback is not None:
        set_mip_parameters_callback(e.model)
    #info = dict()
    mcs, err_val = e.enumerate_mcs(max_mcs_size=max_mcs_size, max_mcs_num=max_mcs_num, enum_method=enum_method,
                            model=model, targets=targets, desired=desired, timeout=timeout, #info=info,
                            reaction_display_attr='subset_id' if network_compression else 'id')
    #print(f"MILP time: {info['time']:.2f} seconds")
    if network_compression:
        xsubT= subT.copy()
        xsubT[numpy.logical_not(intervenable), :] = 0 # only expand to reactions that are intervenable within a given subset
        mcs = expand_mcs(mcs, xsubT)
        if intervention_costs is not None:
            intervention_costs = iv_cost_uncompressed
    elif enum_method == 4:
        mcs = [tuple(sorted(m)) for m in mcs]

    if intervention_costs is not None and numpy.any(intervention_costs[intervenable] != 1):
        print("Sorting and filtering interventions according to their cost")
        mcs = [(m, sum(intervention_costs[list(m)])) for m in mcs]
        mcs = [(m, c) for m,c in mcs if c <= max_mcs_size]
        mcs = sorted(mcs, key=lambda x: x[1])
        mcs = [m for m,_ in mcs]
    elif enum_method == 3 or enum_method == 4: # sort according to intervention size
        mcs = sorted(mcs, key=len)
    # if some intervention costs are 0 supersets with these interventions may or may
    # not be present depending on the enumeration scheme

    return mcs, err_val

def stoich_mat2cobra(stoich_mat, irrev_reac):
    model = cobra.Model('stoich_mat')
    model.add_metabolites([cobra.Metabolite('M'+str(i)) for i in range(stoich_mat.shape[0])])
    model.add_reactions([cobra.Reaction('R'+str(i)) for i in range(stoich_mat.shape[1])])
    for r in range(stoich_mat.shape[1]):
        if irrev_reac[r] == 0:
            model.reactions[r].lower_bound = cobra.Configuration().lower_bound
        model.reactions[r].add_metabolites({model.metabolites[m]: stoich_mat[m, r] for m in numpy.nonzero(stoich_mat[:, r])[0]})
    return model

def equations_to_matrix(model, equations):
    # deprecated
    # add option to use names instead of ids
    # allow equations to be a list of lists
    dual = cobra.Model()
    reaction_ids = [r.id for r in model.reactions]
    dual.add_metabolites([cobra.Metabolite(r) for r in reaction_ids])
    for i in range(len(equations)):
        r = cobra.Reaction("R"+str(i)) 
        dual.add_reactions([r])
        r.build_reaction_from_string('=> '+equations[i])
    dual = cobra.util.array.create_stoichiometric_matrix(dual, array_type='DataFrame')
    if numpy.all(dual.index.values == reaction_ids):
        return dual.values.transpose()
    else:
        raise RuntimeError("Index order was not preserved.")
