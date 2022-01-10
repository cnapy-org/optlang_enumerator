import optlang
import optlang_enumerator.mcs_computation as mcs_computation
import optlang_enumerator.cobra_cnapy as cobra_cnapy

def test():
    ex = cobra_cnapy.CNApyModel.read_sbml_model(r"tests/metatool_example_no_ext.xml")
    # currently tests with GLPK only because it is automatically installed with optlang
    ex.solver = 'glpk'
    for r in ex.reactions: # make all reactions bounded for cobrapy FVA
        if r.lower_bound == -float("inf"):
            r.lower_bound = -1000
        if r.upper_bound == float("inf"):
            r.upper_bound = 1000
    reac_id = ex.reactions.list_attr('id')
    reac_id_symbols = mcs_computation.get_reac_id_symbols(reac_id)
    target = [[("Pyk", ">=", 1), ("Pck", ">=", 1)]]
    target = [mcs_computation.relations2leq_matrix(
                mcs_computation.parse_relations(t, reac_id_symbols=reac_id_symbols), reac_id) for t in target]

    mcs, err_val = mcs_computation.compute_mcs(ex, target, enum_method=1, network_compression=True, max_mcs_size=None)
    assert(err_val == 0)
    assert(len(mcs) == 84)
    assert(all(mcs_computation.check_mcs(ex, target[0], mcs, optlang.interface.INFEASIBLE)))


    mcs2, err_val = mcs_computation.compute_mcs(ex, target, enum_method=3, network_compression=False, max_mcs_size=None)
    assert(err_val == 0)
    assert(set(mcs) == set(mcs2))

    desired = [[("AspCon", ">=", 1)]]
    desired = [mcs_computation.relations2leq_matrix(
                mcs_computation.parse_relations(d, reac_id_symbols=reac_id_symbols), reac_id) for d in desired]

    mcs3, err_val = mcs_computation.compute_mcs(ex, target, desired=desired, enum_method=1, network_compression=True, max_mcs_size=None)
    assert(err_val == 0)
    assert(len(mcs3) == 52)
    assert(all(mcs_computation.check_mcs(ex, target[0], mcs3, optlang.interface.INFEASIBLE)))
    assert(all(mcs_computation.check_mcs(ex, desired[0], mcs3, optlang.interface.OPTIMAL)))

    mcs4, err_val = mcs_computation.compute_mcs(ex, target, desired=desired, enum_method=3, network_compression=False, max_mcs_size=None)
    assert(err_val == 0)
    assert(set(mcs3) == set(mcs4))
