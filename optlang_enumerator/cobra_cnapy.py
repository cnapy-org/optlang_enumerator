import cobra
import hashlib
import pickle

"""
Extensions of cobra classes used in optlang_enumerator
"""

def set_hash(self):
    # float(s): make stable with respect to the number type  of the stoiciometric coefficients (1 != 1.0)
    self.hash_value = hashlib.md5(pickle.dumps(tuple((sorted((m.id, float(s)) for m, s in self.metabolites.items()),
                            self.lower_bound, self.upper_bound)))).digest()
cobra.Reaction.hash_value = None
cobra.Reaction.set_hash = set_hash

def set_reaction_hashes(self):
    for r in self.reactions:
        r.set_hash()

cobra.Model.set_reaction_hashes = set_reaction_hashes
"""
stoichiometry_hash() only sets up the initial hash object,
further data can be added with update() and it can 
be processed with digest() or hexdigest()
"""
cobra.Model.stoichiometry_hash = lambda self: hashlib.md5(pickle.dumps(self.reactions.list_attr("hash_value")))
