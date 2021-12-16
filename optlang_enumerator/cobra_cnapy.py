import cobra
from cobra.util.context import get_context
from functools import partial
import hashlib
import pickle

"""
Extensions of cobra classes used in optlang_enumerator
"""

def set_hash_value(self):
    context = get_context(self)
    if context:
        context(partial(self.restore_hash_value, self._hash_value))
    # float(s): make stable with respect to the number type  of the stoiciometric coefficients (1 != 1.0)
    self._hash_value = hashlib.md5(pickle.dumps(tuple((sorted((m.id, float(s)) for m, s in self.metabolites.items()),
                            self.lower_bound, self.upper_bound)))).digest()

def get_hash_value(self):
    return self._hash_value

def restore_hash_value(self, hash_value):
    self._hash_value = hash_value

cobra.Reaction._hash_value = None
"""
hash_value takes only the reaction stoichiometry (via the metabolites)
and reaction bounds into account
"""
cobra.Reaction.hash_value = property(fset=None, fget=get_hash_value)
cobra.Reaction.set_hash_value = set_hash_value
cobra.Reaction.restore_hash_value = restore_hash_value

def set_reaction_hashes(self):
    for r in self.reactions:
        r.set_hash_value()

cobra.Model.set_reaction_hashes = set_reaction_hashes
"""
stoichiometry_hash() only sets up the initial hash object,
further data can be added with update() and it can 
be processed with digest() or hexdigest()
"""
cobra.Model.stoichiometry_hash = lambda self: hashlib.md5(pickle.dumps(self.reactions.list_attr("hash_value")))

cobra.Model._stoichiometry_hash_object = None
def set_stoichiometry_hash_object(self):
    context = get_context(self)
    if context:
        context(partial(self.restore_stoichiometry_hash_object, self._stoichiometry_hash_object))
    self._stoichiometry_hash_object = self.stoichiometry_hash()
    self._stoichiometry_hash_object.digest()

def get_stoichiometry_hash_object(self):
    return self._stoichiometry_hash_object

def restore_stoichiometry_hash_object(self, stoichiometry_hash_object):
    self._stoichiometry_hash_object= stoichiometry_hash_object

cobra.Model.set_stoichiometry_hash_object = set_stoichiometry_hash_object
"""
a copy() of the stoichiometry_hash_object can then be used for 
calculating problem-specific hashes
"""
cobra.Model.stoichiometry_hash_object = property(fset=None, fget=get_stoichiometry_hash_object)
cobra.Model.restore_stoichiometry_hash_object = restore_stoichiometry_hash_object

class CNApyModel(cobra.Model):
    @staticmethod
    def read_sbml_model(file_name):
        model: cobra.Model = cobra.io.read_sbml_model(file_name)
        model.set_reaction_hashes()
        model.set_stoichiometry_hash_object()
        """
        kludge because in COBRApy creating a model from a SBML file
        is not implemented as method of the Model class
        this is ugly but works because all new properties are added
        to cobra.Reaction and cobra.Model
        """
        model.__class__ = CNApyModel
        return model

    def __init__(self, id_or_model=None, name=None):
        super().__init__(id_or_model=id_or_model, name=name)
        if id_or_model is None:
            self.id = ""
        self.set_reaction_hashes()
        self.set_stoichiometry_hash_object()
